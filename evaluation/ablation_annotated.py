# 这是 evaluation/ablation.py 的逐行注释版，用于阅读学习。
# 它保留原脚本逻辑；如果直接运行，也会执行消融实验。

# 启用 Python 未来版本的注解行为，避免类型注解在运行时被立即求值。
from __future__ import annotations

# 导入 argparse，用于解析命令行参数，例如 --provider、--modes。
import argparse
# 导入 json，用于把实验结果写成 JSON / JSONL 文件。
import json
# 导入 random，用于固定 Python 标准库随机种子，保证实验可复现。
import random
# 导入 sys，用于在脚本直接运行时临时修改模块搜索路径。
import sys
# 导入 time，用于记录每道题的开始时间和运行耗时。
import time
# 从 pathlib 导入 Path，用面向对象方式处理文件路径。
from pathlib import Path
# 导入类型注解工具：List、Optional、Sequence 等只帮助读代码和静态检查。
from typing import Iterable, List, Optional, Sequence

# 导入 numpy，用于固定随机种子、生成合成时序窗口、计算均值等。
import numpy as np

# 如果这个文件是被 python evaluation/ablation_annotated.py 直接运行，而不是作为包导入。
if __package__ in {None, ""}:
    # 把项目根目录加入 Python 搜索路径，确保 core、evaluation 等包能被找到。
    sys.path.append(str(Path(__file__).resolve().parents[1]))

# 导入配置加载函数、YAML/JSON 配置读取函数、路径解析函数。
from core.config import load_project_config, load_yaml, resolve_path
# 导入解释请求结构和问题项结构。
from core.schema import ExplanationRequest, QuestionItem
# 导入总评估器，用于计算幻觉率、引用一致性和回答质量。
from evaluation.evaluator import Evaluator
# 导入时序特征提取器，用于为 rag-ts 模式构造 TS-Fact Card。
from feature_extraction.extractor import TSFeatureExtractor
# 导入解释服务，它会调用 RAG 检索、prompt 构造和 LLM provider。
from llm_interface.base import ExplanationService


# 定义命令行里使用的模式名和系统内部模式名之间的映射。
MODE_MAP = {
    # no_rag 输出目录名对应模块 3 里的 no-rag 模式。
    "no_rag": "no-rag",
    # rag_only 输出目录名对应模块 3 里的 rag-only 模式。
    "rag_only": "rag-only",
    # rag_ts 输出目录名对应模块 3 里的 rag-ts 模式。
    "rag_ts": "rag-ts",
}


# 定义读取标准化问题库的函数，输入是问题库文件路径。
def load_question_bank(path: str | Path) -> List[QuestionItem]:
    # 使用项目里的 load_yaml 读取 YAML/JSON 风格的问题库。
    data = load_yaml(path)
    # 把 questions 列表里的每个字典转换为 QuestionItem 数据对象。
    return [QuestionItem(**item) for item in data.get("questions", [])]


# 定义一个函数，为每道题生成合成的 TS-Fact Card。
def synthetic_fact_card(question: QuestionItem, seed: int = 42):
    # 根据全局 seed 和问题 id 构造随机数生成器，让每道题的合成窗口稳定可复现。
    #ord 把字符转为数字
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in question.id))
    # 设置合成窗口长度为 96 个时间点。
    length = 96
    # 生成时间索引 t，取值为 0 到 95。
    t = np.arange(length)
    # 把三个变量堆叠成一个形状为 96 x 3 的多变量时间序列窗口。
    base = np.stack(
        # 下面这个列表里每个元素都是一个变量的一维时间序列。
        [
            # 第一个变量 load：带轻微上升趋势和 24 步周期。
            0.02 * t + np.sin(2 * np.pi * t / 24),
            # 第二个变量 pressure：带 16 步周期。
            np.cos(2 * np.pi * t / 16),
            # 第三个变量 noise：均值为 0、标准差为 0.2 的随机噪声。
            rng.normal(0, 0.2, length),
        ],
        # 指定沿列方向堆叠，得到每列一个变量。
        axis=1,
    )
    # 在 pressure 变量的第 55 到 62 个时间点人为加入异常抬升。
    base[55:63, 1] += 3.0
    # 根据每个时间点相对窗口均值的偏离程度，构造简易异常分数。
    scores = np.mean(np.abs(base - base.mean(axis=0)), axis=1)
    # 创建 TSFeatureExtractor，用于提取趋势、周期、变点、贡献度等时序特征。
    extractor = TSFeatureExtractor()
    # 返回提取出的 TS-Fact Card；变量名固定为 load、pressure、noise。
    return extractor.extract(base, window_id=f"synthetic-{question.id}", anomaly_scores=scores, feature_names=["load", "pressure", "noise"])


# 定义消融实验主函数。
def run_ablation(
    # 使用 * 强制后面的参数必须以关键字形式传入，避免调用时混淆。
    *,
    # 可选的配置文件路径；不传则使用 config/config.yaml。
    config_path: str | Path | None = None,
    # 指定 LLM provider，可取 offline、qwen、deepseek。
    provider_name: str = "offline",
    # 可选限制问题数量，调试时可用 --limit 只跑前几题。
    limit: Optional[int] = None,
    # 可选指定只跑哪些模式，例如只跑 no_rag 和 rag_ts。
    modes: Optional[Sequence[str]] = None,
# 函数返回一个字典，里面是每个模式的汇总指标。
) -> dict:
    # 加载项目配置，包括随机种子、结果目录、问题库路径等。
    config = load_project_config(config_path)
    # 从配置里读取随机种子，如果没有配置则默认使用 42。
    seed = int(config.get("project", {}).get("seed", 42))
    # 固定 Python 标准库 random 的随机种子。
    random.seed(seed)
    # 固定 numpy 的随机种子。
    np.random.seed(seed)
    # 获取项目根目录路径。
    root = Path(config["_project_root"])
    # 根据配置解析问题库文件路径。
    question_path = resolve_path(config.get("evaluation", {}).get("question_bank", "experiments/question_bank.yaml"), root)
    # 读取标准化问题库，得到 QuestionItem 列表。
    questions = load_question_bank(question_path)
    # 如果用户设置了 limit，就只保留前 limit 道题。
    if limit:
        # 截断问题列表，用于快速调试。
        questions = questions[:limit]

    # 创建解释服务；内部会根据 provider_name 调用 offline、Qwen 或 DeepSeek。
    service = ExplanationService(config_path, provider_name=provider_name)
    # 创建评估器；用于对每条回答计算指标。
    evaluator = Evaluator(config_path)
    # 解析实验结果根目录，默认是 experiments/results。
    results_root = resolve_path(config.get("evaluation", {}).get("results_dir", "experiments/results"), root)
    # 确保结果目录存在；如果不存在就创建。
    results_root.mkdir(parents=True, exist_ok=True)
    # 初始化汇总结果字典，后面按模式写入统计指标。
    summary = {}

    # 如果用户传了 modes，就只跑用户指定的模式；否则默认跑 MODE_MAP 的全部模式。
    selected_modes = list(modes) if modes else list(MODE_MAP.keys())
    # 找出用户传入但不在 MODE_MAP 里的非法模式名。
    invalid_modes = [mode for mode in selected_modes if mode not in MODE_MAP]
    # 如果存在非法模式，就直接抛出错误，避免悄悄跑错实验。
    if invalid_modes:
        # 报错信息会列出所有未知模式。
        raise ValueError(f"Unknown ablation mode(s): {', '.join(invalid_modes)}")

    # 逐个运行选中的消融模式。
    for output_dir in selected_modes:
        # 根据输出目录名找到模块 3 使用的真实模式名。
        mode = MODE_MAP[output_dir]
        # 为当前模式创建独立结果目录，例如 experiments/results/rag_ts。
        mode_dir = results_root / output_dir
        # 确保当前模式的结果目录存在。
        mode_dir.mkdir(parents=True, exist_ok=True)
        # 当前模式的回答结果文件，每行是一条 JSON。
        out_path = mode_dir / "answers.jsonl"
        # 当前模式的进度日志文件，用来观察跑到第几题。
        progress_path = mode_dir / "progress.jsonl"
        # 当前模式下的所有记录会先保存在 records 里，最后统计平均指标。
        records = []
        # 同时打开答案文件和进度文件；使用 w 表示每次运行都会覆盖旧结果。
        with out_path.open("w", encoding="utf-8") as out_f, progress_path.open("w", encoding="utf-8") as progress_f:
            # enumerate 给每个问题编号，从 1 开始计数，便于显示 1/64。
            for idx, question in enumerate(questions, start=1):
                # 记录当前问题开始处理的时间戳。
                started = time.time()
                # 在终端打印当前进度，例如 [rag_ts] 1/64 MC001 start。
                print(f"[{output_dir}] {idx}/{len(questions)} {question.id} start", flush=True)
                # 向进度文件写入 start 事件。
                progress_f.write(
                    # 把进度信息序列化成 JSON 字符串。
                    json.dumps(
                        # 这里是单条进度记录的字典。
                        {
                            # 当前运行的是哪个输出模式目录。
                            "mode": output_dir,
                            # 当前题号。
                            "index": idx,
                            # 总题数。
                            "total": len(questions),
                            # 问题 id，例如 MC001。
                            "id": question.id,
                            # 事件类型：start 表示开始处理。
                            "event": "start",
                            # 当前开始时间戳，保留三位小数。
                            "time": round(started, 3),
                        },
                        # ensure_ascii=False 保证中文不会被转义成 unicode 编码。
                        ensure_ascii=False,
                    )
                    # 每条 JSONL 记录以换行结尾。
                    + "\n"
                )
                # 立即刷新进度文件，防止程序中途卡住时看不到最新进度。
                progress_f.flush()
                # 使用 try 捕获单题错误，避免一道题失败导致整个实验中断。
                try:
                    # rag-ts 模式需要 TS-Fact Card；其他模式不需要，所以给 None。
                    fact_card = synthetic_fact_card(question, seed) if mode == "rag-ts" else None
                    # 调用模块 3 的解释服务，生成回答。
                    answer = service.answer(
                        # 构造 ExplanationRequest，把问题、模式、数据集、模型和特征卡传进去。
                        ExplanationRequest(
                            # 用户问题文本。
                            question=question.question,
                            # 当前运行模式，例如 no-rag 或 rag-ts。
                            mode=mode,
                            # 问题指定的数据集名，可用于 RAG 元数据过滤。
                            dataset_name=question.dataset_name,
                            # 问题指定的模型名，可用于 RAG 元数据过滤。
                            model_name=question.model_name,
                            # rag-ts 模式下传入 TS-Fact Card；其他模式是 None。
                            fact_card=fact_card,
                        )
                    )
                    # 对回答进行评估，得到幻觉率、引用一致性和回答质量。
                    metrics = evaluator.evaluate(answer)
                    # 计算当前问题总耗时，保留三位小数。
                    elapsed = round(time.time() - started, 3)
                    # 组装当前问题的完整实验记录。
                    record = {
                        # 问题 id。
                        "id": question.id,
                        # 问题类别，例如 model_comparison。
                        "category": question.category,
                        # 原始问题文本。
                        "question": question.question,
                        # LLM 生成的回答文本。
                        "answer": answer.answer,
                        # 回答中出现的引用编号列表。
                        "citations": answer.citations,
                        # 当前回答的评估指标。
                        "metrics": metrics,
                        # 当前题运行耗时。
                        "elapsed_seconds": elapsed,
                    }
                    # 把当前记录加入内存列表，便于最后计算平均值。
                    records.append(record)
                    # 把当前记录写入 answers.jsonl。
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    # 立即刷新答案文件，确保中途停止也能保留已完成题目。
                    out_f.flush()
                    # 向进度文件写入 done 事件。
                    progress_f.write(
                        # 把完成事件序列化为 JSON。
                        json.dumps(
                            # 这里是完成事件的具体内容。
                            {
                                # 当前模式目录名。
                                "mode": output_dir,
                                # 当前题号。
                                "index": idx,
                                # 总题数。
                                "total": len(questions),
                                # 问题 id。
                                "id": question.id,
                                # 事件类型：done 表示成功完成。
                                "event": "done",
                                # 当前题耗时。
                                "elapsed_seconds": elapsed,
                                # 完成时刻的时间戳。
                                "time": round(time.time(), 3),
                            },
                            # 保持中文原样输出。
                            ensure_ascii=False,
                        )
                        # JSONL 每条记录以换行结尾。
                        + "\n"
                    )
                    # 立即刷新进度文件。
                    progress_f.flush()
                    # 在终端打印当前题完成和耗时。
                    print(f"[{output_dir}] {idx}/{len(questions)} {question.id} done in {elapsed}s", flush=True)
                # 捕获当前问题处理过程中的任何异常。
                except Exception as exc:
                    # 即使出错，也计算当前题耗时。
                    elapsed = round(time.time() - started, 3)
                    # 组装错误记录，保证 answers.jsonl 中也有这一题的信息。
                    record = {
                        # 问题 id。
                        "id": question.id,
                        # 问题类别。
                        "category": question.category,
                        # 原始问题。
                        "question": question.question,
                        # 出错时没有回答，所以 answer 为空字符串。
                        "answer": "",
                        # 出错时没有引用。
                        "citations": [],
                        # 出错时没有可用指标。
                        "metrics": {},
                        # 当前题耗时。
                        "elapsed_seconds": elapsed,
                        # 错误类型和错误信息。
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    # 把错误记录加入 records，便于最后统计 error_count。
                    records.append(record)
                    # 把错误记录写入 answers.jsonl。
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    # 立即刷新答案文件。
                    out_f.flush()
                    # 向进度文件写入 error 事件。
                    progress_f.write(
                        # 把错误事件序列化成 JSON。
                        json.dumps(
                            # 这里是错误事件的具体内容。
                            {
                                # 当前模式目录名。
                                "mode": output_dir,
                                # 当前题号。
                                "index": idx,
                                # 总题数。
                                "total": len(questions),
                                # 问题 id。
                                "id": question.id,
                                # 事件类型：error 表示该题失败。
                                "event": "error",
                                # 当前题耗时。
                                "elapsed_seconds": elapsed,
                                # 错误类型和错误消息。
                                "error": f"{type(exc).__name__}: {exc}",
                                # 出错时刻的时间戳。
                                "time": round(time.time(), 3),
                            },
                            # 保持中文原样输出。
                            ensure_ascii=False,
                        )
                        # JSONL 每条记录以换行结尾。
                        + "\n"
                    )
                    # 立即刷新进度文件。
                    progress_f.flush()
                    # 在终端打印错误信息，方便用户看到卡在哪道题。
                    print(f"[{output_dir}] {idx}/{len(questions)} {question.id} error after {elapsed}s: {exc}", flush=True)
        # 只保留真正有评估指标的记录，过滤掉出错记录。
        valid_records = [r for r in records if r.get("metrics") and "hallucination" in r["metrics"]]
        # 计算平均幻觉率；如果没有有效记录，则设为 0。
        avg_h = float(np.mean([r["metrics"]["hallucination"]["hallucination_rate"] for r in valid_records])) if valid_records else 0.0
        # 计算平均回答质量分；如果没有有效记录，则设为 0。
        avg_q = float(np.mean([r["metrics"]["answer_quality"]["score"] for r in valid_records])) if valid_records else 0.0
        # 计算平均引用一致性；如果没有有效记录，则设为 0。
        avg_c = float(np.mean([r["metrics"]["citation_consistency_rate"] for r in valid_records])) if valid_records else 0.0
        # 把当前模式的汇总指标写入 summary 字典。
        summary[output_dir] = {
            # 当前模式总共处理了多少条记录，包括错误记录。
            "question_count": len(records),
            # 当前模式有多少条成功评估的记录。
            "valid_question_count": len(valid_records),
            # 当前模式有多少条错误记录。
            "error_count": len(records) - len(valid_records),
            # 平均幻觉率，小数形式。
            "avg_hallucination_rate": round(avg_h, 4),
            # 平均幻觉率，百分比形式。
            "avg_hallucination_pct": round(avg_h * 100, 2),
            # 平均引用一致性，小数形式。
            "avg_citation_consistency_rate": round(avg_c, 4),
            # 平均引用一致性，百分比形式。
            "avg_citation_consistency_pct": round(avg_c * 100, 2),
            # 平均回答质量分。
            "avg_quality_score": round(avg_q, 4),
            # 当前模式 answers.jsonl 的路径。
            "path": str(out_path),
        }
    # 定义汇总 JSON 文件路径。
    summary_path = results_root / "ablation_summary.json"
    # 把 summary 写成格式化 JSON 文件。
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 初始化 Markdown 汇总内容。
    md = ["# Ablation Summary", ""]
    # 遍历每个模式的汇总指标，用于生成 Markdown 列表。
    for mode, item in summary.items():
        # 添加当前模式的一行简短指标说明。
        md.append(
            # 这一行包含题目数量、引用一致性、幻觉率和质量分。
            f"- {mode}: questions={item['question_count']}, citation_consistency={item['avg_citation_consistency_pct']}%, hallucination={item['avg_hallucination_pct']}%, avg_quality_score={item['avg_quality_score']}"
        )
    # 如果同时跑了 no_rag 和 rag_ts，就额外生成一条简历可用的对比句。
    if "no_rag" in summary and "rag_ts" in summary:
        # 取出 no_rag 的汇总指标。
        no_rag = summary["no_rag"]
        # 取出 rag_ts 的汇总指标。
        rag_ts = summary["rag_ts"]
        # 向 Markdown 中追加简历指标部分。
        md.extend(
            # extend 接收一个列表，会把列表里的多行追加到 md。
            [
                # 空行，用于 Markdown 排版。
                "",
                # 二级标题，说明下面是简历指标。
                "## Resume Metric",
                # 拼接一句中文对比总结。
                (
                    # 句子前半段说明引用一致性的变化。
                    "相较 No-RAG 模式，RAG+TS-Feature 的引用一致性由 "
                    # 插入 no_rag 和 rag_ts 的引用一致性百分比。
                    f"{no_rag['avg_citation_consistency_pct']}% 提升至 {rag_ts['avg_citation_consistency_pct']}%，"
                    # 插入 no_rag 和 rag_ts 的幻觉率百分比。
                    f"幻觉率由 {no_rag['avg_hallucination_pct']}% 降至 {rag_ts['avg_hallucination_pct']}%。"
                ),
            ]
        )
    # 把 Markdown 汇总写入 ablation_summary.md。
    (results_root / "ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    # 返回 summary，方便命令行 main 打印或其他代码调用。
    return summary


# 定义命令行入口函数。
def main() -> None:
    # 创建命令行参数解析器，并给出脚本描述。
    parser = argparse.ArgumentParser(description="Run TS-Explain ablation modes.")
    # 添加 --config 参数，允许用户指定配置文件。
    parser.add_argument("--config", default=None)
    # 添加 --provider 参数，指定使用 offline、qwen 还是 deepseek。
    parser.add_argument("--provider", default="offline", choices=["offline", "qwen", "deepseek"])
    # 添加 --limit 参数，允许用户只跑前 N 道题。
    parser.add_argument("--limit", type=int, default=None)
    # 添加 --modes 参数，允许用户指定要跑的消融模式列表。
    parser.add_argument(
        # 参数名是 --modes。
        "--modes",
        # nargs="+" 表示该参数后面可以跟一个或多个模式名。
        nargs="+",
        # 默认值 None 表示不指定时跑全部模式。
        default=None,
        # choices 限制模式名必须来自 MODE_MAP 的 key。
        choices=list(MODE_MAP.keys()),
        # help 是命令行帮助文本。
        help="Ablation modes to run, e.g. --modes no_rag rag_ts",
    )
    # 解析命令行参数。
    args = parser.parse_args()
    # 调用 run_ablation 执行实验，并传入命令行参数。
    summary = run_ablation(config_path=args.config, provider_name=args.provider, limit=args.limit, modes=args.modes)
    # 把汇总结果打印到终端，使用中文友好的 JSON 输出。
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 如果这个文件被直接运行，则执行 main；如果被 import，则不自动运行。
if __name__ == "__main__":
    # 调用命令行入口函数。
    main()
