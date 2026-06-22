from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from core.schema import ExplanationRequest
from feature_extraction.extractor import TSFeatureExtractor
from feature_extraction.real_window_resolver import RealWindowResolver
from llm_interface.base import ExplanationService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask TS-Explain a Chinese anomaly-detection question.")
    parser.add_argument("question", nargs="?", help="中文问题。也可以用 --question 传入。")
    parser.add_argument("--question", dest="question_flag", help="中文问题。")
    parser.add_argument("--mode", default="rag-only", choices=["no-rag", "rag-only", "rag-ts"])
    parser.add_argument("--provider", default="offline", choices=["offline", "qwen", "deepseek"])
    parser.add_argument("--dataset", dest="dataset_name", default=None, help="数据集名，例如 SMD、MSL、PSM。")
    parser.add_argument("--model", dest="model_name", default=None, help="模型名，例如 DADA、Anomaly Transformer、CATCH。")
    parser.add_argument("--window-csv", default=None, help="可选：手动提供 CSV 窗口文件；启用后生成 TS-Fact Card。")
    parser.add_argument("--auto-window", action="store_true", help="自动从本地真实数据集中截取异常窗口；当前支持 SMD。")
    parser.add_argument("--window-length", type=int, default=256, help="自动窗口长度，默认 256。")
    parser.add_argument(
        "--window-strategy",
        default="max_label_density",
        choices=["max_label_density", "first_anomaly"],
        help="真实窗口选择策略：标签密度最高窗口或第一个异常窗口。",
    )
    parser.add_argument("--window-id", default="cli-window")
    parser.add_argument("--show-window-info", action="store_true", help="打印自动解析出的真实窗口位置。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = args.question_flag or args.question
    if not question:
        raise SystemExit('请提供问题，例如：python scripts/ask.py "为什么 DADA 在 SMD 上表现较好？" --dataset SMD --model DADA')
    if args.auto_window and args.window_csv:
        raise SystemExit("--auto-window 和 --window-csv 不能同时使用。")

    fact_card = None
    mode = args.mode
    if args.window_csv:
        import pandas as pd

        window = pd.read_csv(args.window_csv)
        fact_card = TSFeatureExtractor().extract(window, window_id=args.window_id)
        mode = "rag-ts"
    elif args.auto_window:
        if not args.dataset_name:
            raise SystemExit("使用 --auto-window 时必须同时提供 --dataset，例如 --dataset SMD。")
        resolver = RealWindowResolver()
        real_window, fact_card = resolver.extract_fact_card(
            args.dataset_name,
            model_name=args.model_name,
            window_length=args.window_length,
            strategy=args.window_strategy,
        )
        mode = "rag-ts"
        if args.show_window_info:
            print(
                f"[real-window] dataset={real_window.dataset_name}, start={real_window.start}, "
                f"end={real_window.end}, label_density={real_window.label_density:.2%}, "
                f"strategy={real_window.strategy}"
            )

    service = ExplanationService(provider_name=args.provider)
    answer = service.answer(
        ExplanationRequest(
            question=question,
            mode=mode,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            fact_card=fact_card,
        )
    )
    print(answer.answer)


if __name__ == "__main__":
    main()
