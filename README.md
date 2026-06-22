# TS-Explain

TS-Explain 是一个面向多变量时间序列异常检测的 RAG 解释系统。它把论文/README 知识、实验结果、数据集元数据、超参配置、模型代码摘要和窗口级时序特征组织成可追溯证据，用于回答中文自然语言问题。

## 当前交付内容

- RAG 知识库：支持五类来源接入、增量 upsert、ChromaDB 后端和本地向量索引兜底。
- 混合检索：向量相似度、BM25 关键词、元数据过滤、轻量 rerank。
- TS-Fact Card：趋势、周期、异常剖面、变量相关性、变点、残差模式和变量贡献。
- LLM 解释：支持 `no-rag`、`rag-only`、`rag-ts` 三种模式；Qwen、DeepSeek 和离线模板 provider。
- 评估体系：自动幻觉率近似、引用一致性检查、启发式质量分和人工评估指南。
- 问题库：64 道标准化问题，覆盖模型对比、单模型归因、窗口解释、数据集分析。

## 快速开始

在项目根目录 `C:\Users\Jack\Documents\Timeseries\TS-Explain` 运行：

```powershell
..\.venv\Scripts\python.exe -m knowledge_base.build_kb --reset --backend local
..\.venv\Scripts\python.exe -m evaluation.ablation --provider offline
..\.venv\Scripts\python.exe -m unittest discover -s tests
```

PowerShell 中如果不喜欢相对路径，可以直接使用：

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m knowledge_base.build_kb --reset --backend local
```

直接提问建议使用命令行入口，避免 Windows PowerShell 管道把中文转成问号：

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe .\scripts\ask.py "为什么 Anomaly Transformer 在 SMD 上表现较好？" --mode rag-only --dataset SMD
```

如果要解释一个真实 SMD 异常窗口，可以让系统自动从本地 `SMD_test.npy` 和 `SMD_test_label.npy` 中截取异常标签密度最高的窗口：

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe .\scripts\ask.py "DADA 在 SMD 的真实异常窗口为什么被判为异常？" --provider deepseek --dataset SMD --model DADA --auto-window --window-length 256 --show-window-info
```

注意：当前 `--auto-window` 使用真实 SMD 标签定位窗口，不代表 DADA 的逐点 anomaly score；DADA 的模型级证据仍由 RAG 检索实验结果和模型说明提供。

## 安装完整依赖

当前代码在缺少 ChromaDB、statsmodels、ruptures、dashscope、openai 时也能离线运行。要启用完整后端：

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

配置 API key：

```powershell
$env:DASHSCOPE_API_KEY="你的通义千问 key"
$env:DEEPSEEK_API_KEY="你的 DeepSeek key"
```

运行真实 provider：

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m evaluation.ablation --provider qwen
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m evaluation.ablation --provider deepseek
```

## 目录说明

- `config/`：全局配置、RAG 分块策略、时序特征参数。
- `knowledge_base/`：ingest、chunk、embedding、store、retrieval 和构建入口。
- `feature_extraction/`：统计特征、相关性、异常剖面、归因和 prompt 文本生成。
- `llm_interface/`：provider、prompt、结构化输出解析和三模式解释服务。
- `evaluation/`：消融 runner、自动指标和人工评估说明。
- `experiments/`：问题库、消融配置和输出结果。

## 引用规范

检索结果会被格式化为 `[S0001]` 形式。Prompt 要求 LLM 对事实性陈述标注引用；评估模块会检查回答引用是否存在于本次检索结果，并用词面重叠给出半自动一致性提示。生产环境可以把 `citation_fidelity.py` 中的轻量检查替换为中文 NLI 或 bge-reranker-v2-m3。

## 重要说明

离线 provider 只用于跑通流程和生成可复现实验骨架，不代表真实大模型能力。要得到正式消融结果，请安装依赖、构建 ChromaDB 或本地索引，并使用 Qwen/DeepSeek API 重新运行 `evaluation.ablation`。

## Multi-Agent Workflow Prototype

本项目新增了一个轻量 Multi-Agent Workflow 原型，不替换原有 RAG pipeline，而是在原有模块外层增加可追踪的 Agent 编排。

普通 RAG pipeline 的流程是：

```text
question -> retrieve -> generate -> evaluate
```

Agent workflow 的流程是：

```text
Supervisor
  -> QueryRouter: decide mode and whether a real window is needed
  -> EvidenceRetriever: call the existing Retriever
  -> WindowAnalyzer: optionally call RealWindowResolver for SMD real window
  -> AnswerWriter: call the existing ExplanationService / provider
  -> CitationAuditor: call the existing Evaluator
  -> MemoryCompressor: summarize evidence, decisions, and failures
```

新增目录：

- `agent_runtime/`: Pydantic schema, Supervisor, agents, workflow CLI, JSONL run store.
- `memory/`: compressed memory summary for long run histories.
- `sandbox/`: allowlisted tool runner with timeout, cwd restriction, stdout/stderr capture, and error type.
- `harness/`: resumable batch harness with failed-sample rerun and comparison report.

Run a minimal workflow:

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli "为什么 DADA 在 SMD 上表现较好？" --provider offline --dataset SMD --model DADA
```

Run with a real SMD window:

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli "DADA 在 SMD 的真实异常窗口为什么被判为异常？" --provider offline --dataset SMD --model DADA --auto-window --window-length 128
```

Each run writes traceable state to:

```text
experiments/agent_runs/<run_id>/
  events.jsonl
  state.json
  run_record.json
```

This first phase is an architecture skeleton: it provides the Agent loop, persistence, memory compression, sandboxed tools, SMD real-window integration, and a resumable harness without rewriting the existing RAG, feature extraction, LLM provider, or evaluation modules.

## Agent Evidence Consistency

The upgraded workflow treats `state.retrieved` as the single evidence ledger for one run.

- `EvidenceRetriever` writes RAG evidence into `state.retrieved`.
- `WindowAnalyzer` writes the real-window TS-Fact Card into the same list with citation id `S9001`.
- `AnswerWriter` no longer performs a second retrieval. It calls `ExplanationService` with the already retrieved evidence.
- `CitationAuditor` evaluates the final answer against the same evidence list.
- `memory_summary.key_evidence` records all citation ids from the run, including `S9001`.

This matters because an auditable explanation must be able to answer one basic question: did the final answer use the same evidence that the trace says was retrieved? If retrieval, generation, and evaluation use different evidence batches, citation consistency and hallucination metrics become hard to trust.

## Sandbox Boundary

Workflow tools now run through a subprocess-level lightweight sandbox:

- tool names must be registered in an allowlist;
- tool working directories must stay inside the project root;
- each tool call has a timeout, and a timed-out subprocess is terminated;
- stdout, stderr, error type, error message, and elapsed time are captured in the observation record.

This is a workflow safety boundary, not a full OS security sandbox. It is intended to keep project tools controlled and auditable, but it should not be treated as safe execution for arbitrary untrusted code.

## Harness Resume Semantics

- `resume=True`: skip samples that already have a terminal result, including both `succeeded` and `failed`.
- `rerun_failed=True`: run only samples whose latest result is `failed`.
- each sample result records `attempt`, `previous_status`, `run_id`, `error`, and `metrics`.
- `comparison_report.md` reports total, succeeded, failed, rerun count, latest attempt, hallucination rate, and citation consistency.

## Shared Config In Agent Tools

`MultiAgentWorkflow(config_path=...)` resolves the config path once and passes it into subprocess tools explicitly. The retriever, real-window resolver, explanation service, and evaluator therefore share the same configuration during one run. This avoids a subtle reproducibility bug where the main workflow used a custom config while child tools silently fell back to `config/config.yaml`.

## Prompt Evidence Injection

`S9001` remains in `state.retrieved` so run records, memory, and evaluation can audit the real-window TS-Fact Card. In `rag-ts` prompt construction, however, `ts_fact_card` evidence is removed from the ordinary retrieval context and injected only through the dedicated TS-Fact Card block. This keeps the prompt shorter and avoids repeating the same window evidence twice.

## Hallucination Metric Semantics

The hallucination metric no longer treats "citation id exists" as "claim is supported." For cited claims, the cited evidence must exist and pass the lexical support threshold. The metric now reports `citation_exists_claims`, `citation_weak_claims`, and `citation_support_rate` alongside the existing `claim_count`, `supported_claims`, `unsupported_claims`, and `hallucination_rate` fields.

## Hybrid Routing

The current agent router is a lightweight hybrid router, not a LangChain router. The deterministic rule router remains the main path: it infers dataset/model filters, detects window questions, comparison questions, and chooses `rag-only` or `rag-ts`.

Each route now carries `confidence`, `router_type`, and `uncertainty_reasons`. If the rule route confidence is high, the workflow uses it directly. If confidence is below the threshold, the workflow calls `maybe_llm_route`; the current implementation safely marks the route as `hybrid` and records that the LLM structured router fallback is not enabled yet.

This keeps routing reproducible today while leaving a clean replacement point for LangChain structured output or LangGraph conditional edges. A future LangGraph version can express branches such as `rag-only`, `rag-ts`, `comparison`, and failed-window route revision as explicit conditional edges.

Comparison routing now extracts multiple models and datasets when they appear in the question. For example, a question such as "DADA 和 CATCH 在 SMD 上比哪个好？" is routed as `comparison`, with `model_names=["DADA", "CATCH"]` and `dataset_names=["SMD"]`. The evidence retriever calls the existing retrieval tool once per comparison target and merges the resulting evidence by citation id, so the answer writer receives evidence for both models instead of only the first detected model.

The current comparison router is still rule-first. More complex comparison targets, implicit baselines, or cross-dataset comparisons can later be handled by the planned LLM structured router fallback.

## LangGraph Workflow

TS-Explain now keeps two workflow engines side by side:

- `classic`: the original hand-written `Supervisor` workflow in `agent_runtime/workflow.py`.
- `langgraph`: a new StateGraph workflow in `agent_runtime/langgraph_workflow.py`.

Both engines reuse the same Pydantic `AgentState`, `RouteDecision`, sandboxed tool functions, memory compressor, and JSONL run store. The LangGraph version expresses the current agent graph as:

```text
route -> retrieve -> optional window -> optional review -> answer -> audit -> finish
```

Conditional edges decide the branch:

- after `route`: `needs_retrieval=True` goes to `retrieve`, otherwise `review` or `answer`;
- after `retrieve`: `needs_window=True` goes to `window`, otherwise `review` or `answer`;
- after `window`: goes to `review` or `answer`, with failed window analysis downgraded to `rag-only`;
- after `review`: approval continues to `answer`, cancellation goes to `fail`;
- after `answer`: success goes to `audit`, failure goes to `fail`;
- after `audit`: success goes to `finish`, failure goes to `fail`.

The router is still the project hybrid router, not a LangChain router. LangGraph is used for workflow control and conditional edges; future work can replace the low-confidence routing branch with LangChain structured output or richer LangGraph routing nodes.

Run the LangGraph engine from the same CLI:

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli "DADA 和 CATCH 在 SMD 上比哪个好？" --provider offline --engine langgraph
```

### LangGraph Checkpoint And Human Review

`LangGraphWorkflow` can compile the graph with a checkpointer. The `memory` backend uses `InMemorySaver`, which is suitable for local tests and same-process resume only. The `sqlite` backend uses `SqliteSaver` from `langgraph-checkpoint-sqlite` and can resume a human review flow across separate CLI processes.

Each graph run uses a `thread_id`:

- if no `thread_id` is provided, the workflow uses `state.run_id`;
- `thread_id` identifies the LangGraph checkpoint timeline;
- `get_state(thread_id)` and `get_state_history(thread_id)` can inspect the latest checkpoint and history.

Human-in-the-loop review is opt-in. The default `run(...)` path still executes automatically. When `run_interruptible(..., human_review=True)` is used, the graph pauses at the `review` node before answer generation by calling LangGraph `interrupt(...)`. The review payload contains the question, route, top retrieved evidence, an optional TS-Fact Card preview, and allowed actions:

```text
approve | drop_citations | edit_question | cancel
```

Resume is done with LangGraph `Command(resume=...)`, for example:

```python
workflow.resume_interrupt("demo-thread", {"action": "approve"})
workflow.resume_interrupt("demo-thread", {"action": "drop_citations", "drop_citation_ids": ["S0329"]})
```

CLI preview:

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli "DADA 在 SMD 上为什么表现好？" --engine langgraph --provider offline --human-review --thread-id demo-1
```

Cross-process CLI resume should use the same SQLite checkpoint file in both commands:

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli "Why did DADA perform well on SMD?" --engine langgraph --human-review --thread-id demo-1 --checkpoint-backend sqlite --checkpoint-path experiments/checkpoints/demo.sqlite
```

```powershell
C:\Users\Jack\Documents\Timeseries\.venv\Scripts\python.exe -m agent_runtime.cli --engine langgraph --thread-id demo-1 --resume-json "{\"action\":\"approve\"}" --checkpoint-backend sqlite --checkpoint-path experiments/checkpoints/demo.sqlite
```

This human review step is an approval/editing workflow, not an LLM structured router. The LLM router remains a reserved interface in the hybrid router path.
