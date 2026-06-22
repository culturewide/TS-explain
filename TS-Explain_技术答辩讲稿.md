# TS-Explain 技术答辩讲稿

## 第 1 页：项目标题
讲法：先说明这是技术答辩，不是商业宣传。项目目标是用 RAG 加 Multi-Agent workflow 解释多变量时间序列异常检测结果。重点讲状态管理、工具沙盒、记忆、Harness 和 LangGraph。

## 第 2 页：项目要解决的问题
讲法：普通 RAG 只能检索文本，但本项目要回答模型表现、真实异常窗口和多模型比较三类问题。真实窗口解释需要时序统计事实，不只是论文摘要。

## 第 3 页：整体架构图
讲法：用户问题进入 Router，决定 rag-only、rag-ts 或 comparison；Retriever 找 RAG 证据；WindowAnalyzer 可选生成 TS-Fact Card；AnswerWriter 生成答案；CitationAuditor 评估；最后 MemoryCompressor 和 store 落盘。

## 第 4 页：多 Agent 分工
讲法：每个 Agent 只负责一个环节。QueryRouter 做路由，EvidenceRetriever 做检索，WindowAnalyzer 做窗口事实卡，AnswerWriter 生成答案，CitationAuditor 评估，Supervisor 或 LangGraph 负责流程编排。

## 第 5 页：Pydantic 状态管理
讲法：AgentState 是一次运行的事实账本。route、retrieved、fact_card_text、answer、citations、metrics、steps、memory_summary 都在里面。state.json 是状态快照，events.jsonl 是过程事件，runs.jsonl 是运行索引。

## 第 6 页：路由策略
讲法：当前不是 LangChain Router，而是规则路由为主的 hybrid router。规则识别模型、数据集、窗口词和比较词，并输出 confidence 和 uncertainty_reasons。低置信度时 maybe_llm_route 只是预留接口。

## 第 7 页：真实窗口解释与 S9001
讲法：WindowAnalyzer 从真实 SMD 窗口生成 TS-Fact Card，包含趋势、周期、相关性、变点和变量贡献等。S9001 是窗口事实卡的引用 id，会进入 state.retrieved、answer citations、metrics 和 memory。注意：这不是 DADA 的逐点 anomaly score。

## 第 8 页：工具沙盒和执行记录
讲法：ToolSandbox 用子进程执行 allowlist 工具，限制 cwd，支持 timeout，捕获 stdout/stderr/error。每次调用形成 ToolCall 和 Observation，并写入 AgentStep，方便追踪失败。

## 第 9 页：Harness 测试与失败恢复
讲法：AgentHarness 支持批量样本、resume 和 rerun_failed，输出 harness_results.jsonl 和 comparison_report.md。当前 rerun_failed 是重新跑失败项，还没有利用失败上下文做智能恢复。

## 第 10 页：LangGraph 接入
讲法：classic workflow 是手写 Supervisor，LangGraph workflow 是新增 StateGraph 版本。两者复用同一套状态、工具、记忆和 store。LangGraph 用 conditional edges 表达 route、retrieve、window、answer、audit、finish 分支。它提升的是编排清晰度和可扩展性，不直接提升答案质量。

## 第 11 页：测试与当前效果
讲法：展示测试命令和结果：Ran 26 tests in 58.515s, OK。覆盖 classic workflow、LangGraph workflow、真实 SMD 窗口、S9001 一致性、comparison 多模型检索、sandbox 和 harness。

## 第 12 页：不足与后续计划
讲法：主动说明不足：LLM structured router 还没真正接入，comparison answer 还可结构化，LangGraph 还没接 checkpoint/human-in-the-loop，citation 评估偏词面匹配。后续可以做 run viewer 可视化 state.json 和 events.jsonl。

## 适合老师追问的问题
- 为什么要用 Agent workflow，而不是普通 RAG pipeline？
- S9001 是什么？为什么要单独处理？
- LangGraph 和 classic workflow 的区别是什么？
- citation_fidelity 和 hallucination_rate 如何计算？
- ToolSandbox 是不是完整安全沙盒？
- 真实窗口解释是否等于模型自身 anomaly score？
- comparison 检索怎么保证两个模型都有证据？
