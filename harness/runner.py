from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from agent_runtime.schema import HarnessItem, HarnessResult
from agent_runtime.workflow import MultiAgentWorkflow
from core.config import load_yaml


class AgentHarness:
    """Resumable harness for multi-agent workflow samples."""

    def __init__(self, *, output_dir: str | Path, provider: str = "offline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.results_path = self.output_dir / "harness_results.jsonl"
        self.report_path = self.output_dir / "comparison_report.md"

    def load_items(self, path: str | Path) -> List[HarnessItem]:
        raw = load_yaml(path)
        if isinstance(raw, dict):
            items = raw.get("items", raw.get("questions", []))
        else:
            items = raw
        return [HarnessItem.model_validate(item) for item in items]

    def result_history(self) -> Dict[str, List[HarnessResult]]:
        if not self.results_path.exists():
            return {}
        history: Dict[str, List[HarnessResult]] = {}
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = HarnessResult.model_validate_json(line)
                history.setdefault(result.id, []).append(result)
        return history

    def existing_results(self) -> Dict[str, HarnessResult]:
        return {sample_id: records[-1] for sample_id, records in self.result_history().items() if records}

    def run_items(
        self,
        items: Iterable[HarnessItem],
        *,
        resume: bool = True,
        rerun_failed: bool = False,
    ) -> Dict[str, HarnessResult]:
        final = self.existing_results()
        with self.results_path.open("a", encoding="utf-8") as f:
            for item in items:
                prior = final.get(item.id)
                if rerun_failed:
                    if prior is None or prior.status != "failed":
                        continue
                elif resume and prior and prior.status in {"succeeded", "failed"}:
                    continue

                attempt = (prior.attempt + 1) if prior else 1
                previous_status = prior.status if prior else None
                workflow = MultiAgentWorkflow(
                    provider=item.provider or self.provider,
                    runs_dir=self.output_dir / "agent_runs",
                )
                try:
                    state = workflow.run(
                        question=item.question,
                        dataset_name=item.dataset_name,
                        model_name=item.model_name,
                        auto_window=item.auto_window,
                    )
                    result = HarnessResult(
                        id=item.id,
                        run_id=state.run_id,
                        status="succeeded" if state.status == "succeeded" else "failed",
                        attempt=attempt,
                        previous_status=previous_status,
                        output_dir=str(workflow.store.run_dir(state.run_id)),
                        metrics=state.metrics,
                        error="; ".join(state.failures) if state.failures else None,
                    )
                except Exception as exc:
                    result = HarnessResult(
                        id=item.id,
                        status="failed",
                        attempt=attempt,
                        previous_status=previous_status,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                final[item.id] = result
                f.write(result.model_dump_json() + "\n")
                f.flush()
        self.write_report(final.values())
        return final

    def write_report(self, results: Iterable[HarnessResult]) -> None:
        rows = list(results)
        succeeded = [r for r in rows if r.status == "succeeded"]
        failed = [r for r in rows if r.status == "failed"]
        rerun_count = sum(1 for r in rows if r.attempt > 1)
        lines = [
            "# Agent Harness Report",
            "",
            f"- total: {len(rows)}",
            f"- succeeded: {len(succeeded)}",
            f"- failed: {len(failed)}",
            f"- rerun count: {rerun_count}",
            "",
            "| id | status | attempt | previous status | hallucination | citation consistency | run |",
            "|---|---|---:|---|---:|---:|---|",
        ]
        for item in rows:
            metrics = item.metrics or {}
            h = metrics.get("hallucination", {}).get("hallucination_rate")
            c = metrics.get("citation_consistency_rate")
            h_text = "" if h is None else f"{float(h) * 100:.2f}%"
            c_text = "" if c is None else f"{float(c) * 100:.2f}%"
            lines.append(
                f"| {item.id} | {item.status} | {item.attempt} | {item.previous_status or ''} | "
                f"{h_text} | {c_text} | {item.run_id or ''} |"
            )
        if failed:
            lines.extend(["", "## Failed Items"])
            for item in failed:
                lines.append(f"- {item.id}: attempt={item.attempt}; error={item.error}")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
