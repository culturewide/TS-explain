from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_runtime.schema import HarnessItem, HarnessResult
from agent_runtime.workflow import MultiAgentWorkflow
from harness.runner import AgentHarness


class AgentWorkflowTest(unittest.TestCase):
    def test_minimal_agent_workflow_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = MultiAgentWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
            )
            self.assertEqual(state.status, "succeeded")
            self.assertTrue(state.answer)
            run_dir = Path(tmp) / state.run_id
            self.assertTrue((run_dir / "events.jsonl").exists())
            self.assertTrue((run_dir / "state.json").exists())
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["event_type"] == "tool_started" for event in events))
            self.assertTrue(state.memory_summary)

    def test_agent_workflow_with_real_smd_window_keeps_s9001_consistent(self) -> None:
        smd_dir = Path(__file__).resolve().parents[2] / "ts-explain-assets" / "Anomaly-Transformer" / "dataset" / "SMD"
        if not (smd_dir / "SMD_test.npy").exists():
            self.skipTest("SMD assets are not available")
        with tempfile.TemporaryDirectory() as tmp:
            workflow = MultiAgentWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(
                question="Why was the real DADA anomaly window on SMD marked anomalous?",
                dataset_name="SMD",
                model_name="DADA",
                auto_window=True,
                window_length=64,
            )
            self.assertEqual(state.status, "succeeded")
            self.assertIn("SMD_test.npy", state.fact_card_text or "")
            self.assertTrue(any(item.get("citation_id") == "S9001" for item in state.retrieved))
            self.assertIn("S9001", state.citations)
            fidelity = state.metrics.get("citation_fidelity", [])
            self.assertTrue(any(item.get("citation_id") == "S9001" for item in fidelity))
            self.assertTrue(state.memory_summary)
            self.assertIn("S9001:ts_fact_card", state.memory_summary.key_evidence)

    def test_harness_resume_skips_terminal_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = AgentHarness(output_dir=tmp, provider="offline")
            prior = HarnessResult(id="sample-1", status="failed", attempt=1, error="old failure")
            harness.results_path.write_text(prior.model_dump_json() + "\n", encoding="utf-8")

            results = harness.run_items(
                [HarnessItem(id="sample-1", question="Why did DADA perform well on SMD?", dataset_name="SMD")],
                resume=True,
                rerun_failed=False,
            )

            self.assertEqual(results["sample-1"].status, "failed")
            self.assertEqual(results["sample-1"].attempt, 1)
            lines = harness.results_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

    def test_harness_rerun_failed_only_updates_failed_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = AgentHarness(output_dir=tmp, provider="offline")
            failed = HarnessResult(id="failed-1", status="failed", attempt=1, error="old failure")
            succeeded = HarnessResult(id="succeeded-1", status="succeeded", attempt=1, run_id="old-run")
            harness.results_path.write_text(
                failed.model_dump_json() + "\n" + succeeded.model_dump_json() + "\n",
                encoding="utf-8",
            )

            results = harness.run_items(
                [
                    HarnessItem(
                        id="failed-1",
                        question="Why did DADA perform well on SMD?",
                        dataset_name="SMD",
                        model_name="DADA",
                    ),
                    HarnessItem(
                        id="succeeded-1",
                        question="Why did DADA perform well on SMD?",
                        dataset_name="SMD",
                        model_name="DADA",
                    ),
                ],
                resume=True,
                rerun_failed=True,
            )

            self.assertEqual(results["failed-1"].attempt, 2)
            self.assertEqual(results["failed-1"].previous_status, "failed")
            self.assertEqual(results["succeeded-1"].attempt, 1)
            report = harness.report_path.read_text(encoding="utf-8")
            self.assertIn("rerun count: 1", report)
            self.assertIn("| failed-1 |", report)

    def test_workflow_passes_custom_config_path_to_subprocess_tools(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_config = project_root / "config" / "config.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            custom_config = tmp_path / "config.yaml"
            shutil.copyfile(source_config, custom_config)
            data = json.loads(custom_config.read_text(encoding="utf-8"))
            data["evaluation"]["support_threshold"] = 0.91
            custom_config.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            workflow = MultiAgentWorkflow(config_path=custom_config, provider="offline", runs_dir=tmp_path / "runs")
            state = workflow.run(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
            )

            self.assertEqual(state.status, "succeeded")
            self.assertEqual(state.metrics["hallucination"]["support_threshold"], 0.91)
            tool_args = [
                step.tool_call.args
                for step in state.steps
                if step.tool_call and step.tool_call.tool_name in {"retrieve_evidence", "generate_answer", "evaluate_answer"}
            ]
            self.assertTrue(tool_args)
            self.assertTrue(all(args.get("config_path") == str(custom_config.resolve()) for args in tool_args))

    def test_comparison_workflow_retrieves_multiple_model_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = MultiAgentWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(
                question="DADA \u548c CATCH \u5728 SMD \u4e0a\u6bd4\u54ea\u4e2a\u597d\uff1f",
                dataset_name=None,
                model_name=None,
            )

            self.assertEqual(state.status, "succeeded")
            self.assertTrue(state.route)
            self.assertEqual(state.route.question_type, "comparison")
            self.assertIn("DADA", state.route.model_names)
            self.assertIn("CATCH", state.route.model_names)
            retrieved_models = {
                item.get("metadata", {}).get("model_name")
                for item in state.retrieved
                if item.get("metadata", {}).get("model_name")
            }
            self.assertIn("DADA", retrieved_models)
            self.assertIn("CATCH", retrieved_models)
            self.assertTrue(state.memory_summary)
            self.assertTrue(
                any("comparison_targets=models:DADA,CATCH; datasets:SMD" in item for item in state.memory_summary.decisions)
            )


if __name__ == "__main__":
    unittest.main()
