from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from agent_runtime.langgraph_workflow import LangGraphWorkflow

    LANGGRAPH_AVAILABLE = True
except Exception:
    LangGraphWorkflow = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph is not installed")
class LangGraphWorkflowTest(unittest.TestCase):
    def _event_types(self, run_dir: Path) -> list[str]:
        return [
            json.loads(line)["event_type"]
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_langgraph_workflow_runs_minimal_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
            )

            self.assertEqual(state.status, "succeeded")
            self.assertTrue(state.answer)
            self.assertTrue(state.route)
            self.assertTrue(state.memory_summary)
            run_dir = Path(tmp) / state.run_id
            self.assertTrue((run_dir / "state.json").exists())
            self.assertTrue((run_dir / "events.jsonl").exists())
            events = self._event_types(run_dir)
            self.assertIn("langgraph_route_started", events)
            self.assertIn("langgraph_run_finished", events)

    def test_langgraph_workflow_runs_comparison_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(question="DADA \u548c CATCH \u5728 SMD \u4e0a\u6bd4\u54ea\u4e2a\u597d\uff1f")

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

    def test_langgraph_workflow_runs_real_smd_window_when_assets_exist(self) -> None:
        smd_dir = Path(__file__).resolve().parents[2] / "ts-explain-assets" / "Anomaly-Transformer" / "dataset" / "SMD"
        if not (smd_dir / "SMD_test.npy").exists():
            self.skipTest("SMD assets are not available")
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp)
            state = workflow.run(
                question="Why was the real DADA anomaly window on SMD marked anomalous?",
                dataset_name="SMD",
                model_name="DADA",
                auto_window=True,
                window_length=64,
            )

            self.assertEqual(state.status, "succeeded")
            self.assertTrue(any(item.get("citation_id") == "S9001" for item in state.retrieved))
            self.assertIn("S9001", state.citations)
            self.assertTrue(state.memory_summary)
            self.assertIn("S9001:ts_fact_card", state.memory_summary.key_evidence)

    def test_langgraph_checkpoint_history_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp, enable_checkpoint=True)
            result = workflow.run_interruptible(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
                human_review=False,
                thread_id="test-checkpoint",
            )

            self.assertEqual(result.state.status, "succeeded")
            history = workflow.get_state_history("test-checkpoint")
            self.assertGreaterEqual(len(history), 2)
            run_dir = Path(result.output_dir)
            events = self._event_types(run_dir)
            self.assertIn("langgraph_checkpoint_enabled", events)

    def test_langgraph_human_review_interrupt_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp, enable_checkpoint=True)
            result = workflow.run_interruptible(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
                human_review=True,
                thread_id="test-hitl",
            )

            self.assertTrue(result.interrupted)
            self.assertTrue(result.interrupts)
            payload = result.interrupts[0]["value"]
            self.assertEqual(payload["question"], "Why did DADA perform well on SMD?")
            self.assertIn("approve", payload["allowed_actions"])
            self.assertIn("top_retrieved_evidence", payload)

            resumed = workflow.resume_interrupt("test-hitl", {"action": "approve"})

            self.assertFalse(resumed.interrupted)
            self.assertEqual(resumed.state.status, "succeeded")
            self.assertTrue(resumed.state.answer)
            events = self._event_types(Path(resumed.output_dir))
            self.assertIn("langgraph_interrupted", events)
            self.assertIn("langgraph_review_resumed", events)
            self.assertIn("langgraph_run_finished", events)

    def test_langgraph_human_review_drop_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = LangGraphWorkflow(provider="offline", runs_dir=tmp, enable_checkpoint=True)
            result = workflow.run_interruptible(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
                human_review=True,
                thread_id="test-hitl-drop",
            )

            payload = result.interrupts[0]["value"]
            candidate_ids = [
                item.get("citation_id")
                for item in payload["top_retrieved_evidence"]
                if item.get("citation_id")
            ]
            if not candidate_ids:
                self.skipTest("No citation id available in review payload")
            dropped_id = candidate_ids[0]
            resumed = workflow.resume_interrupt(
                "test-hitl-drop",
                {"action": "drop_citations", "drop_citation_ids": [dropped_id]},
            )

            self.assertEqual(resumed.state.status, "succeeded")
            self.assertFalse(any(item.get("citation_id") == dropped_id for item in resumed.state.retrieved))

    def test_langgraph_human_review_cross_process_sqlite_resume(self) -> None:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        except Exception:
            self.skipTest("langgraph SQLite checkpointer is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoints" / "hitl.sqlite"
            workflow1 = LangGraphWorkflow(
                provider="offline",
                runs_dir=tmp,
                enable_checkpoint=True,
                checkpoint_backend="sqlite",
                checkpoint_path=checkpoint_path,
            )
            result = workflow1.run_interruptible(
                question="Why did DADA perform well on SMD?",
                dataset_name="SMD",
                model_name="DADA",
                human_review=True,
                thread_id="cross-process-test",
            )
            self.assertTrue(result.interrupted)
            run_dir = Path(result.output_dir)
            workflow1.close()
            del workflow1

            workflow2 = LangGraphWorkflow(
                provider="offline",
                runs_dir=tmp,
                enable_checkpoint=True,
                checkpoint_backend="sqlite",
                checkpoint_path=checkpoint_path,
            )
            resumed = workflow2.resume_interrupt("cross-process-test", {"action": "approve"})

            self.assertEqual(resumed.state.status, "succeeded")
            self.assertTrue(resumed.state.answer)
            events = self._event_types(run_dir)
            self.assertIn("langgraph_interrupted", events)
            self.assertIn("langgraph_resume_started", events)
            self.assertIn("langgraph_run_finished", events)
            workflow2.close()


if __name__ == "__main__":
    unittest.main()
