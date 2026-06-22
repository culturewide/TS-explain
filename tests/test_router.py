from __future__ import annotations

import unittest

from agent_runtime.router import LLM_ROUTER_DISABLED_REASON, maybe_llm_route
from agent_runtime.schema import RouteDecision
from agent_runtime.tool_functions import route_query


REAL_WINDOW_QUESTION = "DADA \u5728 SMD \u7684\u771f\u5b9e\u5f02\u5e38\u7a97\u53e3\u4e3a\u4ec0\u4e48\u88ab\u5224\u4e3a\u5f02\u5e38\uff1f"
AMBIGUOUS_QUESTION = "\u8fd9\u4e2a\u6a21\u578b\u4e3a\u4ec0\u4e48\u8868\u73b0\u4e0d\u597d\uff1f"
COMPLETE_COMPARISON = "DADA \u548c CATCH \u5728 SMD \u4e0a\u6bd4\u54ea\u4e2a\u597d\uff1f"
AMBIGUOUS_COMPARISON = "DADA \u548c\u54ea\u4e2a\u6a21\u578b\u6bd4\u66f4\u597d\uff1f"


class HybridRouterTest(unittest.TestCase):
    def test_clear_real_window_question_gets_high_confidence_rule_route(self) -> None:
        raw = route_query(
            question=REAL_WINDOW_QUESTION,
            dataset_name=None,
            model_name=None,
            auto_window=False,
        )
        route = RouteDecision.model_validate(raw["route"])
        final = maybe_llm_route(
            question=REAL_WINDOW_QUESTION,
            current_route=route,
            provider="offline",
        )

        self.assertEqual(final.question_type, "window_explanation")
        self.assertEqual(final.mode, "rag-ts")
        self.assertTrue(final.needs_window)
        self.assertEqual(final.dataset_name, "SMD")
        self.assertEqual(final.model_name, "DADA")
        self.assertGreaterEqual(final.confidence, 0.75)
        self.assertIn(final.router_type, {"rule", "hybrid"})

    def test_ambiguous_question_gets_low_confidence_hybrid_marker(self) -> None:
        clear = RouteDecision.model_validate(
            route_query(
                question=REAL_WINDOW_QUESTION,
                dataset_name=None,
                model_name=None,
                auto_window=False,
            )["route"]
        )
        ambiguous = RouteDecision.model_validate(
            route_query(
                question=AMBIGUOUS_QUESTION,
                dataset_name=None,
                model_name=None,
                auto_window=False,
            )["route"]
        )
        final = maybe_llm_route(question=AMBIGUOUS_QUESTION, current_route=ambiguous, provider="offline")

        self.assertIsInstance(final, RouteDecision)
        self.assertLess(final.confidence, clear.confidence)
        self.assertTrue(final.uncertainty_reasons)
        self.assertEqual(final.router_type, "hybrid")
        self.assertIn(LLM_ROUTER_DISABLED_REASON, final.uncertainty_reasons)

    def test_complete_comparison_extracts_multiple_models(self) -> None:
        route = RouteDecision.model_validate(
            route_query(
                question=COMPLETE_COMPARISON,
                dataset_name=None,
                model_name=None,
                auto_window=False,
            )["route"]
        )

        self.assertEqual(route.question_type, "comparison")
        self.assertEqual(route.mode, "rag-only")
        self.assertTrue(route.needs_retrieval)
        self.assertIn("DADA", route.model_names)
        self.assertIn("CATCH", route.model_names)
        self.assertIn("SMD", route.dataset_names)
        self.assertGreaterEqual(route.confidence, 0.75)

    def test_ambiguous_comparison_records_missing_second_model(self) -> None:
        complete = RouteDecision.model_validate(
            route_query(
                question=COMPLETE_COMPARISON,
                dataset_name=None,
                model_name=None,
                auto_window=False,
            )["route"]
        )
        ambiguous = RouteDecision.model_validate(
            route_query(
                question=AMBIGUOUS_COMPARISON,
                dataset_name=None,
                model_name=None,
                auto_window=False,
            )["route"]
        )

        self.assertEqual(ambiguous.question_type, "comparison")
        self.assertEqual(ambiguous.model_names, ["DADA"])
        self.assertLess(ambiguous.confidence, complete.confidence)
        self.assertIn("comparison question has fewer than two models", ambiguous.uncertainty_reasons)


if __name__ == "__main__":
    unittest.main()
