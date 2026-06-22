from __future__ import annotations

import unittest

import numpy as np

from feature_extraction.extractor import TSFeatureExtractor


class FeatureExtractionTest(unittest.TestCase):
    def test_fact_card_contains_core_sections(self) -> None:
        t = np.arange(96)
        values = np.stack(
            [
                0.03 * t + np.sin(2 * np.pi * t / 24),
                np.cos(2 * np.pi * t / 12),
                np.random.default_rng(0).normal(0, 0.1, len(t)),
            ],
            axis=1,
        )
        values[40:48, 1] += 4
        scores = np.mean(np.abs(values - values.mean(axis=0)), axis=1)
        card = TSFeatureExtractor().extract(values, window_id="w1", anomaly_scores=scores, feature_names=["a", "b", "c"])
        self.assertEqual(card.length, 96)
        self.assertIsNotNone(card.anomaly_profile)
        self.assertGreater(len(card.trends), 0)
        self.assertGreater(len(card.attributions), 0)
        self.assertIn("TS-Fact Card", card.narrative)


if __name__ == "__main__":
    unittest.main()

