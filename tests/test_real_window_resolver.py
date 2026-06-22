from __future__ import annotations

import unittest
from pathlib import Path

from feature_extraction.real_window_resolver import RealWindowResolver


class RealWindowResolverTest(unittest.TestCase):
    def test_resolve_smd_real_window_when_assets_exist(self) -> None:
        smd_dir = Path(__file__).resolve().parents[2] / "ts-explain-assets" / "Anomaly-Transformer" / "dataset" / "SMD"
        if not (smd_dir / "SMD_test.npy").exists():
            self.skipTest("SMD assets are not available")
        real_window, card = RealWindowResolver().extract_fact_card("SMD", model_name="DADA", window_length=64)
        self.assertEqual(real_window.dataset_name, "SMD")
        self.assertEqual(real_window.values.shape[0], 64)
        self.assertGreater(real_window.values.shape[1], 1)
        self.assertIn("真实窗口来源", card.narrative)
        self.assertIn("SMD_test.npy", card.narrative)


if __name__ == "__main__":
    unittest.main()
