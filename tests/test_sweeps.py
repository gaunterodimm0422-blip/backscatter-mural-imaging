import unittest
from pathlib import Path

import numpy as np

from backscatter.config import load_config
from backscatter.sweeps import _row_metrics, _truth_signal_mask


ROOT = Path(__file__).resolve().parents[1]


class SweepTruthScoringTests(unittest.TestCase):
    def test_observable_selection_is_scored_by_truth_without_using_it_as_cut(self):
        config = load_config(ROOT / "configs" / "baseline.json")
        pairs = {
            "d1_compt_volume": np.asarray(
                ["concrete", "concrete", "concrete", "NULL"]
            ),
            "d1_n_phantom_compton": np.asarray([1, 2, 1, 0]),
            "d2_n_phantom_compton": np.asarray([0, 0, 1, 0]),
        }
        truth_signal = _truth_signal_mask(pairs, config)
        np.testing.assert_array_equal(
            truth_signal,
            np.asarray([True, False, False, False]),
        )

        reconstruction = {
            "valid": np.asarray([True, True, True, False]),
            "x_mm": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "y_mm": np.zeros(4),
            "z_mm": np.asarray([-10.0, -11.0, -12.0, -13.0]),
            "truth_matched": np.asarray([True, True, True, False]),
            "truth_x_mm": np.zeros(4),
            "truth_y_mm": np.zeros(4),
            "truth_z_mm": np.asarray([-10.0, -10.0, -10.0, np.nan]),
        }
        row = _row_metrics(
            category="baseline",
            setting="test",
            selected=np.asarray([True, True, True, True]),
            truth_signal=truth_signal,
            reconstruction=reconstruction,
            n_primaries=100,
            standard_target_kind=None,
            target_depth_mm=None,
        )
        self.assertEqual(row["n_observable_selected_valid"], 3)
        self.assertEqual(row["n_truth_signal_valid"], 1)
        self.assertAlmostEqual(row["truth_signal_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(row["efficiency_per_primary"], 0.01)
        self.assertAlmostEqual(row["observable_efficiency_per_primary"], 0.03)


if __name__ == "__main__":
    unittest.main()
