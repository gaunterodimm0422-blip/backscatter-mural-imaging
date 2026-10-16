import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backscatter.hardware_figures import materialize_response_analysis_view


class HardwareFigureAnalysisViewTests(unittest.TestCase):
    def test_response_view_uses_response_arrays_and_updates_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            response = root / "response"
            source.mkdir()
            response.mkdir()
            with (source / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "event_counts": {
                            "primaries": 100,
                            "common_event_pairs_before_filters": 5,
                            "selection_chain": {"old": 1},
                            "selected_before_geometric_solver": 1,
                            "selected_and_reconstructed": 1,
                        },
                        "warnings": [],
                    },
                    handle,
                )
            np.savez_compressed(
                response / "pairs_response.npz",
                key=np.arange(3),
                d1_energy_keV=np.ones(3),
            )
            np.savez_compressed(
                response / "reconstruction_response.npz",
                key=np.arange(3),
                valid=np.asarray([True, False, True]),
            )
            with (response / "response_analysis.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "n_initial_pairs": 3,
                        "n_selected": 2,
                        "n_selected_valid": 1,
                        "selection_chain": {"response_cut": 2},
                    },
                    handle,
                )

            manifest = materialize_response_analysis_view(
                source_analysis_dir=source,
                response_dir=response,
            )

            self.assertTrue((response / "pairs.npz").is_file())
            self.assertTrue((response / "reconstruction.npz").is_file())
            self.assertFalse((response / "events.npz").exists())
            with (response / "summary.json").open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            self.assertEqual(
                summary["event_counts"]["selected_and_reconstructed"], 1
            )
            self.assertEqual(
                summary["event_counts"]["selection_chain"],
                {"response_cut": 2},
            )
            self.assertIn("pairs.npz", manifest["aliases"])


if __name__ == "__main__":
    unittest.main()
