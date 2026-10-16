import tempfile
import unittest
from pathlib import Path

import numpy as np
import uproot

from backscatter.rootio import (
    aggregate_hits_tree,
    build_pairs,
    read_gate_metadata,
)
from backscatter.truth import match_truth_to_pairs


def _hits(event_id, edep, time, x, photon, primary=None):
    n = len(event_id)
    if primary is None:
        primary = photon
    return {
        "runID": np.zeros(n, dtype=np.int32),
        "eventID": np.asarray(event_id, dtype=np.int32),
        "edep": np.asarray(edep, dtype=np.float32),
        "time": np.asarray(time, dtype=np.float64),
        "posX": np.asarray(x, dtype=np.float32),
        "posY": np.zeros(n, dtype=np.float32),
        "posZ": np.ones(n, dtype=np.float32),
        "sourcePosX": np.zeros(n, dtype=np.float32),
        "sourcePosY": np.zeros(n, dtype=np.float32),
        "sourcePosZ": np.full(n, 17.5, dtype=np.float32),
        "photonID": np.asarray(photon, dtype=np.int32),
        "primaryID": np.asarray(primary, dtype=np.int32),
        "nPhantomCompton": np.asarray([1] * n, dtype=np.int32),
        "nCrystalCompton": np.zeros(n, dtype=np.int32),
        "nPhantomRayleigh": np.zeros(n, dtype=np.int32),
        "nCrystalRayleigh": np.zeros(n, dtype=np.int32),
    }


class RootIoTests(unittest.TestCase):
    def test_hit_aggregation_and_pair_time_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.root"
            with uproot.recreate(path) as root:
                root["pet_data"] = {
                    "total_nb_primaries": np.array([10], dtype=np.int32),
                    "latest_event_ID": np.array([9], dtype=np.int32),
                    "start_time_sec": np.array([0.0]),
                    "stop_time_sec": np.array([1.0]),
                }
                root["Hits_BSdetector"] = _hits(
                    [1, 1, 2],
                    [0.10, 0.05, 0.20],
                    [1.001e-6, 1.002e-6, 2.001e-6],
                    [10.0, 20.0, 30.0],
                    [1, 1, 1],
                )
                root["Hits_crystal"] = _hits(
                    [1, 2],
                    [0.511, 0.511],
                    [1.000e-6, 2.000e-6],
                    [0.0, 1.0],
                    [2, 2],
                )

            d1 = aggregate_hits_tree(path, "Hits_BSdetector")
            d2 = aggregate_hits_tree(path, "Hits_crystal")
            self.assertEqual(len(d1["key"]), 2)
            self.assertAlmostEqual(float(d1["energy_keV"][0]), 150.0, places=4)
            self.assertAlmostEqual(float(d1["x_mm"][0]), 40.0 / 3.0, places=4)

            pairs = build_pairs(d1, d2)
            self.assertEqual(len(pairs["key"]), 2)
            self.assertTrue(np.all(pairs["different_primary_photons"]))
            self.assertAlmostEqual(float(pairs["dt_ns"][0]), 1.0, places=6)

            metadata = read_gate_metadata(path)
            self.assertEqual(metadata["total_nb_primaries"], 10)
            self.assertEqual(metadata["primary_event_count"], 10)
            self.assertEqual(
                metadata["primary_event_count_source"], "total_nb_primaries"
            )

    def test_gate_941_last_event_id_metadata_is_converted_to_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate_941.root"
            with uproot.recreate(path) as root:
                root["pet_data"] = {
                    "total_nb_primaries": np.array([9], dtype=np.int32),
                    "latest_event_ID": np.array([9], dtype=np.int32),
                    "start_time_sec": np.array([0.0]),
                    "stop_time_sec": np.array([1.0]),
                }

            metadata = read_gate_metadata(path)
            self.assertEqual(metadata["total_nb_primaries"], 9)
            self.assertEqual(metadata["latest_event_ID"], 9)
            self.assertEqual(metadata["primary_event_count"], 10)
            self.assertEqual(
                metadata["primary_event_count_source"],
                "latest_event_ID_plus_one",
            )

    def test_truth_matches_geant4_primary_id_not_gate_photon_label(self):
        pairs = {
            "key": np.array([44], dtype=np.int64),
            "d1_photon_id": np.array([2], dtype=np.int16),
            "d1_primary_id": np.array([1], dtype=np.int16),
        }
        truth = {
            "key": np.array([44], dtype=np.int64),
            "parent_id": np.array([1], dtype=np.int32),
            "x_mm": np.array([1.0]),
            "y_mm": np.array([2.0]),
            "z_mm": np.array([-3.0]),
            "n_compton_vertices": np.array([1], dtype=np.int16),
        }

        matched = match_truth_to_pairs(pairs, truth)
        self.assertTrue(matched["matched"][0])
        self.assertEqual(matched["z_mm"][0], -3.0)


if __name__ == "__main__":
    unittest.main()
