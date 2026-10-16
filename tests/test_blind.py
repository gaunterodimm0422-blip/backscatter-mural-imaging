import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backscatter.blind import (
    evaluate_blind_void_campaign,
    score_blind_points,
)


class BlindVoidScoreTests(unittest.TestCase):
    @staticmethod
    def _uniform_points():
        x_values = np.arange(-140.0, 140.1, 2.0)
        depth_values = np.arange(15.0, 45.1, 2.0)
        x_grid, depth_grid = np.meshgrid(
            x_values,
            depth_values,
            indexing="xy",
        )
        return x_grid.ravel(), depth_grid.ravel()

    def test_identical_sample_and_reference_have_zero_maximum(self):
        x, depth = self._uniform_points()
        result = score_blind_points(
            sample_x_mm=x,
            sample_depth_mm=depth,
            reference_x_mm=x,
            reference_depth_mm=depth,
            roi_radius_mm=8.0,
            search_x_interval_mm=(-20.0, 20.0),
            search_depth_interval_mm=(30.0, 30.0),
            search_step_mm=10.0,
            slab_x_interval_mm=(-250.0, 250.0),
        )
        self.assertAlmostEqual(result["maximum"]["cnr"], 0.0)
        self.assertEqual(result["n_candidates"], 5)
        self.assertEqual(result["n_valid_candidates"], 5)

    def test_removed_central_counts_localize_the_blind_maximum(self):
        reference_x, reference_depth = self._uniform_points()
        inside_defect = (
            reference_x**2 + (reference_depth - 30.0) ** 2
            <= 8.0**2
        )
        sample_x = reference_x[~inside_defect]
        sample_depth = reference_depth[~inside_defect]
        result = score_blind_points(
            sample_x_mm=sample_x,
            sample_depth_mm=sample_depth,
            reference_x_mm=reference_x,
            reference_depth_mm=reference_depth,
            roi_radius_mm=8.0,
            search_x_interval_mm=(-20.0, 20.0),
            search_depth_interval_mm=(30.0, 30.0),
            search_step_mm=10.0,
            slab_x_interval_mm=(-250.0, 250.0),
        )
        maximum = result["maximum"]
        self.assertEqual(maximum["candidate_x_mm"], 0.0)
        self.assertEqual(maximum["candidate_depth_mm"], 30.0)
        self.assertGreater(maximum["cnr"], 4.0)

    def test_reversed_search_interval_is_rejected(self):
        x, depth = self._uniform_points()
        with self.assertRaises(ValueError):
            score_blind_points(
                sample_x_mm=x,
                sample_depth_mm=depth,
                reference_x_mm=x,
                reference_depth_mm=depth,
                roi_radius_mm=8.0,
                search_x_interval_mm=(20.0, -20.0),
                search_depth_interval_mm=(30.0, 30.0),
                search_step_mm=10.0,
                slab_x_interval_mm=(-250.0, 250.0),
            )

    def test_frozen_campaign_commitments_are_verified_before_reveal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            truth_cases = []
            score_cases = []
            for index in range(1, 13):
                case_id = f"blindq_test_case{index:03d}"
                is_positive = index <= 6
                truth_cases.append(
                    {
                        "case_id": case_id,
                        "class": "void" if is_positive else "no_void",
                        "x_mm": 0.0 if is_positive else None,
                        "depth_mm": 30.0 if is_positive else None,
                    }
                )
                score_cases.append(
                    {
                        "case_id": case_id,
                        "maximum_score": 5.0 if is_positive else 2.0,
                        "candidate_flag": is_positive,
                        "estimated_x_mm": 0.0,
                        "estimated_depth_mm": 30.0,
                    }
                )

            truth = {
                "schema_version": "1.0",
                "campaign_tag": "blindq_test",
                "cases": truth_cases,
            }
            truth_path = root / "truth_key.json"
            truth_path.write_text(
                json.dumps(truth, indent=2) + "\n",
                encoding="utf-8",
            )
            salt = bytes(range(32))
            salt_path = root / "truth_salt.txt"
            salt_path.write_text(salt.hex() + "\n", encoding="ascii")
            commitment = hashlib.sha256(
                salt + truth_path.read_bytes()
            ).hexdigest()
            commitment_path = root / "truth_commitment.sha256"
            commitment_path.write_text(
                commitment + "\n",
                encoding="ascii",
            )

            scores = {
                "schema_version": "1.0",
                "campaign_tag": "blindq_test",
                "scores_frozen_before_truth_reveal": True,
                "contains_truth_labels_or_positions": False,
                "public_truth_commitment_sha256": commitment,
                "frozen_protocol": {
                    "candidate_threshold": 4.0,
                    "localization_tolerance_mm_after_reveal": 15.0,
                },
                "cases": score_cases,
            }
            score_path = root / "blind_scores_locked.json"
            score_path.write_text(
                json.dumps(scores, indent=2) + "\n",
                encoding="utf-8",
            )
            score_hash = hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest()
            score_hash_path = root / "score_bundle_sha256.txt"
            score_hash_path.write_text(
                score_hash + "\n",
                encoding="ascii",
            )

            report = evaluate_blind_void_campaign(
                score_bundle_path=score_path,
                committed_score_hash_path=score_hash_path,
                truth_key_path=truth_path,
                truth_salt_path=salt_path,
                truth_commitment_path=commitment_path,
                out_dir=root / "out",
            )
            self.assertTrue(
                report["workflow_qualification_rule"]["passed"]
            )
            self.assertEqual(
                report["positive_cases"]["correct_localized_detections"],
                6,
            )
            self.assertEqual(report["negative_cases"]["false_alarms"], 0)
            self.assertTrue(
                (root / "out" / "blind_qualification_results.md").exists()
            )

            score_hash_path.write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(ValueError):
                evaluate_blind_void_campaign(
                    score_bundle_path=score_path,
                    committed_score_hash_path=score_hash_path,
                    truth_key_path=truth_path,
                    truth_salt_path=salt_path,
                    truth_commitment_path=commitment_path,
                    out_dir=root / "tampered",
                )


if __name__ == "__main__":
    unittest.main()
