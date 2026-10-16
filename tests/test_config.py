import copy
import unittest
from pathlib import Path

from backscatter.config import ConfigError, load_config, validate_config
from backscatter.experiments import (
    build_blind_analysis_config,
    build_concrete_void_config,
    build_config_variant,
    build_standard_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_baseline_config_is_consistent(self):
        config = load_config(ROOT / "configs" / "baseline.json")
        self.assertEqual(config["phantom"]["z_interval_mm"], [-100.0, 0.0])
        self.assertEqual(config["detectors"]["d2"]["pitch_mm"][0], 1.0)

    def test_geometry_mismatch_is_rejected(self):
        config = load_config(ROOT / "configs" / "baseline.json")
        broken = copy.deepcopy(config)
        broken["phantom"]["z_interval_mm"] = [-225.0, -125.0]
        with self.assertRaises(ConfigError):
            validate_config(broken)

    def test_standard_target_config_uses_target_truth_and_depth(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_standard_config(
            base,
            kind="double_point",
            depth_mm=30.0,
            thickness_mm=1.0,
            radius_mm=1.0,
            separation_mm=10.0,
            primary_events=1000,
        )
        self.assertEqual(config["phantom"]["z_interval_mm"], [-30.5, -29.5])
        self.assertEqual(
            config["phantom"]["target_centers_mm"],
            [[-5.0, 0.0, -30.0], [5.0, 0.0, -30.0]],
        )
        self.assertEqual(
            config["selection"]["phantom_truth_volume_contains"], "target"
        )
        self.assertEqual(config["phantom"]["environment_material"], "Vacuum")
        self.assertEqual(
            config["phantom"]["measurement_role"],
            "intrinsic_geometric_response",
        )
        self.assertIn(
            "axial reference plane",
            config["project"]["coordinate_convention"],
        )

    def test_rod_config_uses_diameter_as_z_extent(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_standard_config(
            base,
            kind="rod",
            depth_mm=30.0,
            radius_mm=1.0,
            lateral_mm=300.0,
            primary_events=1000,
        )
        self.assertEqual(config["phantom"]["size_mm"], [2.0, 300.0, 2.0])
        self.assertEqual(config["phantom"]["z_interval_mm"], [-31.0, -29.0])
        self.assertEqual(config["phantom"]["diameter_mm"], 2.0)

    def test_standard_variant_records_seed_and_physical_d1_radius(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_standard_config(
            base,
            kind="thin_layer",
            depth_mm=30.0,
            primary_events=2_000_000,
            random_seed=24061874,
            d1_rmax_mm=150.0,
        )
        self.assertEqual(config["simulation"]["random_seed"], 24061874)
        self.assertEqual(config["detectors"]["d1"]["radius_mm"], [20.0, 150.0])
        self.assertEqual(config["selection"]["d1_radius_mm"], [20.0, 150.0])
        self.assertEqual(
            config["sweep_grid"]["d1_radius_max_mm"],
            [75.0, 100.0, 125.0, 150.0],
        )
        self.assertIn("seed_24061874", config["project"]["baseline_id"])
        self.assertIn("d1r_150mm", config["project"]["baseline_id"])

    def test_generic_variant_updates_concrete_seed_and_radius(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_config_variant(
            base,
            primary_events=2_000_000,
            random_seed=24061874,
            d1_rmax_mm=175.0,
            variant_id="concrete_seed2_d1r175",
        )
        self.assertEqual(config["simulation"]["primary_events"], 2_000_000)
        self.assertEqual(config["simulation"]["random_seed"], 24061874)
        self.assertEqual(config["detectors"]["d1"]["radius_mm"], [20.0, 175.0])
        self.assertEqual(config["selection"]["d1_radius_mm"], [20.0, 175.0])
        self.assertEqual(
            config["project"]["baseline_id"],
            "concrete_seed2_d1r175",
        )

    def test_concrete_void_config_records_nested_geometry(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_concrete_void_config(
            base,
            depth_mm=30.0,
            radius_mm=5.0,
            length_mm=300.0,
            x_mm=10.0,
            primary_events=2000,
            random_seed=24061875,
        )
        void = config["phantom"]["void"]
        self.assertEqual(void["center_mm"], [10.0, 0.0, -30.0])
        self.assertEqual(void["local_z_mm"], 20.0)
        self.assertEqual(void["diameter_mm"], 10.0)
        self.assertEqual(
            config["selection"]["phantom_truth_policy"],
            "ignore",
        )

    def test_concrete_void_must_fit_inside_slab(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        with self.assertRaises(ValueError):
            build_concrete_void_config(
                base,
                depth_mm=3.0,
                radius_mm=5.0,
            )

    def test_blind_analysis_config_contains_no_answer(self):
        base = load_config(ROOT / "configs" / "baseline.json")
        config = build_blind_analysis_config(
            base,
            primary_events=1_000_000,
        )
        self.assertEqual(config["simulation"]["primary_events"], 1_000_000)
        self.assertEqual(
            config["selection"]["phantom_truth_policy"],
            "ignore",
        )
        self.assertNotIn("void", config["phantom"])
        self.assertEqual(
            config["phantom"]["measurement_role"],
            "blind_void_detection",
        )


if __name__ == "__main__":
    unittest.main()
