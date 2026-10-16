import unittest

import numpy as np

from backscatter.config import load_config
from backscatter.response import (
    apply_detector_response,
    quantize_positions_to_voxel_centers,
)


class DetectorResponseTests(unittest.TestCase):
    def test_position_quantization_uses_three_dimensional_voxel_centres(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 10.0],
                [1.54, -0.01, 14.99],
            ]
        )
        result = quantize_positions_to_voxel_centers(
            positions,
            resolution_mm=(1.55, 1.55, 5.0),
        )
        expected = np.asarray(
            [
                [0.775, 0.775, 12.5],
                [0.775, -0.775, 12.5],
            ]
        )
        np.testing.assert_allclose(result, expected)

    def test_updated_hardware_response_is_deterministic_and_nonmutating(self):
        config = load_config("configs/hardware_20260731.json")
        n = 100_000
        pairs = {
            "key": np.arange(n),
            "d1_energy_keV": np.full(n, 511.0),
            "d2_energy_keV": np.full(n, 511.0),
            "d1_x_mm": np.linspace(-3.0, 3.0, n),
            "d1_y_mm": np.linspace(-2.0, 2.0, n),
            "d1_z_mm": np.linspace(10.0, 25.0, n),
            "d2_x_mm": np.linspace(-3.0, 3.0, n),
            "d2_y_mm": np.linspace(-2.0, 2.0, n),
            "d2_z_mm": np.linspace(50.0, 70.0, n),
        }
        original_d1 = pairs["d1_energy_keV"].copy()
        first, report = apply_detector_response(pairs, config, random_seed=9)
        second, _ = apply_detector_response(pairs, config, random_seed=9)

        np.testing.assert_array_equal(pairs["d1_energy_keV"], original_d1)
        np.testing.assert_array_equal(
            first["d1_energy_keV"], second["d1_energy_keV"]
        )
        self.assertAlmostEqual(
            np.std(first["d1_energy_keV"]) * 2.354820045,
            0.08 * 511.0,
            delta=0.35,
        )
        self.assertAlmostEqual(
            np.std(first["d2_energy_keV"]) * 2.354820045,
            0.12 * 511.0,
            delta=0.50,
        )
        self.assertAlmostEqual(
            report["detectors"]["d1"]["energy_fwhm_keV_at_511"],
            40.88,
        )
        self.assertAlmostEqual(
            report["detectors"]["d2"]["energy_fwhm_keV_at_511"],
            61.32,
        )


if __name__ == "__main__":
    unittest.main()
