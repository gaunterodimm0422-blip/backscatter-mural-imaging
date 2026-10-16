import unittest

import numpy as np

from backscatter.kinematics import (
    compton_angle_from_energy_keV,
    compton_scattered_energy_keV,
    synthetic_selftest,
)


class KinematicsTests(unittest.TestCase):
    def test_backscatter_energy_limits(self):
        energy = compton_scattered_energy_keV(np.array([90.0, 180.0]))
        self.assertAlmostEqual(float(energy[0]), 255.5, places=9)
        self.assertAlmostEqual(float(energy[1]), 511.0 / 3.0, places=9)

    def test_invalid_low_energy_is_not_clipped_to_180(self):
        theta, valid = compton_angle_from_energy_keV(np.array([160.0, 200.0]))
        self.assertFalse(bool(valid[0]))
        self.assertTrue(np.isnan(theta[0]))
        self.assertTrue(bool(valid[1]))

    def test_closed_form_solver_recovers_truth(self):
        report = synthetic_selftest(n=5000, seed=123)
        self.assertTrue(report["passed"])
        self.assertLess(report["max_vertex_error_mm"], 1e-8)


if __name__ == "__main__":
    unittest.main()

