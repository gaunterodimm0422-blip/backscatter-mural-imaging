import unittest

import numpy as np

from backscatter.electronics import (
    FWHM_TO_SIGMA,
    energy_sigma_keV,
    smear_energy_keV,
)


class ElectronicsTests(unittest.TestCase):
    def test_ten_percent_resolution_has_51p1_keV_fwhm_at_511(self):
        sigma = energy_sigma_keV(
            np.asarray([511.0]),
            fwhm_fraction_at_511=0.10,
        )[0]
        self.assertAlmostEqual(sigma * FWHM_TO_SIGMA, 51.1, places=8)

    def test_absolute_fwhm_scales_with_square_root_energy(self):
        sigma = energy_sigma_keV(
            np.asarray([511.0, 127.75]),
            fwhm_fraction_at_511=0.10,
        )
        self.assertAlmostEqual(sigma[1] / sigma[0], 0.5, places=8)

    def test_zero_resolution_preserves_energy(self):
        energy = np.asarray([100.0, 511.0])
        smeared = smear_energy_keV(
            energy,
            fwhm_fraction_at_511=0.0,
            standard_normal=np.asarray([10.0, -10.0]),
        )
        np.testing.assert_array_equal(smeared, energy)


if __name__ == "__main__":
    unittest.main()
