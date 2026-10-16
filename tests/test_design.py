import unittest

from backscatter.config import load_config
from backscatter.design import d1_mass_kg, d2_active_mass_kg


class DetectorDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/baseline.json")

    def test_d1_mass_decreases_with_radius(self):
        mass_100 = d1_mass_kg(self.config, 100.0)
        mass_200 = d1_mass_kg(self.config, 200.0)
        self.assertGreater(mass_200, mass_100)
        self.assertAlmostEqual(mass_100, 2.999, delta=0.01)
        self.assertAlmostEqual(mass_200, 12.372, delta=0.02)

    def test_d2_active_mass(self):
        self.assertAlmostEqual(
            d2_active_mass_kg(self.config),
            0.2867,
            delta=0.001,
        )


if __name__ == "__main__":
    unittest.main()
