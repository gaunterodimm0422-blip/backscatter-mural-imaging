import unittest

from backscatter.voids import _control_centers, compute_void_cnr


class ConcreteVoidTests(unittest.TestCase):
    def test_equal_normalized_counts_have_zero_cnr(self):
        metrics = compute_void_cnr(
            void_roi_count=100,
            reference_roi_count=200,
            void_control_count=500,
            reference_control_count=1000,
        )
        self.assertAlmostEqual(metrics["control_normalization_scale"], 0.5)
        self.assertAlmostEqual(metrics["event_deficit"], 0.0)
        self.assertAlmostEqual(metrics["contrast"], 0.0)
        self.assertAlmostEqual(metrics["cnr"], 0.0)

    def test_count_deficit_has_positive_contrast_and_cnr(self):
        metrics = compute_void_cnr(
            void_roi_count=60,
            reference_roi_count=200,
            void_control_count=500,
            reference_control_count=1000,
        )
        self.assertAlmostEqual(metrics["expected_no_void_roi_count"], 100.0)
        self.assertAlmostEqual(metrics["contrast"], 0.4)
        self.assertGreater(metrics["cnr"], 0.0)

    def test_empty_control_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_void_cnr(
                void_roi_count=1,
                reference_roi_count=1,
                void_control_count=0,
                reference_control_count=1,
            )

    def test_small_void_control_layout_keeps_30_mm_floor(self):
        centers = _control_centers(
            void_x_mm=0.0,
            radius_mm=7.5,
            slab_x_interval_mm=(-100.0, 100.0),
        )
        self.assertEqual(centers, [-60.0, -30.0, 30.0, 60.0])

    def test_large_void_control_layout_is_nearest_non_overlapping(self):
        centers = _control_centers(
            void_x_mm=0.0,
            radius_mm=15.0,
            slab_x_interval_mm=(-100.0, 100.0),
        )
        self.assertEqual(centers, [-60.0, -30.0, 30.0, 60.0])

    def test_control_layout_is_clipped_to_slab(self):
        centers = _control_centers(
            void_x_mm=0.0,
            radius_mm=15.0,
            slab_x_interval_mm=(-50.0, 50.0),
        )
        self.assertEqual(centers, [-30.0, 30.0])


if __name__ == "__main__":
    unittest.main()
