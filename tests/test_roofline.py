"""
Unit tests for Roofline calculation logic.
"""

import unittest
from tilelens.hardware.database import get_hardware
from tilelens.hardware.spec import Precision
from tilelens.core.roofline import RooflineModel, BoundType, calculate_gemm_intensity


class TestRoofline(unittest.TestCase):

    def setUp(self):
        self.h100 = get_hardware("h100")
        self.tpu_v5e = get_hardware("tpu_v5e")

    def test_h100_ridge_point(self):
        """H100 SXM (989 TFLOPs BF16 / 3.35 TB/s HBM3) has ridge point ~295 FLOPs/Byte."""
        model = RooflineModel(self.h100, precision=Precision.BF16)
        expected_ridge = (989.0 * 1000.0) / 3350.0
        self.assertAlmostEqual(model.ridge_point, expected_ridge, places=2)

    def test_compute_bound_evaluation(self):
        """High operational intensity should be compute-bound and capped at peak TFLOPs."""
        model = RooflineModel(self.h100, precision=Precision.BF16)
        # Intensity = 1000 FLOPs / Byte (well above 295.2)
        res = model.evaluate(total_flops=1e12, total_bytes=1e9)
        self.assertEqual(res.bound_type, BoundType.COMPUTE_BOUND)
        self.assertAlmostEqual(res.attainable_tflops, self.h100.compute.get_peak_tflops(Precision.BF16), places=1)
        self.assertAlmostEqual(res.compute_efficiency_pct, 100.0, places=1)

    def test_memory_bound_evaluation(self):
        """Low operational intensity should be memory-bound and throttled by bandwidth."""
        model = RooflineModel(self.h100, precision=Precision.BF16)
        # Intensity = 10 FLOPs / Byte (below 295.2)
        res = model.evaluate(total_flops=1e10, total_bytes=1e9)
        self.assertEqual(res.bound_type, BoundType.MEMORY_BOUND)
        expected_tflops = 10.0 * (self.h100.memory.hbm_bandwidth_gbs / 1000.0)
        self.assertAlmostEqual(res.attainable_tflops, expected_tflops, places=1)
        self.assertLess(res.compute_efficiency_pct, 100.0)

    def test_tpu_v5e_evaluation(self):
        """TPU v5e (197 TFLOPs BF16 / 819 GB/s HBM2e)."""
        model = RooflineModel(self.tpu_v5e, precision=Precision.BF16)
        res = model.evaluate(total_flops=2e12, total_bytes=2e9)
        self.assertIsNotNone(res)
        self.assertGreater(res.attainable_tflops, 0)


if __name__ == "__main__":
    unittest.main()
