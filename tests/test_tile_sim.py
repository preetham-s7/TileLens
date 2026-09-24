"""
Unit tests for Tile Simulator & Memory Hierarchy calculations.
"""

import unittest
from tilelens.hardware.database import get_hardware
from tilelens.hardware.spec import Precision
from tilelens.core.tile_simulator import TileSimulator, TileConfig
from tilelens.core.analyzer import PerformanceAnalyzer


class TestTileSimulator(unittest.TestCase):

    def setUp(self):
        self.h100 = get_hardware("h100")
        self.sim = TileSimulator(self.h100)

    def test_sram_allocation_and_utilization(self):
        """Verify SRAM calculation for 128x128x64 BF16 with 2 pipeline stages."""
        # A: 128 * 64 * 2 bytes = 16,384 bytes (16 KB)
        # B: 64 * 128 * 2 bytes = 16,384 bytes (16 KB)
        # 2 stages: (16 + 16) * 2 = 64 KB
        # C accum: 128 * 128 * 4 bytes (FP32) = 65,536 bytes (64 KB)
        # Total SRAM required = 128 KB
        cfg = TileConfig(tile_m=128, tile_n=128, tile_k=64, pipeline_stages=2, precision=Precision.BF16)
        res = self.sim.simulate_gemm(4096, 4096, 4096, cfg)

        self.assertAlmostEqual(res.total_sram_required_kb, 128.0, places=1)
        self.assertFalse(res.sram_overflow)
        # H100 SM has 228 KB, so 128 KB fits exactly 1 active block per SM
        self.assertEqual(res.active_blocks_per_sm, 1)

    def test_sram_overflow_detection(self):
        """Oversized tile should trigger sram_overflow flag."""
        # 512x512x128 tile requires > 1.5 MB SRAM, far exceeding 228 KB SM limit
        cfg = TileConfig(tile_m=512, tile_n=512, tile_k=128, pipeline_stages=2, precision=Precision.BF16)
        res = self.sim.simulate_gemm(4096, 4096, 4096, cfg)

        self.assertTrue(res.sram_overflow)
        self.assertEqual(res.active_blocks_per_sm, 0)

    def test_analyzer_suggestions(self):
        """Analyzer should detect memory bound status and propose optimizations."""
        analyzer = PerformanceAnalyzer(self.h100)
        # Small tile has high HBM reload and low arithmetic intensity
        small_tile = TileConfig(tile_m=32, tile_n=32, tile_k=32, pipeline_stages=2, precision=Precision.BF16)
        report = analyzer.analyze_gemm(4096, 4096, 4096, small_tile)

        self.assertIsNotNone(report.optimal_tile_config)
        self.assertTrue(any(s.category == "BANDWIDTH" for s in report.suggestions))

    def test_invalid_tile_settings(self):
        for field in ("tile_m", "tile_n", "tile_k", "pipeline_stages"):
            for value in (0, -1, 1.5, True, float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field + " must be a positive integer"):
                        TileConfig(**{field: value})

    def test_invalid_matrix_dimensions(self):
        for axis in range(3):
            for value in (0, -1, 1.5, True):
                dimensions = [4096, 4096, 4096]
                dimensions[axis] = value
                with self.subTest(axis=axis, value=value):
                    with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                        self.sim.simulate_gemm(*dimensions)

    def test_mutated_tile_settings_are_validated(self):
        cfg = TileConfig()
        cfg.pipeline_stages = -1
        with self.assertRaisesRegex(ValueError, "pipeline_stages"):
            self.sim.simulate_gemm(4096, 4096, 4096, cfg)


if __name__ == "__main__":
    unittest.main()
