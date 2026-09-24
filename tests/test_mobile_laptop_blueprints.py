"""
Unit tests for mobile chips, laptop PC processors, silicon blueprints, and operational profiles.
"""

import unittest
from tilelens.hardware.database import get_hardware, list_available_hardware, HARDWARE_DATABASE
from tilelens.hardware.spec import Precision, HardwareType
from tilelens.core.roofline import RooflineModel, BoundType
from tilelens.core.tile_simulator import TileSimulator, TileConfig


class TestMobileLaptopBlueprints(unittest.TestCase):

    def test_mobile_chip_resolution(self):
        """Test alias and lookup resolution for mobile phone chips."""
        a17 = get_hardware("a17")
        self.assertEqual(a17.name, "Apple A17 Pro")
        self.assertEqual(a17.hardware_type, HardwareType.MOBILE_SOC)
        self.assertEqual(a17.process_node_nm, 3.0)
        self.assertEqual(a17.npu_tops, 35.0)

        snapdragon = get_hardware("snapdragon")
        self.assertEqual(snapdragon.name, "Qualcomm Snapdragon 8 Gen 3")
        self.assertEqual(snapdragon.hardware_type, HardwareType.MOBILE_SOC)
        self.assertEqual(snapdragon.npu_tops, 45.0)

        tensor = get_hardware("pixel9")
        self.assertEqual(tensor.name, "Google Tensor G4")
        self.assertEqual(tensor.hardware_type, HardwareType.MOBILE_SOC)

        dimensity = get_hardware("dimensity9300")
        self.assertEqual(dimensity.name, "MediaTek Dimensity 9300")
        self.assertEqual(dimensity.hardware_type, HardwareType.MOBILE_SOC)

    def test_laptop_chip_resolution(self):
        """Test alias and lookup resolution for laptop / PC processors."""
        m4 = get_hardware("m4")
        self.assertEqual(m4.name, "Apple M4")
        self.assertEqual(m4.hardware_type, HardwareType.LAPTOP_SOC)
        self.assertEqual(m4.memory.hbm_bandwidth_gbs, 120.0)

        m3_max = get_hardware("m3max")
        self.assertEqual(m3_max.name, "Apple M3 Max")
        self.assertEqual(m3_max.memory.hbm_capacity_gb, 128.0)

        lunar = get_hardware("lunarlake")
        self.assertEqual(lunar.name, "Intel Core Ultra 7 288V")
        self.assertEqual(lunar.npu_tops, 48.0)

        strix = get_hardware("ryzenai")
        self.assertEqual(strix.name, "AMD Ryzen AI 9 HX 370")
        self.assertEqual(strix.npu_tops, 50.0)

        rtx4060 = get_hardware("rtx4060mobile")
        self.assertEqual(rtx4060.name, "NVIDIA GeForce RTX 4060 Laptop")
        self.assertEqual(rtx4060.hardware_type, HardwareType.GPU)

    def test_blueprint_blocks_and_capabilities(self):
        """Verify that every chip in the database has blueprint blocks and operational capabilities."""
        for key, spec in HARDWARE_DATABASE.items():
            self.assertIsNotNone(spec.name, f"Chip {key} missing name")
            self.assertGreater(len(spec.blueprint_blocks), 0, f"Chip {key} has no blueprint blocks")
            self.assertGreater(len(spec.capabilities), 0, f"Chip {key} has no capabilities")

            # Check block properties
            for b in spec.blueprint_blocks:
                self.assertIsNotNone(b.name)
                self.assertIn(b.category, ["compute", "tensor", "npu", "cache", "memory_ctrl", "interconnect", "media", "io"])
                self.assertIsNotNone(b.description)

            # Check blueprint summary method executes without error
            summary_text = spec.blueprint_summary()
            self.assertIn("CHIP BLUEPRINT", summary_text)
            self.assertIn("What Else This Chip Does", summary_text)

    def test_apple_a17_roofline_and_tiling(self):
        """Verify roofline and tile simulation work properly on mobile SoC (Apple A17 Pro)."""
        a17 = get_hardware("a17")
        model = RooflineModel(a17, precision=Precision.FP16)
        
        # At intensity = 100 FLOPs/B (above ridge = 4300/51.2 = ~84 FLOPs/B)
        res_compute = model.evaluate(total_flops=1e11, total_bytes=1e9)
        self.assertEqual(res_compute.bound_type, BoundType.COMPUTE_BOUND)
        self.assertAlmostEqual(res_compute.attainable_tflops, a17.compute.get_peak_tflops(Precision.FP16), places=1)

        # At intensity = 10 FLOPs/B (below ridge, memory-bound)
        res_mem = model.evaluate(total_flops=1e10, total_bytes=1e9)
        self.assertEqual(res_mem.bound_type, BoundType.MEMORY_BOUND)

        # Tile simulator
        sim = TileSimulator(a17)
        tile_cfg = TileConfig(tile_m=64, tile_n=64, tile_k=32, pipeline_stages=2, precision=Precision.FP16)
        sim_res = sim.simulate_gemm(1024, 1024, 1024, tile_cfg)
        self.assertFalse(sim_res.sram_overflow)
        self.assertGreater(sim_res.total_sram_required_kb, 0)

    def test_apple_m4_roofline(self):
        """Verify Apple M4 roofline model and unified memory throughput."""
        m4 = get_hardware("m4")
        model = RooflineModel(m4, precision=Precision.FP16)
        ridge = (8.6 * 1000.0) / 120.0  # ~71.67 FLOPs/Byte
        self.assertAlmostEqual(model.ridge_point, ridge, places=2)


if __name__ == "__main__":
    unittest.main()
