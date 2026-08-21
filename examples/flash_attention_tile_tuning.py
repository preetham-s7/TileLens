"""
Example 2: Tile Size Tuning and SRAM Exploration on NVIDIA H100 Hopper SM
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tilelens import get_hardware, Precision, TileSimulator, TileConfig, RooflineModel

def main():
    hw = get_hardware("h100")
    sim = TileSimulator(hw)
    roof_model = RooflineModel(hw, precision=Precision.BF16)

    M, N, K = 4096, 4096, 4096
    total_flops = 2.0 * M * N * K

    tile_candidates = [
        ("Small Tile (32x32x32)", TileConfig(tile_m=32, tile_n=32, tile_k=32, pipeline_stages=2)),
        ("Medium Tile (64x64x32)", TileConfig(tile_m=64, tile_n=64, tile_k=32, pipeline_stages=2)),
        ("Standard Tile (128x128x64)", TileConfig(tile_m=128, tile_n=128, tile_k=64, pipeline_stages=2)),
        ("Large Async TMA (256x128x64)", TileConfig(tile_m=256, tile_n=128, tile_k=64, pipeline_stages=3)),
        ("Oversized Tile (512x256x64)", TileConfig(tile_m=512, tile_n=256, tile_k=64, pipeline_stages=3)),
    ]

    print(f"=== NVIDIA H100 Hopper SM Tile Size Exploration (SRAM Limit: {hw.memory.sram_per_core_kb} KB) ===\n")
    print(f"{'Tile Configuration':<30} | {'SRAM Req (KB)':<14} | {'HBM Traffic (MB)':<16} | {'Intensity (FLOP/B)':<18} | {'Attainable TF':<14} | {'Status'}")
    print("-" * 115)

    for name, cfg in tile_candidates:
        res = sim.simulate_gemm(M, N, K, cfg)
        rf = roof_model.evaluate(total_flops, res.actual_hbm_bytes)

        status = "CRITICAL OVERFLOW" if res.sram_overflow else ("Compute-Bound" if rf.bound_type.value == "COMPUTE_BOUND" else "Memory-Bound")
        print(
            f"{name:<30} | {res.total_sram_required_kb:>12.1f} KB | {res.actual_hbm_bytes / 1e6:>14.2f} MB | {res.effective_arithmetic_intensity:>16.1f}   | {rf.attainable_tflops:>12.1f} TF | {status}"
        )

    print("-" * 115)
    print("\nTakeaway: Larger tiles increase SRAM data reuse (reducing HBM traffic), which raises operational intensity into the compute-bound regime until hitting physical SRAM capacity limits.")

if __name__ == "__main__":
    main()
