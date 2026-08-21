"""
Example 1: Comparing LLM Projection Layers on NVIDIA H100 vs Google TPU v5e & v5p
"""

import sys
import os
# Add parent dir to path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tilelens import get_hardware, Precision, PerformanceAnalyzer, TileConfig
from tilelens.viz import save_html_report

def main():
    print("=" * 70)
    print("TileLens Case Study: LLaMA-3 70B FFN Up-Projection Layer")
    print("=" * 70)
    
    # LLaMA-3 70B Hidden Size: d_model = 8192, intermediate_size = 28672
    # Batch size * sequence length = 4096 tokens
    M = 4096
    N = 28672
    K = 8192

    devices = ["h100", "tpu_v5e", "tpu_v5p", "a100"]
    roofline_results = []
    tile_results = []

    for dev_name in devices:
        hw = get_hardware(dev_name)
        analyzer = PerformanceAnalyzer(hw)

        # Standard 128x128x64 BF16 Tile with 2-stage buffering
        tile_cfg = TileConfig(tile_m=128, tile_n=128, tile_k=64, pipeline_stages=2, precision=Precision.BF16)
        report = analyzer.analyze_gemm(M, N, K, tile_cfg)

        roofline_results.append(report.roofline_result)
        tile_results.append(report.tile_result)

        print(report.summary())
        print()

    # Save interactive dashboard
    output_html = "llama3_ffn_comparison.html"
    save_html_report(output_html, roofline_results, tile_results, title="LLaMA-3 70B FFN Layer: H100 vs TPU v5e vs TPU v5p")
    print(f"[Done] Exported interactive dashboard to: {output_html}")


if __name__ == "__main__":
    main()
