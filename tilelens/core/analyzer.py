"""
Co-Design Performance Analyzer and Optimization Recommender for AI Accelerators.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from tilelens.hardware.spec import HardwareSpec, Precision
from tilelens.core.roofline import RooflineModel, RooflineResult, BoundType
from tilelens.core.tile_simulator import TileSimulator, TileConfig, TileSimulationResult


@dataclass
class OptimizationSuggestion:
    category: str        # 'SRAM', 'BANDWIDTH', 'PRECISION', 'TILING'
    severity: str        # 'CRITICAL', 'WARNING', 'INFO'
    message: str
    action_item: str


@dataclass
class OptimizationReport:
    hardware_name: str
    roofline_result: RooflineResult
    tile_result: TileSimulationResult
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    optimal_tile_config: Optional[TileConfig] = None

    def summary(self) -> str:
        lines = [
            f"============================================================",
            f" TILELENS CO-DESIGN PERFORMANCE REPORT",
            f" Hardware: {self.hardware_name}",
            f" Workload: GEMM ({self.tile_result.gemm_m} x {self.tile_result.gemm_n} x {self.tile_result.gemm_k})",
            f"============================================================",
            "",
            self.roofline_result.summary(),
            "",
            self.tile_result.summary(),
            "",
            "--- Hardware Bottleneck & Optimization Diagnostics ---",
        ]
        if not self.suggestions:
            lines.append("  [OK] No major hardware bottlenecks detected. Tile configuration is well-optimized.")
        else:
            for s in self.suggestions:
                lines.append(f"  [{s.severity}] ({s.category}): {s.message}")
                lines.append(f"     -> Recommendation: {s.action_item}")
        
        if self.optimal_tile_config:
            lines.append("")
            lines.append(f"--- Recommended Optimal Tile Configuration ---")
            lines.append(
                f"  Tile (Tm, Tn, Tk): ({self.optimal_tile_config.tile_m}, {self.optimal_tile_config.tile_n}, {self.optimal_tile_config.tile_k}) "
                f"| Stages: {self.optimal_tile_config.pipeline_stages}"
            )
        lines.append("============================================================")
        return "\n".join(lines)


class PerformanceAnalyzer:
    """Analyzes execution and provides hardware-software co-design insights."""

    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.tile_sim = TileSimulator(hardware)

    def analyze_gemm(
        self,
        M: int,
        N: int,
        K: int,
        tile_config: Optional[TileConfig] = None,
    ) -> OptimizationReport:
        """Runs full analysis and produces optimization recommendations."""
        if tile_config is None:
            tile_config = TileConfig()

        # Run tile simulation
        tile_res = self.tile_sim.simulate_gemm(M, N, K, tile_config)

        # Run roofline model with actual memory traffic from tiling
        roofline_model = RooflineModel(self.hardware, precision=tile_config.precision)
        total_flops = 2.0 * M * N * K
        roofline_res = roofline_model.evaluate(total_flops, tile_res.actual_hbm_bytes)

        suggestions: List[OptimizationSuggestion] = []

        # 1. SRAM Overflow Diagnostic
        if tile_res.sram_overflow:
            suggestions.append(OptimizationSuggestion(
                category="SRAM",
                severity="CRITICAL",
                message=(
                    f"Tile configuration requires {tile_res.total_sram_required_kb:.1f} KB SRAM per core, "
                    f"exceeding the hardware limit of {tile_res.hardware_sram_limit_kb:.1f} KB."
                ),
                action_item="Reduce Tile M/N dimensions or decrease pipeline stages to avoid kernel launch failure.",
            ))

        # 2. Memory-Bound vs Compute-Bound
        if roofline_res.bound_type == BoundType.MEMORY_BOUND:
            gap = roofline_res.ridge_point - roofline_res.operational_intensity
            suggestions.append(OptimizationSuggestion(
                category="BANDWIDTH",
                severity="WARNING",
                message=(
                    f"Kernel is Memory-Bandwidth Bound (Operational Intensity is {roofline_res.operational_intensity:.1f} FLOPs/Byte, "
                    f"below the hardware ridge point of {roofline_res.ridge_point:.1f} FLOPs/Byte). Compute utilization is only {roofline_res.compute_efficiency_pct:.1f}%."
                ),
                action_item="Increase tile sizes (Tm, Tn) to increase SRAM data reuse, or switch to FP8/INT8 precision.",
            ))

        # 3. Precision Optimization
        if tile_config.precision in (Precision.FP32, Precision.FP16, Precision.BF16):
            if Precision.FP8 in self.hardware.compute.dense_tflops:
                fp8_peak = self.hardware.compute.dense_tflops[Precision.FP8]
                curr_peak = roofline_res.peak_tflops
                speedup_pot = fp8_peak / curr_peak if curr_peak > 0 else 1.0
                suggestions.append(OptimizationSuggestion(
                    category="PRECISION",
                    severity="INFO",
                    message=(
                        f"Hardware supports FP8 Tensor Cores ({fp8_peak:,.0f} TFLOPs vs current {curr_peak:,.0f} TFLOPs). "
                        f"Quantizing to FP8 cuts memory traffic by 50% and provides up to {speedup_pot:.1f}x speedup."
                    ),
                    action_item="Consider FP8 quantization for this GEMM layer.",
                ))

        # 4. Tail Wave / Occupancy Diagnostic
        if tile_res.tail_wave_penalty_pct > 25.0:
            suggestions.append(OptimizationSuggestion(
                category="TILING",
                severity="WARNING",
                message=(
                    f"Tail wave penalty is {tile_res.tail_wave_penalty_pct:.1f}%. A fraction of SMs/Cores remain idle in the final wave."
                ),
                action_item="Adjust Tile M/N block dimensions so total grid tiles evenly divides compute unit count.",
            ))

        # 5. Search for optimal tile configuration
        optimal_config = self._find_optimal_tile_config(M, N, K, tile_config.precision)

        return OptimizationReport(
            hardware_name=self.hardware.name,
            roofline_result=roofline_res,
            tile_result=tile_res,
            suggestions=suggestions,
            optimal_tile_config=optimal_config,
        )

    def _find_optimal_tile_config(self, M: int, N: int, K: int, precision: Precision) -> Optional[TileConfig]:
        """Heuristic search for best tile config that fits in SRAM and maximizes operational intensity."""
        candidate_tiles = [
            (256, 128, 64, 2),
            (128, 256, 64, 2),
            (128, 128, 64, 3),
            (128, 128, 64, 2),
            (128, 64, 64, 2),
            (64, 64, 32, 2),
            (32, 32, 16, 2),
        ]
        best_config = None
        best_intensity = -1.0

        for tm, tn, tk, stages in candidate_tiles:
            cfg = TileConfig(tile_m=tm, tile_n=tn, tile_k=tk, pipeline_stages=stages, precision=precision)
            res = self.tile_sim.simulate_gemm(M, N, K, cfg)
            if not res.sram_overflow:
                if res.effective_arithmetic_intensity > best_intensity:
                    best_intensity = res.effective_arithmetic_intensity
                    best_config = cfg

        return best_config
