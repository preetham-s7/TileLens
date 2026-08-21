"""
Roofline Model calculation engine for AI Accelerators (GPUs and TPUs).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from tilelens.hardware.spec import HardwareSpec, Precision


class BoundType(str, Enum):
    COMPUTE_BOUND = "COMPUTE_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    BALANCED = "BALANCED"


@dataclass
class RooflineResult:
    """Detailed results from a Roofline analysis."""
    hardware_name: str
    precision: Precision
    total_flops: float                 # Total floating point operations
    total_bytes: float                 # Total bytes moved across memory bus (HBM/DRAM)
    operational_intensity: float       # Arithmetic Intensity (FLOPs / Byte)
    ridge_point: float                 # Hardware knee/ridge point (Peak TFLOPs / Peak TB/s)
    peak_tflops: float                 # Hardware peak TFLOPs for target precision
    memory_bandwidth_gbs: float        # Hardware peak memory bandwidth (GB/s)
    attainable_tflops: float           # Max attainable TFLOPs under Roofline model
    attainable_gflops: float           # Max attainable GFLOPs
    execution_time_ms: float           # Ideal execution time in milliseconds
    bound_type: BoundType              # Compute-bound vs. Memory-bound
    compute_efficiency_pct: float      # % of peak compute utilized
    bandwidth_utilization_pct: float   # % of peak memory bandwidth required/utilized

    def summary(self) -> str:
        bound_str = "Compute-Bound" if self.bound_type == BoundType.COMPUTE_BOUND else "Memory-Bound (Bandwidth Throttled)"
        return (
            f"--- Roofline Performance Analysis [{self.hardware_name} | {self.precision.value.upper()}] ---\n"
            f"  Operational Intensity: {self.operational_intensity:.2f} FLOPs/Byte (Hardware Ridge: {self.ridge_point:.2f} FLOPs/Byte)\n"
            f"  Classification:        {bound_str}\n"
            f"  Attainable Compute:    {self.attainable_tflops:,.2f} TFLOPs ({self.compute_efficiency_pct:.1f}% of {self.peak_tflops:,.2f} TFLOPs peak)\n"
            f"  Memory Bandwidth:      {self.bandwidth_utilization_pct:.1f}% of {self.memory_bandwidth_gbs:,.0f} GB/s peak\n"
            f"  Ideal Execution Time:  {self.execution_time_ms:.4f} ms"
        )


class RooflineModel:
    """Calculates theoretical roofline limits for a given hardware target and precision."""

    def __init__(self, hardware: HardwareSpec, precision: Precision = Precision.BF16, sparse: bool = False):
        self.hardware = hardware
        self.precision = precision
        self.sparse = sparse
        self.peak_tflops = hardware.compute.get_peak_tflops(precision, sparse=sparse)
        self.bandwidth_gbs = hardware.memory.hbm_bandwidth_gbs
        self.bandwidth_tbs = hardware.memory.hbm_bandwidth_tbs

        # Ridge point = Peak FLOPs / Peak Memory Bandwidth
        # (TFLOPs * 10^12) / (TB/s * 10^12) = FLOPs / Byte
        self.ridge_point = (self.peak_tflops * 1000.0) / self.bandwidth_gbs

    def evaluate(self, total_flops: float, total_bytes: float) -> RooflineResult:
        """Evaluate a specific workload with given FLOPs and memory footprint (Bytes)."""
        if total_bytes <= 0:
            raise ValueError("Total bytes transferred must be greater than 0.")
        if total_flops < 0:
            raise ValueError("Total FLOPs cannot be negative.")

        # Operational Intensity (FLOPs / Byte)
        operational_intensity = total_flops / total_bytes

        # Attainable TFLOPs = min(Peak TFLOPs, Operational Intensity * Peak TB/s)
        memory_bound_tflops = operational_intensity * self.bandwidth_tbs
        attainable_tflops = min(self.peak_tflops, memory_bound_tflops)
        attainable_gflops = attainable_tflops * 1000.0

        # Ideal execution time = total_flops / (attainable_tflops * 10^12) seconds
        if attainable_tflops > 0:
            execution_time_ms = (total_flops / (attainable_tflops * 1e12)) * 1000.0
        else:
            execution_time_ms = 0.0

        # Classification
        if operational_intensity >= self.ridge_point:
            bound_type = BoundType.COMPUTE_BOUND
        else:
            bound_type = BoundType.MEMORY_BOUND

        compute_eff = (attainable_tflops / self.peak_tflops) * 100.0
        
        # Effective bandwidth required = total_bytes / (execution_time in sec)
        if execution_time_ms > 0:
            req_bandwidth_gbs = (total_bytes / (execution_time_ms / 1000.0)) / 1e9
            bw_util = min(100.0, (req_bandwidth_gbs / self.bandwidth_gbs) * 100.0)
        else:
            bw_util = 0.0

        return RooflineResult(
            hardware_name=self.hardware.name,
            precision=self.precision,
            total_flops=total_flops,
            total_bytes=total_bytes,
            operational_intensity=operational_intensity,
            ridge_point=self.ridge_point,
            peak_tflops=self.peak_tflops,
            memory_bandwidth_gbs=self.bandwidth_gbs,
            attainable_tflops=attainable_tflops,
            attainable_gflops=attainable_gflops,
            execution_time_ms=execution_time_ms,
            bound_type=bound_type,
            compute_efficiency_pct=compute_eff,
            bandwidth_utilization_pct=bw_util,
        )


def calculate_gemm_intensity(
    M: int,
    N: int,
    K: int,
    precision: Precision = Precision.BF16,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
) -> tuple[float, float, float]:
    """
    Computes (FLOPs, Bytes, Arithmetic Intensity) for a GEMM: C [M, N] = A [M, K] x B [K, N]
    
    If tile dimensions are provided, accounts for SRAM tile reuse.
    Without tiling (ideal minimum memory footprint):
      Bytes = (M*K + K*N + M*N) * byte_size
    With tiling:
      Bytes = ((M*N/tile_n) * K + (M*N/tile_m) * K + M*N) * byte_size
    """
    bytes_per_elem = precision.byte_size
    total_flops = 2.0 * M * N * K

    if tile_m is None or tile_n is None or tile_k is None:
        # Ideal lower bound on memory transfer (each matrix element loaded/stored exactly once from HBM)
        bytes_a = M * K * bytes_per_elem
        bytes_b = K * N * bytes_per_elem
        bytes_c = M * N * bytes_per_elem
        total_bytes = bytes_a + bytes_b + bytes_c
    else:
        # Realistic tiled memory transfer from HBM
        num_tiles_m = (M + tile_m - 1) // tile_m
        num_tiles_n = (N + tile_n - 1) // tile_n
        num_tiles_k = (K + tile_k - 1) // tile_k

        # Matrix A tile is reloaded for each N-tile
        total_bytes_a = num_tiles_n * (M * K) * bytes_per_elem
        # Matrix B tile is reloaded for each M-tile
        total_bytes_b = num_tiles_m * (K * N) * bytes_per_elem
        # Matrix C is written once
        total_bytes_c = M * N * bytes_per_elem
        total_bytes = total_bytes_a + total_bytes_b + total_bytes_c

    intensity = total_flops / total_bytes if total_bytes > 0 else 0.0
    return total_flops, total_bytes, intensity
