"""
Tile Memory Hierarchy Simulator for GPU Tensor Cores and TPU Systolic Arrays.
"""

import math
from dataclasses import dataclass
from typing import Optional
from tilelens.hardware.spec import HardwareSpec, Precision


@dataclass
class TileConfig:
    """Configuration for a Matrix Tile computation."""
    tile_m: int = 128                  # Block size along M dimension
    tile_n: int = 128                  # Block size along N dimension
    tile_k: int = 64                   # Block size along K dimension (reduction axis)
    pipeline_stages: int = 2           # Multi-buffering depth (2 for double buffer, 3-4 for Hopper async TMA)
    precision: Precision = Precision.BF16
    accum_precision: Precision = Precision.FP32  # Accumulators usually kept in FP32


@dataclass
class TileSimulationResult:
    """Detailed metrics from tile memory allocation and movement simulation."""
    hardware_name: str
    tile_config: TileConfig
    gemm_m: int
    gemm_n: int
    gemm_k: int
    
    # SRAM Footprint
    a_tile_sram_kb: float             # Memory for A tile per stage
    b_tile_sram_kb: float             # Memory for B tile per stage
    c_tile_sram_kb: float             # Memory for C tile accumulator
    total_sram_required_kb: float     # Total SRAM needed per compute unit (SM/Core)
    hardware_sram_limit_kb: float     # Available hardware SRAM per SM/Core
    sram_utilization_pct: float       # Percentage of SRAM used
    sram_overflow: bool               # True if tile configuration exceeds hardware limit
    
    # Concurrency and Waves
    total_grid_tiles: int             # Total tiles needed to cover (M x N)
    active_blocks_per_sm: int         # Max concurrent blocks that fit in SRAM per SM
    total_waves: float                # Grid waves across all SMs/Cores
    tail_wave_penalty_pct: float      # Tail effect efficiency loss
    
    # Memory Traffic & Reuse
    ideal_hbm_bytes: float            # Zero-redundancy minimum bytes
    actual_hbm_bytes: float           # Actual bytes transferred accounting for tile re-reads
    tile_reuse_factor: float          # Actual / Ideal memory amplification
    effective_arithmetic_intensity: float  # FLOPs / Actual Bytes

    def summary(self) -> str:
        overflow_warning = " [CRITICAL: SRAM OVERFLOW!]" if self.sram_overflow else " [SRAM OK]"
        return (
            f"--- Tile Memory Simulation [{self.hardware_name}] ---\n"
            f"  Matrix Size (M,N,K):   ({self.gemm_m}, {self.gemm_n}, {self.gemm_k}) | Precision: {self.tile_config.precision.value.upper()}\n"
            f"  Tile Config (Tm,Tn,Tk): ({self.tile_config.tile_m}, {self.tile_config.tile_n}, {self.tile_config.tile_k}) | Stages: {self.tile_config.pipeline_stages}\n"
            f"  SRAM Required / SM:    {self.total_sram_required_kb:.1f} KB / {self.hardware_sram_limit_kb:.1f} KB ({self.sram_utilization_pct:.1f}%){overflow_warning}\n"
            f"  Concurrent Blocks / SM: {self.active_blocks_per_sm} blocks\n"
            f"  Grid Tiles:            {self.total_grid_tiles} blocks (Waves across chip: {self.total_waves:.2f})\n"
            f"  HBM Memory Traffic:    {self.actual_hbm_bytes / 1e6:,.2f} MB (Tile Reuse Factor: {self.tile_reuse_factor:.2f}x vs Ideal {self.ideal_hbm_bytes / 1e6:,.2f} MB)\n"
            f"  Effective Intensity:   {self.effective_arithmetic_intensity:.2f} FLOPs/Byte"
        )


class TileSimulator:
    """Simulates tile partitioning and memory hierarchy interaction."""

    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware

    def simulate_gemm(
        self,
        M: int,
        N: int,
        K: int,
        tile_config: Optional[TileConfig] = None,
    ) -> TileSimulationResult:
        """Simulate execution of GEMM on the hardware with specified tile configuration."""
        if tile_config is None:
            tile_config = TileConfig()

        elem_bytes = tile_config.precision.byte_size
        accum_bytes = tile_config.accum_precision.byte_size

        # 1. SRAM allocation per SM/Core
        # Stage buffer for A: Tm * Tk
        a_tile_bytes = tile_config.tile_m * tile_config.tile_k * elem_bytes
        # Stage buffer for B: Tk * Tn
        b_tile_bytes = tile_config.tile_k * tile_config.tile_n * elem_bytes
        # Accumulator for C: Tm * Tn
        c_tile_bytes = tile_config.tile_m * tile_config.tile_n * accum_bytes

        total_sram_bytes = ((a_tile_bytes + b_tile_bytes) * tile_config.pipeline_stages) + c_tile_bytes
        total_sram_kb = total_sram_bytes / 1024.0
        hw_sram_kb = self.hardware.memory.sram_per_core_kb

        sram_overflow = total_sram_kb > hw_sram_kb
        sram_util_pct = min(100.0, (total_sram_kb / hw_sram_kb) * 100.0) if hw_sram_kb > 0 else 100.0

        # 2. Concurrency and Grid Sizing
        grid_m = math.ceil(M / tile_config.tile_m)
        grid_n = math.ceil(N / tile_config.tile_n)
        total_grid_tiles = grid_m * grid_n

        if sram_overflow or total_sram_kb <= 0:
            active_blocks_per_sm = 0
        else:
            active_blocks_per_sm = max(1, int(hw_sram_kb // total_sram_kb))

        num_sm = max(1, self.hardware.memory.num_compute_units)
        max_concurrent_grid = num_sm * max(1, active_blocks_per_sm)
        total_waves = total_grid_tiles / max_concurrent_grid
        full_waves = math.floor(total_waves)
        remainder = total_grid_tiles % max_concurrent_grid
        
        if remainder > 0:
            tail_wave_penalty = (1.0 - (remainder / max_concurrent_grid)) * 100.0
        else:
            tail_wave_penalty = 0.0

        # 3. Data movement (HBM -> SRAM)
        ideal_bytes = (M * K + K * N + M * N) * elem_bytes
        
        # Each tile in grid (grid_m * grid_n) loads its slice of K in steps of tile_k
        k_steps = math.ceil(K / tile_config.tile_k)
        # Matrix A elements are read grid_n times
        actual_bytes_a = (M * K * elem_bytes) * grid_n
        # Matrix B elements are read grid_m times
        actual_bytes_b = (K * N * elem_bytes) * grid_m
        # Matrix C written once
        actual_bytes_c = (M * N * elem_bytes)

        actual_bytes = actual_bytes_a + actual_bytes_b + actual_bytes_c
        reuse_factor = actual_bytes / ideal_bytes if ideal_bytes > 0 else 1.0

        total_flops = 2.0 * M * N * K
        effective_intensity = total_flops / actual_bytes if actual_bytes > 0 else 0.0

        return TileSimulationResult(
            hardware_name=self.hardware.name,
            tile_config=tile_config,
            gemm_m=M,
            gemm_n=N,
            gemm_k=K,
            a_tile_sram_kb=a_tile_bytes / 1024.0,
            b_tile_sram_kb=b_tile_bytes / 1024.0,
            c_tile_sram_kb=c_tile_bytes / 1024.0,
            total_sram_required_kb=total_sram_kb,
            hardware_sram_limit_kb=hw_sram_kb,
            sram_utilization_pct=sram_util_pct,
            sram_overflow=sram_overflow,
            total_grid_tiles=total_grid_tiles,
            active_blocks_per_sm=active_blocks_per_sm,
            total_waves=total_waves,
            tail_wave_penalty_pct=tail_wave_penalty,
            ideal_hbm_bytes=ideal_bytes,
            actual_hbm_bytes=actual_bytes,
            tile_reuse_factor=reuse_factor,
            effective_arithmetic_intensity=effective_intensity,
        )
