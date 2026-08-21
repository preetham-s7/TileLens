"""
Core simulation and analysis engine for TileLens.
"""

from tilelens.core.roofline import (
    BoundType,
    RooflineModel,
    RooflineResult,
    calculate_gemm_intensity,
)
from tilelens.core.tile_simulator import (
    TileConfig,
    TileSimulationResult,
    TileSimulator,
)
from tilelens.core.analyzer import (
    OptimizationReport,
    OptimizationSuggestion,
    PerformanceAnalyzer,
)

__all__ = [
    "BoundType",
    "RooflineModel",
    "RooflineResult",
    "calculate_gemm_intensity",
    "TileConfig",
    "TileSimulationResult",
    "TileSimulator",
    "OptimizationReport",
    "OptimizationSuggestion",
    "PerformanceAnalyzer",
]
