"""
TileLens: AI Accelerator Hardware-Software Co-Design & Roofline Memory Tile Visualizer
"""

__version__ = "0.1.0"

from tilelens.hardware.spec import HardwareSpec, Precision, HardwareType, MemoryHierarchy, ComputeCapacity
from tilelens.hardware.database import HARDWARE_DATABASE, get_hardware, list_available_hardware
from tilelens.core.roofline import RooflineModel, RooflineResult, calculate_gemm_intensity
from tilelens.core.tile_simulator import TileSimulator, TileConfig, TileSimulationResult
from tilelens.core.analyzer import PerformanceAnalyzer, OptimizationReport

__all__ = [
    "HardwareSpec",
    "Precision",
    "HardwareType",
    "MemoryHierarchy",
    "ComputeCapacity",
    "HARDWARE_DATABASE",
    "get_hardware",
    "list_available_hardware",
    "RooflineModel",
    "RooflineResult",
    "calculate_gemm_intensity",
    "TileSimulator",
    "TileConfig",
    "TileSimulationResult",
    "PerformanceAnalyzer",
    "OptimizationReport",
]
