"""
Hardware Specification and Data Models for GPUs, TPUs, and Custom AI Accelerators.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Precision(str, Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def byte_size(self) -> float:
        """Returns byte size per scalar element."""
        mapping = {
            Precision.FP64: 8.0,
            Precision.FP32: 4.0,
            Precision.TF32: 4.0,  # 19-bit format, stored as 4-byte container in memory
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.FP8: 1.0,
            Precision.INT8: 1.0,
            Precision.INT4: 0.5,
        }
        return mapping[self]

    @property
    def bits(self) -> int:
        return int(self.byte_size * 8)


class HardwareType(str, Enum):
    GPU = "GPU"
    TPU = "TPU"
    CUSTOM_ASIC = "CUSTOM_ASIC"
    CPU = "CPU"
    MOBILE_SOC = "MOBILE_SOC"
    LAPTOP_SOC = "LAPTOP_SOC"
    NPU = "NPU"


@dataclass
class BlueprintBlock:
    """An architectural silicon block in the chip blueprint."""
    name: str
    category: str        # compute, tensor, npu, cache, memory_ctrl, interconnect, media, io
    description: str
    count: int = 1
    area_percent: float = 0.0
    specs: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryHierarchy:
    """Represents the memory bandwidth and capacity hierarchy."""
    hbm_bandwidth_gbs: float          # Main off-chip memory bandwidth in GB/s (HBM3 / GDDR6X / LPDDR5X)
    hbm_capacity_gb: float            # Main off-chip memory capacity in GB
    sram_per_core_kb: float           # On-chip fast scratchpad / shared memory per SM or Core in KB
    l2_cache_mb: float                # Shared L2/L3 cache capacity in MB
    num_compute_units: int            # Number of Streaming Multiprocessors (SMs) or TPU Cores / MMUs
    register_file_kb_per_core: float = 0.0  # Register file per core in KB

    @property
    def total_sram_mb(self) -> float:
        """Total on-chip aggregate SRAM across all compute units."""
        return (self.sram_per_core_kb * self.num_compute_units) / 1024.0

    @property
    def hbm_bandwidth_tbs(self) -> float:
        """HBM bandwidth in TB/s."""
        return self.hbm_bandwidth_gbs / 1000.0


@dataclass
class ComputeCapacity:
    """Peak theoretical compute throughput across precision formats (in TFLOPs or TOPs)."""
    dense_tflops: Dict[Precision, float] = field(default_factory=dict)
    sparse_tflops: Dict[Precision, float] = field(default_factory=dict)

    def get_peak_tflops(self, precision: Precision, sparse: bool = False) -> float:
        """Returns peak TFLOPs (dense or sparse) for a given precision."""
        if sparse and precision in self.sparse_tflops:
            return self.sparse_tflops[precision]
        if precision in self.dense_tflops:
            return self.dense_tflops[precision]
        raise ValueError(f"Precision {precision} compute capacity not specified for this hardware.")


@dataclass
class HardwareSpec:
    """Comprehensive Hardware Specification for an AI Accelerator."""
    name: str
    vendor: str
    hardware_type: HardwareType
    architecture: str
    memory: MemoryHierarchy
    compute: ComputeCapacity
    tdp_watts: Optional[float] = None
    process_node_nm: Optional[float] = None
    description: str = ""
    # Silicon Blueprint & Microarchitecture Metadata
    transistor_count_billion: Optional[float] = None
    die_size_mm2: Optional[float] = None
    packaging: str = ""
    memory_type: str = ""
    memory_bus_width_bits: Optional[int] = None
    npu_tops: Optional[float] = None
    target_device: str = "Cloud / Data Center"
    blueprint_blocks: list = field(default_factory=list)
    capabilities: list = field(default_factory=list)

    def summary(self) -> str:
        bf16_peak = self.compute.dense_tflops.get(Precision.BF16, self.compute.dense_tflops.get(Precision.FP16, 0.0))
        fp8_peak = self.compute.dense_tflops.get(Precision.FP8, 0.0)
        npu_str = f" | NPU: {self.npu_tops:.1f} TOPS" if self.npu_tops else ""
        return (
            f"{self.name} ({self.vendor} {self.architecture})\n"
            f"  Target: {self.target_device} | Type: {self.hardware_type.value} | Process: {self.process_node_nm}nm | TDP: {self.tdp_watts}W\n"
            f"  Compute Units: {self.memory.num_compute_units} Cores/SMs{npu_str}\n"
            f"  Memory: {self.memory.hbm_bandwidth_gbs:,.0f} GB/s ({self.memory.hbm_capacity_gb} GB {self.memory_type or 'DRAM'})\n"
            f"  SRAM per Core: {self.memory.sram_per_core_kb:,.1f} KB (Total on-chip: {self.memory.total_sram_mb:.1f} MB)\n"
            f"  Peak Dense BF16/FP16: {bf16_peak:,.1f} TFLOPs | Peak FP8: {fp8_peak:,.1f} TFLOPs"
        )

    def blueprint_summary(self) -> str:
        """Returns detailed architectural blueprint breakdown and operational profile."""
        lines = [
            f"=== 🔍 CHIP BLUEPRINT & ARCHITECTURE: {self.name} ===",
            f"Vendor: {self.vendor} | Architecture: {self.architecture} | Class: {self.target_device}",
            f"Silicon Process: {self.process_node_nm}nm | Transistors: {self.transistor_count_billion or 'N/A'} Billion | Die Size: {self.die_size_mm2 or 'N/A'} mm² | TDP: {self.tdp_watts}W",
            f"Packaging: {self.packaging or 'Standard Monolithic'} | Memory Interface: {self.memory_type or 'DRAM'} ({self.memory_bus_width_bits or 'N/A'}-bit bus)",
            "",
            "--- Microarchitectural Blueprint Blocks ---"
        ]
        if self.blueprint_blocks:
            for b in self.blueprint_blocks:
                cnt_str = f"[{b.count}x] " if b.count > 1 else ""
                lines.append(f"  * {cnt_str}{b.name} ({b.category.upper()}): {b.description}")
                if b.specs:
                    spec_items = ", ".join([f"{k}: {v}" for k, v in b.specs.items()])
                    lines.append(f"      Specs: {spec_items}")
        else:
            lines.append(f"  * {self.memory.num_compute_units}x Compute Cores / SMs")
            lines.append(f"  * {self.memory.sram_per_core_kb} KB On-Chip SRAM Scratchpad per core")
            lines.append(f"  * {self.memory.l2_cache_mb} MB Shared L2 Cache")
            lines.append(f"  * {self.memory.hbm_bandwidth_gbs:,.0f} GB/s Memory Interface ({self.memory.hbm_capacity_gb} GB)")

        lines.extend([
            "",
            "--- What Else This Chip Does (Operational Capabilities) ---"
        ])
        if self.capabilities:
            for cap in self.capabilities:
                lines.append(f"  ✓ {cap}")
        else:
            lines.append("  ✓ General-purpose Matrix Multiplications (GEMM) & Tensor Acceleration")
            lines.append("  ✓ Transformer & LLM Attention Computation")
        
        return "\n".join(lines)
