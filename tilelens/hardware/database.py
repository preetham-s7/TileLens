"""
Database of verified AI accelerator hardware profiles (NVIDIA GPUs, Google TPUs, AMD, and Custom ASICs).
"""

from typing import Dict, List, Optional
from tilelens.hardware.spec import (
    HardwareSpec,
    HardwareType,
    MemoryHierarchy,
    ComputeCapacity,
    Precision,
)

HARDWARE_DATABASE: Dict[str, HardwareSpec] = {
    # --------------------------------------------------------------------------
    # NVIDIA GPUs
    # --------------------------------------------------------------------------
    "nvidia_h100_sxm": HardwareSpec(
        name="NVIDIA H100 SXM5",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Hopper (GH100)",
        process_node_nm=4.0,
        tdp_watts=700.0,
        description="NVIDIA Hopper Architecture Flagship Data Center GPU with 4th Gen Tensor Cores & TMA.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=3350.0,  # 3.35 TB/s HBM3
            hbm_capacity_gb=80.0,
            sram_per_core_kb=228.0,    # 228 KB Shared Memory per SM
            l2_cache_mb=50.0,
            num_compute_units=132,     # 132 SMs
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP64: 34.0,
                Precision.FP32: 67.0,
                Precision.TF32: 494.5,
                Precision.FP16: 989.0,
                Precision.BF16: 989.0,
                Precision.FP8: 1978.0,
                Precision.INT8: 1978.0,
            },
            sparse_tflops={
                Precision.TF32: 989.0,
                Precision.FP16: 1978.0,
                Precision.BF16: 1978.0,
                Precision.FP8: 3958.0,
                Precision.INT8: 3958.0,
            },
        ),
    ),

    "nvidia_a100_sxm": HardwareSpec(
        name="NVIDIA A100 SXM4",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Ampere (GA100)",
        process_node_nm=7.0,
        tdp_watts=400.0,
        description="NVIDIA Ampere Architecture Data Center GPU with 3rd Gen Tensor Cores.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=2039.0,  # 2.04 TB/s HBM2e
            hbm_capacity_gb=80.0,
            sram_per_core_kb=164.0,    # 164 KB Shared Memory per SM
            l2_cache_mb=40.0,
            num_compute_units=108,     # 108 SMs
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP64: 19.5,
                Precision.FP32: 19.5,
                Precision.TF32: 156.0,
                Precision.FP16: 312.0,
                Precision.BF16: 312.0,
                Precision.INT8: 624.0,
            },
            sparse_tflops={
                Precision.TF32: 312.0,
                Precision.FP16: 624.0,
                Precision.BF16: 624.0,
                Precision.INT8: 1248.0,
            },
        ),
    ),

    "nvidia_b200": HardwareSpec(
        name="NVIDIA Blackwell B200",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Blackwell (GB200/B200)",
        process_node_nm=4.0,
        tdp_watts=1000.0,
        description="NVIDIA Blackwell Dual-Die GPU with 2nd Gen Transformer Engine & 5th Gen Tensor Cores.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=8000.0,  # 8.0 TB/s HBM3e
            hbm_capacity_gb=192.0,
            sram_per_core_kb=256.0,
            l2_cache_mb=128.0,
            num_compute_units=160,
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP64: 45.0,
                Precision.FP32: 90.0,
                Precision.TF32: 1125.0,
                Precision.FP16: 2250.0,
                Precision.BF16: 2250.0,
                Precision.FP8: 4500.0,
                Precision.INT4: 9000.0,
            },
            sparse_tflops={
                Precision.FP16: 4500.0,
                Precision.BF16: 4500.0,
                Precision.FP8: 9000.0,
                Precision.INT4: 18000.0,
            },
        ),
    ),

    "nvidia_rtx_4090": HardwareSpec(
        name="NVIDIA GeForce RTX 4090",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Ada Lovelace (AD102)",
        process_node_nm=4.0,
        tdp_watts=450.0,
        description="Flagship Consumer GPU with 4th Gen Tensor Cores and GDDR6X memory.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=1008.0,  # 1.008 TB/s GDDR6X
            hbm_capacity_gb=24.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=72.0,
            num_compute_units=128,
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 82.6,
                Precision.FP16: 330.0,
                Precision.BF16: 330.0,
                Precision.FP8: 660.0,
                Precision.INT8: 660.0,
            },
            sparse_tflops={
                Precision.FP16: 660.0,
                Precision.BF16: 660.0,
                Precision.FP8: 1320.0,
                Precision.INT8: 1320.0,
            },
        ),
    ),

    # --------------------------------------------------------------------------
    # Google TPUs
    # --------------------------------------------------------------------------
    "google_tpu_v4": HardwareSpec(
        name="Google TPU v4",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v4 (Systolic Matrix Multiply Unit)",
        process_node_nm=7.0,
        tdp_watts=200.0,
        description="Google Tensor Processing Unit v4 with 128x128 3D Torus Interconnect & 2 Matrix Multiply Units (MXUs).",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=1200.0,  # 1.2 TB/s HBM2
            hbm_capacity_gb=32.0,
            sram_per_core_kb=16384.0,  # 16 MB Vector Memory (VMEM) / Scratchpad per TensorCore
            l2_cache_mb=0.0,           # Software-managed scratchpad instead of hardware cache
            num_compute_units=2,       # 2 Cores per TPU chip (each has 4 MXUs)
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 34.0,
                Precision.BF16: 275.0,
                Precision.INT8: 550.0,
            },
        ),
    ),

    "google_tpu_v5e": HardwareSpec(
        name="Google TPU v5e",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v5e (ViperLite)",
        process_node_nm=5.0,
        tdp_watts=150.0,
        description="Google TPU v5e designed for cost-efficient LLM training and high-throughput inference.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=819.0,   # 819 GB/s HBM2e
            hbm_capacity_gb=16.0,
            sram_per_core_kb=16384.0,  # 16 MB VMEM per Core
            l2_cache_mb=0.0,
            num_compute_units=1,       # 1 TensorCore per chip (with 4 MXUs)
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.BF16: 197.0,
                Precision.FP16: 197.0,
                Precision.INT8: 394.0,
            },
        ),
    ),

    "google_tpu_v5p": HardwareSpec(
        name="Google TPU v5p",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v5p (Performance Flagship)",
        process_node_nm=4.0,
        tdp_watts=300.0,
        description="Google TPU v5p with high-bandwidth ICI (Inter-Chip Interconnect) and 459 BF16 TFLOPs.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=1600.0,  # 1.6 TB/s HBM3
            hbm_capacity_gb=95.0,
            sram_per_core_kb=32768.0,  # 32 MB VMEM per Core
            l2_cache_mb=0.0,
            num_compute_units=2,       # 2 Cores per chip
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 57.0,
                Precision.BF16: 459.0,
                Precision.FP8: 918.0,
                Precision.INT8: 918.0,
            },
        ),
    ),

    # --------------------------------------------------------------------------
    # Custom Open-Source Silicon / TinyTapeout ASIC
    # --------------------------------------------------------------------------
    "custom_open_tpu_130nm": HardwareSpec(
        name="OpenTPU-130 (SkyWater 130nm)",
        vendor="Open Source",
        hardware_type=HardwareType.CUSTOM_ASIC,
        architecture="16x16 Weight-Stationary Systolic Array",
        process_node_nm=130.0,
        tdp_watts=0.5,
        description="Open-Source Silicon Matrix Multiply Accelerator targeted for SkyWater 130nm / TinyTapeout.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=0.8,     # 800 MB/s SPI/HyperRAM or QSPI off-chip
            hbm_capacity_gb=0.032,     # 32 MB HyperRAM
            sram_per_core_kb=64.0,     # 64 KB on-chip OpenRAM
            l2_cache_mb=0.0,
            num_compute_units=1,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.INT8: 0.0512,  # 51.2 GOPs at 100 MHz (16x16 MACs * 2 = 512 ops/cycle)
            },
        ),
    ),
}

# Aliases for fast user lookup
ALIAS_MAP: Dict[str, str] = {
    "h100": "nvidia_h100_sxm",
    "h100_sxm": "nvidia_h100_sxm",
    "a100": "nvidia_a100_sxm",
    "a100_sxm": "nvidia_a100_sxm",
    "b200": "nvidia_b200",
    "blackwell": "nvidia_b200",
    "rtx4090": "nvidia_rtx_4090",
    "4090": "nvidia_rtx_4090",
    "tpu_v4": "google_tpu_v4",
    "tpuv4": "google_tpu_v4",
    "tpu_v5e": "google_tpu_v5e",
    "tpuv5e": "google_tpu_v5e",
    "v5e": "google_tpu_v5e",
    "tpu_v5p": "google_tpu_v5p",
    "tpuv5p": "google_tpu_v5p",
    "v5p": "google_tpu_v5p",
    "opentpu": "custom_open_tpu_130nm",
}


def get_hardware(name_or_alias: str) -> HardwareSpec:
    """Retrieve a HardwareSpec by name or alias (case-insensitive)."""
    key = name_or_alias.lower().strip()
    resolved_key = ALIAS_MAP.get(key, key)
    if resolved_key in HARDWARE_DATABASE:
        return HARDWARE_DATABASE[resolved_key]
    
    # Fuzzy match by hardware name
    for k, spec in HARDWARE_DATABASE.items():
        if key in spec.name.lower() or key in k:
            return spec

    available = ", ".join(list(ALIAS_MAP.keys()))
    raise KeyError(f"Hardware '{name_or_alias}' not found. Available devices/aliases: {available}")


def list_available_hardware() -> List[HardwareSpec]:
    """Returns all available hardware profiles in the database."""
    return list(HARDWARE_DATABASE.values())
