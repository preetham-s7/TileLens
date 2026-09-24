"""
Database of verified AI accelerator hardware profiles (NVIDIA GPUs, Google TPUs, AMD, Mobile SoCs, Laptop APUs, and Custom ASICs).
Includes silicon microarchitecture blueprints, die specifications, and operational capability profiles.
"""

from typing import Dict, List, Optional
from tilelens.hardware.spec import (
    HardwareSpec,
    HardwareType,
    MemoryHierarchy,
    ComputeCapacity,
    Precision,
    BlueprintBlock,
)

HARDWARE_DATABASE: Dict[str, HardwareSpec] = {
    # --------------------------------------------------------------------------
    # MOBILE CHIPS (Smartphones & Handhelds)
    # --------------------------------------------------------------------------
    "apple_a17_pro": HardwareSpec(
        name="Apple A17 Pro",
        vendor="Apple",
        hardware_type=HardwareType.MOBILE_SOC,
        architecture="Apple A17 Pro (6-Core GPU + 16-Core Neural Engine)",
        target_device="Mobile Phone (iPhone 15 Pro / Max)",
        process_node_nm=3.0,
        transistor_count_billion=19.0,
        die_size_mm2=103.8,
        packaging="TSMC InFO-PoP (Integrated Fan-Out Package-on-Package)",
        memory_type="LPDDR5X-7500 Unified",
        memory_bus_width_bits=128,
        npu_tops=35.0,
        tdp_watts=10.0,
        description="First commercial 3nm smartphone chip with 6-core GPU featuring hardware Ray Tracing and a 35 TOPS Neural Engine.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=51.2,    # 51.2 GB/s LPDDR5X
            hbm_capacity_gb=8.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=24.0,          # 24 MB Unified System Level Cache (SLC)
            num_compute_units=6,       # 6 GPU Cores
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 2.15,
                Precision.FP16: 4.3,
                Precision.BF16: 4.3,
                Precision.INT8: 35.0,  # 35 TOPS on Neural Engine
                Precision.INT4: 70.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Apple 6-Core GPU", "compute", "6 Shader Cores with dedicated Hardware Ray Tracing and Mesh Shading pipelines", count=6, area_percent=26.0, specs={"FP32 Peak": "2.15 TFLOPs", "RT Units": "6 RT Accelerators"}),
            BlueprintBlock("Apple 16-Core Neural Engine (NPU)", "npu", "Dedicated AI accelerator optimized for INT8/FP16 matrix ops and transformer layers", count=16, area_percent=14.0, specs={"Throughput": "35 TOPS INT8", "Target": "Apple Intelligence"}),
            BlueprintBlock("2x 'Everest' CPU Performance Cores", "compute", "Ultrawide out-of-order execution cores with 64KB L1 and 16MB L2 cache", count=2, area_percent=18.0, specs={"Clock": "3.78 GHz", "Cache": "16 MB L2"}),
            BlueprintBlock("4x 'Sawtooth' CPU Efficiency Cores", "compute", "Energy-efficient cores running at 2.11 GHz with 4MB shared L2 cache", count=4, area_percent=9.0, specs={"Clock": "2.11 GHz", "Cache": "4 MB L2"}),
            BlueprintBlock("System Level Cache (SLC)", "cache", "Ultra-fast on-die SRAM buffer shared across CPU, GPU, and NPU to minimize DRAM traffic", count=1, area_percent=12.0, specs={"Capacity": "24 MB", "Bandwidth": "400+ GB/s"}),
            BlueprintBlock("128-bit LPDDR5X Memory Controller", "memory_ctrl", "Dual 64-bit channels interfacing on-package LPDDR5X RAM", count=2, area_percent=9.0, specs={"Bandwidth": "51.2 GB/s", "Bus Width": "128-bit"}),
            BlueprintBlock("Media & Display Engine", "media", "Hardware ProRes, AV1 video decode, and ProMotion 120Hz display controller", count=1, area_percent=8.0, specs={"Video": "4K60 ProRes HDR, AV1", "Display": "120Hz ProMotion"}),
            BlueprintBlock("USB 3.2 Gen 2 / DisplayPort PHY", "io", "10 Gbps high-speed interface for data transfer and external 4K video out", count=1, area_percent=4.0, specs={"Speed": "10 Gbps", "Protocol": "USB-C 3.2"}),
        ],
        capabilities=[
            "Apple Intelligence on-device execution (local summarization, semantic indexing, image creation)",
            "Runs quantized 3B-8B LLMs (Llama-3 8B 4-bit, Gemma-2 2B) locally at 15-22 tokens/sec",
            "Hardware-accelerated console gaming with Ray Tracing (Resident Evil 4, Death Stranding)",
            "Real-time Photonic Engine computational photography & 4K60 ProRes HDR recording",
            "Hardware AV1 video decoding with near-zero battery drain",
        ],
    ),

    "snapdragon_8_gen3": HardwareSpec(
        name="Qualcomm Snapdragon 8 Gen 3",
        vendor="Qualcomm",
        hardware_type=HardwareType.MOBILE_SOC,
        architecture="Qualcomm Kryo + Adreno 750 + Hexagon NPU",
        target_device="Flagship Android Phone (Galaxy S24, OnePlus 12)",
        process_node_nm=4.0,
        transistor_count_billion=15.0,
        die_size_mm2=108.0,
        packaging="TSMC 4N InFO PoP",
        memory_type="LPDDR5X-9600",
        memory_bus_width_bits=64,
        npu_tops=45.0,
        tdp_watts=12.0,
        description="Flagship Android SoC with Adreno 750 GPU and Hexagon NPU supporting up to 10B parameter LLMs directly on-device.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=77.0,    # 77 GB/s LPDDR5X-9600
            hbm_capacity_gb=16.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=12.0,          # 12 MB total cache (CPU L3 + SLC)
            num_compute_units=6,
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 3.4,
                Precision.FP16: 6.8,
                Precision.BF16: 6.8,
                Precision.INT8: 45.0,  # 45 TOPS on Hexagon NPU
                Precision.INT4: 90.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Adreno 750 GPU Subsystem", "compute", "High-performance shader array with Global Illumination Ray Tracing", count=1, area_percent=27.0, specs={"FP32": "3.4 TFLOPs", "Features": "Adreno Ray Tracing Engine"}),
            BlueprintBlock("Hexagon Tensor NPU", "npu", "Fused scalar, vector, and tensor micro-tile accelerators for Generative AI", count=1, area_percent=18.0, specs={"NPU TOPS": "45 TOPS INT8", "Speed": "<1s Stable Diffusion"}),
            BlueprintBlock("Cortex-X4 Prime Core (3.3 GHz)", "compute", "Armv9.2 ultra-performance core for burst computing and gaming", count=1, area_percent=14.0, specs={"Clock": "3.3 GHz", "L2": "2 MB"}),
            BlueprintBlock("5x Cortex-A720 Performance Cores", "compute", "Heavy multitasking and compute cluster (3.2 GHz and 3.0 GHz)", count=5, area_percent=16.0, specs={"Clock": "3.0-3.2 GHz", "L2": "5x 512 KB"}),
            BlueprintBlock("2x Cortex-A520 Efficiency Cores", "compute", "Low power background cores running at 2.3 GHz", count=2, area_percent=5.0, specs={"Clock": "2.3 GHz", "L2": "Shared 1 MB"}),
            BlueprintBlock("12 MB Qualcomm System Cache", "cache", "Interconnected SRAM cache buffering memory transactions", count=1, area_percent=8.0, specs={"Capacity": "12 MB"}),
            BlueprintBlock("Quad-Channel LPDDR5X Controller", "memory_ctrl", "4x 16-bit high-speed channels reaching 77 GB/s bandwidth", count=4, area_percent=7.0, specs={"Speed": "9600 MT/s", "Bandwidth": "77 GB/s"}),
            BlueprintBlock("Spectra 18-bit Triple ISP", "media", "Cognitive ISP executing real-time semantic segmentation on 200MP sensors", count=1, area_percent=5.0, specs={"Throughput": "3.2 Gigapixels/sec"}),
        ],
        capabilities=[
            "Generative AI on-device: Generates Stable Diffusion 1.5 images in ~0.6 seconds",
            "Runs Llama-3 8B INT4 quantized LLM at 18-20 tokens/sec",
            "Unreal Engine 5 Lumen & Ray Tracing with hardware global illumination at 60 FPS",
            "8K30 HDR video capture with simultaneous 64MP photo capture",
            "Snapdragon X75 5G Modem-RF subsystem with 10 Gbps peak download",
        ],
    ),

    "google_tensor_g4": HardwareSpec(
        name="Google Tensor G4",
        vendor="Google",
        hardware_type=HardwareType.MOBILE_SOC,
        architecture="Google Tensor G4 (Mali-G715 + EdgeTPU v3)",
        target_device="Google Pixel 9 / 9 Pro",
        process_node_nm=4.0,
        transistor_count_billion=14.0,
        die_size_mm2=110.0,
        packaging="Samsung 4LPP+ FOPLP",
        memory_type="LPDDR5X",
        memory_bus_width_bits=64,
        npu_tops=37.0,
        tdp_watts=10.0,
        description="Google custom silicon designed in collaboration with Google DeepMind specifically to run Gemini Nano with multimodal inputs.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=68.0,
            hbm_capacity_gb=16.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=8.0,
            num_compute_units=7,
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 1.8,
                Precision.FP16: 3.6,
                Precision.BF16: 3.6,
                Precision.INT8: 37.0,
                Precision.INT4: 74.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Google EdgeTPU Gen 3", "npu", "Custom systolic matrix multiply unit tailored for Gemini Nano multimodality", count=1, area_percent=22.0, specs={"TOPS": "37 TOPS INT8", "Target": "Gemini Nano"}),
            BlueprintBlock("Arm Mali-G715 GPU", "compute", "7-core graphics and compute engine with variable rate shading", count=7, area_percent=24.0, specs={"FP32": "1.8 TFLOPs", "Cores": "7"}),
            BlueprintBlock("Tri-Cluster CPU (1+3+4)", "compute", "1x Cortex-X4 (3.1 GHz), 3x Cortex-A720 (2.6 GHz), 4x Cortex-A520 (1.9 GHz)", count=8, area_percent=26.0, specs={"P-Core": "Cortex-X4", "L3": "8 MB"}),
            BlueprintBlock("System Level Cache (SLC)", "cache", "8 MB shared on-die cache for EdgeTPU and GPU", count=1, area_percent=10.0, specs={"Capacity": "8 MB"}),
            BlueprintBlock("LPDDR5X Memory PHY", "memory_ctrl", "Dual-channel high-speed memory controller", count=2, area_percent=8.0, specs={"Bandwidth": "68 GB/s"}),
            BlueprintBlock("Google ISP & Titan M2", "media", "Computational photography and hardware cryptographic security enclave", count=1, area_percent=10.0, specs={"Security": "Titan M2", "AI ISP": "4K60 HDR"}),
        ],
        capabilities=[
            "Native on-device execution of Google Gemini Nano with Multimodality (text, vision, audio)",
            "Magic Editor, Best Take, Audio Magic Eraser, and Zoom Enhance",
            "Live Whisper-derived real-time call screening and live speech translation",
            "Titan M2 hardware security enclave for tamper-resistant identity protection",
        ],
    ),

    "mediatek_dimensity_9300": HardwareSpec(
        name="MediaTek Dimensity 9300",
        vendor="MediaTek",
        hardware_type=HardwareType.MOBILE_SOC,
        architecture="MediaTek All-Big-Core (Immortalis-G720 + APU 790)",
        target_device="Flagship Mobile Phone",
        process_node_nm=4.0,
        transistor_count_billion=16.0,
        die_size_mm2=112.0,
        packaging="TSMC 4N InFO",
        memory_type="LPDDR5T-9600",
        memory_bus_width_bits=64,
        npu_tops=46.0,
        tdp_watts=12.5,
        description="All-Big-Core architecture smartphone processor featuring 12-core Immortalis GPU and hardware LoRA generative AI APU 790.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=96.0,    # 96 GB/s LPDDR5T
            hbm_capacity_gb=16.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=10.0,
            num_compute_units=12,
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 4.6,
                Precision.FP16: 9.2,
                Precision.BF16: 9.2,
                Precision.INT8: 46.0,
                Precision.INT4: 92.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Immortalis-G720 MC12 GPU", "compute", "12-core raytracing GPU with Deferred Vertex Shading (DVS)", count=12, area_percent=30.0, specs={"FP32": "4.6 TFLOPs", "Ray Tracing": "Hardware Ray Tracing Unit"}),
            BlueprintBlock("MediaTek APU 790 NPU", "npu", "Generative AI Transformer Engine with hardware LoRA fusion acceleration", count=1, area_percent=20.0, specs={"NPU TOPS": "46 TOPS", "LLM": "Up to 33B parameters"}),
            BlueprintBlock("All-Big-Core CPU Array", "compute", "4x Cortex-X4 (up to 3.25 GHz) + 4x Cortex-A720 (2.0 GHz) with no small cores", count=8, area_percent=26.0, specs={"Big Cores": "8 Big Cores", "L3": "8 MB"}),
            BlueprintBlock("10 MB System-Level Cache", "cache", "Ultra-wide low-latency cache buffer", count=1, area_percent=9.0, specs={"Capacity": "10 MB"}),
            BlueprintBlock("LPDDR5T Memory Controller", "memory_ctrl", "Ultra-fast 9600 Mbps memory interface reaching 96 GB/s", count=4, area_percent=8.0, specs={"Bandwidth": "96 GB/s"}),
            BlueprintBlock("Imagiq 990 ISP", "media", "Standalone 18-bit RAW ISP supporting 320MP camera sensors", count=1, area_percent=7.0, specs={"Resolution": "320 MP", "Video": "8K30 HDR"}),
        ],
        capabilities=[
            "First mobile SoC supporting on-device LoRA (Low-Rank Adaptation) fine-tuning fusion",
            "Accelerates 7B/13B parameter LLMs at up to 22 tokens/sec",
            "Hardware console-grade ray tracing at 60 FPS with global illumination",
            "Sub-1-second generative image synthesis using Stable Diffusion",
        ],
    ),

    # --------------------------------------------------------------------------
    # LAPTOP & CLIENT PC CHIPS
    # --------------------------------------------------------------------------
    "apple_m4": HardwareSpec(
        name="Apple M4",
        vendor="Apple",
        hardware_type=HardwareType.LAPTOP_SOC,
        architecture="Apple M4 (10-Core GPU + 16-Core Neural Engine)",
        target_device="Laptop & Tablet PC (MacBook / iPad Pro)",
        process_node_nm=3.0,
        transistor_count_billion=28.0,
        die_size_mm2=165.0,
        packaging="TSMC 3nm InFO",
        memory_type="Unified LPDDR5X-7500",
        memory_bus_width_bits=128,
        npu_tops=38.0,
        tdp_watts=22.0,
        description="Apple 2nd-gen 3nm Apple Silicon chip with 10-core GPU featuring Dynamic Caching and a 38 TOPS Neural Engine.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=120.0,   # 120 GB/s Unified Memory
            hbm_capacity_gb=32.0,
            sram_per_core_kb=256.0,
            l2_cache_mb=32.0,          # 32 MB Unified SLC
            num_compute_units=10,      # 10 GPU cores
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 4.3,
                Precision.FP16: 8.6,
                Precision.BF16: 8.6,
                Precision.INT8: 38.0,  # 38 TOPS Neural Engine
                Precision.INT4: 76.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Apple 10-Core GPU", "compute", "10-core GPU with Dynamic Caching, Hardware Ray Tracing, and Mesh Shading", count=10, area_percent=32.0, specs={"FP32": "4.3 TFLOPs", "Tech": "Dynamic Caching"}),
            BlueprintBlock("16-Core Neural Engine (ANE)", "npu", "38 TOPS dedicated AI processor for Apple Intelligence and local transformers", count=16, area_percent=15.0, specs={"TOPS": "38 TOPS INT8"}),
            BlueprintBlock("4x Performance CPU Cores", "compute", "Next-generation P-cores with 16MB shared L2 cache and ultra-wide decode", count=4, area_percent=18.0, specs={"Clock": "4.4 GHz", "L2": "16 MB"}),
            BlueprintBlock("6x Efficiency CPU Cores", "compute", "Enhanced E-cores with 6MB shared L2 cache for sustained efficiency", count=6, area_percent=10.0, specs={"Clock": "2.8 GHz", "L2": "6 MB"}),
            BlueprintBlock("32 MB System Level Cache (SLC)", "cache", "High-speed shared cache linking CPU, GPU, and NPU directly", count=1, area_percent=11.0, specs={"Capacity": "32 MB", "Bandwidth": "800+ GB/s"}),
            BlueprintBlock("128-bit Unified Memory Controller", "memory_ctrl", "Zero-copy unified memory fabric running at 120 GB/s", count=2, area_percent=8.0, specs={"Bandwidth": "120 GB/s", "Bus": "128-bit"}),
            BlueprintBlock("Next-Gen Media Engine", "media", "Hardware decode/encode for 8K H.264, HEVC, ProRes, and AV1 video", count=1, area_percent=6.0, specs={"Codecs": "AV1, ProRes RAW, HEVC"}),
        ],
        capabilities=[
            "Zero-copy Unified Memory: CPU, GPU, and NPU share the same 32GB pool without PCI transfers",
            "Runs Llama-3 8B (Q4_K_M) locally at 28-32 tokens/sec via MLX or llama.cpp",
            "Hardware Ray Tracing in Blender, Octane X, and Metal-native games",
            "Multistream 8K ProRes 422 video editing in Final Cut Pro",
            "Apple Intelligence on-device privacy-first AI workflows",
        ],
    ),

    "apple_m3_max": HardwareSpec(
        name="Apple M3 Max",
        vendor="Apple",
        hardware_type=HardwareType.LAPTOP_SOC,
        architecture="Apple M3 Max (40-Core GPU + 16-Core NPU)",
        target_device="High-End Workstation Laptop (MacBook Pro 16)",
        process_node_nm=3.0,
        transistor_count_billion=92.0,
        die_size_mm2=470.0,
        packaging="TSMC 3nm InFO",
        memory_type="Unified LPDDR5X",
        memory_bus_width_bits=512,
        npu_tops=18.0,
        tdp_watts=80.0,
        description="Extreme workstation laptop chip with 92 Billion transistors, 40-core GPU, and up to 128 GB Unified Memory at 400 GB/s.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=400.0,   # 400 GB/s Unified Memory
            hbm_capacity_gb=128.0,
            sram_per_core_kb=256.0,
            l2_cache_mb=64.0,
            num_compute_units=40,
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 16.2,
                Precision.FP16: 32.4,
                Precision.BF16: 32.4,
                Precision.INT8: 64.8,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Apple 40-Core GPU", "compute", "40 Shader cores with Dynamic Caching & Hardware Ray Tracing", count=40, area_percent=45.0, specs={"FP32": "16.2 TFLOPs", "FP16": "32.4 TFLOPs"}),
            BlueprintBlock("16-Core Neural Engine", "npu", "Matrix coprocessor for ML inference", count=16, area_percent=8.0, specs={"Throughput": "18 TOPS"}),
            BlueprintBlock("12x Performance CPU Cores", "compute", "3 clusters of 4 P-cores with 48MB total L2 cache", count=12, area_percent=18.0, specs={"Clock": "4.05 GHz", "L2": "48 MB"}),
            BlueprintBlock("4x Efficiency CPU Cores", "compute", "Background task efficiency cluster", count=4, area_percent=5.0, specs={"Clock": "2.75 GHz"}),
            BlueprintBlock("64 MB Unified System Cache", "cache", "Interconnected SRAM on-die buffer", count=1, area_percent=10.0, specs={"Capacity": "64 MB"}),
            BlueprintBlock("512-bit Unified Memory Subsystem", "memory_ctrl", "Massive memory bus delivering 400 GB/s across 128GB unified RAM", count=8, area_percent=14.0, specs={"Bandwidth": "400 GB/s", "Bus": "512-bit"}),
        ],
        capabilities=[
            "Can host and run Llama-3 70B (4-bit quantized) entirely in local memory at 12-14 tokens/sec",
            "128 GB unified VRAM allows loading multi-billion parameter models impossible on consumer GPUs",
            "Real-time 3D rendering with hardware raytracing and complex particle simulations",
            "Compiles massive codebases in parallel across 16 CPU cores",
        ],
    ),

    "intel_lunar_lake_288v": HardwareSpec(
        name="Intel Core Ultra 7 288V",
        vendor="Intel",
        hardware_type=HardwareType.LAPTOP_SOC,
        architecture="Intel Lunar Lake (Xe2 Battlemage + NPU 4.0)",
        target_device="Copilot+ Thin & Light Laptop",
        process_node_nm=3.0,
        transistor_count_billion=18.0,
        die_size_mm2=140.0,
        packaging="Intel Foveros 3D Multi-Tile Package with On-Package Memory",
        memory_type="On-Package LPDDR5X-8533",
        memory_bus_width_bits=128,
        npu_tops=48.0,
        tdp_watts=25.0,
        description="Intel Lunar Lake flagship processor with 48 TOPS NPU, Battlemage Xe2 GPU, and on-package memory delivering over 20 hours battery life.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=137.0,   # 137 GB/s on-package LPDDR5X-8533
            hbm_capacity_gb=32.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=16.0,          # 16 MB Memory-Side Cache
            num_compute_units=8,       # 8 Xe2-cores
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 8.4,
                Precision.FP16: 16.8,
                Precision.BF16: 16.8,
                Precision.INT8: 67.0,  # 67 INT8 TOPs on Xe2 + 48 TOPS on NPU
                Precision.INT4: 134.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("Intel NPU 4.0", "npu", "48 TOPS dedicated neural processing unit with 6 neural compute engines", count=6, area_percent=22.0, specs={"TOPS": "48 TOPS INT8", "Target": "Copilot+ PC"}),
            BlueprintBlock("Intel Xe2 Battlemage GPU", "compute", "8 2nd-gen Xe2-cores with 64 vector engines and 8 ray tracing units", count=8, area_percent=34.0, specs={"Peak": "67 TOPS INT8 / 8.4 TF FP32", "RT Units": "8"}),
            BlueprintBlock("4x 'Lion Cove' Performance Cores", "compute", "High IPC P-cores without Hyper-Threading for maximum efficiency", count=4, area_percent=18.0, specs={"Clock": "5.0 GHz", "L2": "10 MB"}),
            BlueprintBlock("4x 'Skymont' Low-Power E-Cores", "compute", "Sub-cluster handling everyday productivity and background tasks", count=4, area_percent=8.0, specs={"Clock": "3.7 GHz", "L2": "4 MB"}),
            BlueprintBlock("16 MB Memory-Side Cache", "cache", "System-level cache filtering memory requests before DRAM", count=1, area_percent=8.0, specs={"Capacity": "16 MB"}),
            BlueprintBlock("On-Package LPDDR5X Memory", "memory_ctrl", "Dual-channel on-package RAM delivering 137 GB/s", count=2, area_percent=10.0, specs={"Bandwidth": "137 GB/s", "Speed": "8533 MT/s"}),
        ],
        capabilities=[
            "Microsoft Copilot+ PC certified (exceeds the 40 TOPS NPU requirement with 48 TOPS)",
            "Runs local LLMs (Llama-3 8B 4-bit) at 24-26 tokens/sec via OpenVINO",
            "On-package memory reduces trace capacitance, enabling 20+ hours of real-world battery life",
            "Hardware-accelerated XeSS super-resolution and real-time Ray Tracing",
        ],
    ),

    "amd_ryzen_ai_9_hx370": HardwareSpec(
        name="AMD Ryzen AI 9 HX 370",
        vendor="AMD",
        hardware_type=HardwareType.LAPTOP_SOC,
        architecture="AMD Strix Point (Zen 5 + RDNA 3.5 + XDNA 2 NPU)",
        target_device="High-Performance AI Laptop PC",
        process_node_nm=4.0,
        transistor_count_billion=17.5,
        die_size_mm2=232.0,
        packaging="TSMC 4N Monolithic",
        memory_type="LPDDR5X-7500",
        memory_bus_width_bits=128,
        npu_tops=50.0,
        tdp_watts=35.0,
        description="AMD Strix Point processor with 12 Zen 5 CPU cores, 16 RDNA 3.5 CUs, and a 50 TOPS XDNA 2 NPU with Block FP16 support.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=120.0,
            hbm_capacity_gb=32.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=24.0,          # 24 MB L3 Cache
            num_compute_units=16,      # 16 RDNA 3.5 CUs
            register_file_kb_per_core=128.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 7.8,
                Precision.FP16: 15.6,
                Precision.BF16: 15.6,
                Precision.INT8: 50.0,  # 50 TOPS on XDNA 2
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("AMD XDNA 2 NPU", "npu", "50 TOPS spatial array NPU with native Block FP16 mathematical precision support", count=1, area_percent=20.0, specs={"TOPS": "50 TOPS", "Precision": "Block FP16"}),
            BlueprintBlock("Radeon 890M GPU (16 CUs)", "compute", "16 RDNA 3.5 Compute Units (1024 stream processors)", count=16, area_percent=32.0, specs={"FP32": "7.8 TFLOPs", "Clock": "2.9 GHz"}),
            BlueprintBlock("4x Zen 5 Performance Cores", "compute", "Full-width Zen 5 cores with 16MB L3 cache", count=4, area_percent=18.0, specs={"Clock": "5.1 GHz", "L3": "16 MB"}),
            BlueprintBlock("8x Zen 5c Density-Optimized Cores", "compute", "Compact Zen 5c cores with 8MB L3 cache", count=8, area_percent=15.0, specs={"Clock": "3.3 GHz", "L3": "8 MB"}),
            BlueprintBlock("Dual-Channel LPDDR5X Controller", "memory_ctrl", "128-bit memory bus operating at 7500 MT/s", count=2, area_percent=10.0, specs={"Bandwidth": "120 GB/s"}),
            BlueprintBlock("AMD Media & Display Core Next", "media", "AV1 hardware decode/encode with FreeSync and multi-monitor output", count=1, area_percent=5.0, specs={"Display": "4x 4K120 or 2x 8K"}),
        ],
        capabilities=[
            "Industry-first Block FP16 support on NPU: 16-bit accuracy at 8-bit computational efficiency",
            "Copilot+ PC certified with total platform AI compute exceeding 80 TOPS",
            "Smooth 1080p eSports and AAA gaming on integrated Radeon 890M graphics",
            "Seamless acceleration across ONNX Runtime, DirectML, and ROCm",
        ],
    ),

    "nvidia_rtx_4060_mobile": HardwareSpec(
        name="NVIDIA GeForce RTX 4060 Laptop",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Ada Lovelace (AD107)",
        target_device="Gaming & AI Creator Laptop",
        process_node_nm=4.0,
        transistor_count_billion=18.9,
        die_size_mm2=156.0,
        packaging="TSMC 4N Monolithic",
        memory_type="GDDR6",
        memory_bus_width_bits=128,
        tdp_watts=115.0,
        description="Mainstream Ada Lovelace laptop GPU featuring 24 SMs, 96 4th-Gen Tensor Cores, and DLSS 3 Frame Generation.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=256.0,   # 256 GB/s GDDR6
            hbm_capacity_gb=8.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=32.0,          # 32 MB L2 Cache
            num_compute_units=24,      # 24 SMs (3072 CUDA cores)
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 15.1,
                Precision.FP16: 60.5,
                Precision.BF16: 60.5,
                Precision.FP8: 121.0,
                Precision.INT8: 121.0,
            },
            sparse_tflops={
                Precision.FP16: 121.0,
                Precision.BF16: 121.0,
                Precision.FP8: 242.0,
                Precision.INT8: 242.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("24x Ada Lovelace SMs", "compute", "Streaming Multiprocessors with 3072 CUDA Cores and 24 RT Cores", count=24, area_percent=52.0, specs={"FP32": "15.1 TFLOPs", "CUDA Cores": "3072"}),
            BlueprintBlock("96x 4th-Gen Tensor Cores", "tensor", "FP8/FP16 matrix accelerators with FP8 Transformer Engine support", count=96, area_percent=16.0, specs={"FP8": "121 TFLOPs", "FP16": "60.5 TFLOPs"}),
            BlueprintBlock("32 MB High-Speed L2 Cache", "cache", "16x larger L2 cache than predecessor, drastically reducing GDDR6 bus traffic", count=1, area_percent=14.0, specs={"Capacity": "32 MB"}),
            BlueprintBlock("128-bit GDDR6 Memory Controller", "memory_ctrl", "Dual 64-bit channels running at 16 Gbps", count=2, area_percent=10.0, specs={"Bandwidth": "256 GB/s", "VRAM": "8 GB"}),
            BlueprintBlock("8th-Gen NVENC with AV1", "media", "Hardware AV1 encoder delivering 40% higher efficiency than H.264", count=1, area_percent=5.0, specs={"Codecs": "AV1, HEVC, H.264"}),
            BlueprintBlock("PCIe 4.0 x8 Host Interface", "io", "High-speed connection to laptop host CPU", count=1, area_percent=3.0, specs={"Speed": "16 GB/s bidirectional"}),
        ],
        capabilities=[
            "PyTorch and CUDA native development on laptop (Triton, TensorRT, vLLM)",
            "Runs Llama-3 8B 4-bit at 55-60 tokens/sec via TensorRT-LLM",
            "DLSS 3 AI Frame Generation for high FPS raytraced gaming",
            "Stable Diffusion 1.5/XL fast image generation with xFormers and TensorRT",
        ],
    ),

    # --------------------------------------------------------------------------
    # DESKTOP & DATA CENTER SILICON
    # --------------------------------------------------------------------------
    "nvidia_h100_sxm": HardwareSpec(
        name="NVIDIA H100 SXM5",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Hopper (GH100)",
        target_device="Cloud & AI Supercomputer",
        process_node_nm=4.0,
        transistor_count_billion=80.0,
        die_size_mm2=814.0,
        packaging="TSMC CoWoS-S 2.5D",
        memory_type="HBM3",
        memory_bus_width_bits=5120,
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
        blueprint_blocks=[
            BlueprintBlock("132x Hopper SMs", "compute", "Streaming Multiprocessors with DPX instruction set and asynchronous TMA engines", count=132, area_percent=55.0, specs={"SM Count": "132", "SRAM/SM": "228 KB"}),
            BlueprintBlock("528x 4th-Gen Tensor Cores", "tensor", "Transformer Engine tensor cores with FP8 FP16 BF16 mixed-precision", count=528, area_percent=18.0, specs={"BF16 Peak": "989 TFLOPs", "FP8 Peak": "1,978 TFLOPs"}),
            BlueprintBlock("50 MB High-Bandwidth L2 Cache", "cache", "Unified cache with asynchronous crossbar routing", count=1, area_percent=8.0, specs={"Capacity": "50 MB", "Bandwidth": "12 TB/s"}),
            BlueprintBlock("5x HBM3 Memory Stacks (CoWoS)", "memory_ctrl", "5120-bit silicon interposer memory interface", count=5, area_percent=12.0, specs={"Bandwidth": "3,350 GB/s (3.35 TB/s)", "Capacity": "80 GB"}),
            BlueprintBlock("4th-Gen NVLink Engine", "interconnect", "900 GB/s bidirectional multi-GPU clustering fabric", count=18, area_percent=7.0, specs={"Bandwidth": "900 GB/s"}),
        ],
        capabilities=[
            "Large Language Model Training: GPT-4, Llama-3 70B/405B pretraining & fine-tuning",
            "Transformer Engine dynamically switches between FP8 and FP16 for 2-3x speedup",
            "Asynchronous TMA (Tensor Memory Accelerator) bypasses register files directly to Shared Memory",
            "High-throughput vLLM & TensorRT-LLM datacenter inference serving",
        ],
    ),

    "nvidia_b200": HardwareSpec(
        name="NVIDIA Blackwell B200",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Blackwell (GB200/B200)",
        target_device="Hyperscale AI Supercluster",
        process_node_nm=4.0,
        transistor_count_billion=208.0,
        die_size_mm2=1628.0,       # Dual-die (2x 814mm²)
        packaging="TSMC CoWoS-L (10 TB/s NV-HBI Inter-Die Link)",
        memory_type="HBM3e",
        memory_bus_width_bits=8192,
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
        blueprint_blocks=[
            BlueprintBlock("Dual-Die Silicon Array (160 SMs)", "compute", "Twin dies linked seamlessly via 10 TB/s NV-HBI interconnect", count=160, area_percent=54.0, specs={"SMs": "160", "Transistors": "208 Billion"}),
            BlueprintBlock("5th-Gen Tensor Cores (FP4 Engine)", "tensor", "2nd-Gen Transformer Engine with hardware native FP4 and FP8 precision", count=640, area_percent=20.0, specs={"FP8": "4,500 TFLOPs", "FP4": "9,000 TFLOPs"}),
            BlueprintBlock("128 MB Shared L2 Cache", "cache", "Massive on-die cache keeping active tiles on-chip", count=1, area_percent=8.0, specs={"Capacity": "128 MB"}),
            BlueprintBlock("8x HBM3e Memory Stacks", "memory_ctrl", "World-leading 8.0 TB/s memory bandwidth with 192 GB capacity", count=8, area_percent=12.0, specs={"Bandwidth": "8,000 GB/s (8.0 TB/s)", "VRAM": "192 GB"}),
            BlueprintBlock("5th-Gen NVLink Interconnect", "interconnect", "1.8 TB/s bidirectional interconnect for NVL72 rack-scale systems", count=18, area_percent=6.0, specs={"Bandwidth": "1,800 GB/s"}),
        ],
        capabilities=[
            "Serves trillion-parameter MoE (Mixture of Experts) models in real time",
            "Native 4-bit Floating Point (FP4) quantization doubles inference throughput vs FP8",
            "NVLink 72 allows 72 Blackwell GPUs to act as a single 130 TB/s unified GPU",
            "Runs 405B parameter models with unprecedented tokens/second per Watt",
        ],
    ),

    "nvidia_a100_sxm": HardwareSpec(
        name="NVIDIA A100 SXM4",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Ampere (GA100)",
        target_device="Datacenter & Cloud",
        process_node_nm=7.0,
        transistor_count_billion=54.2,
        die_size_mm2=826.0,
        packaging="TSMC CoWoS 2.5D",
        memory_type="HBM2e",
        memory_bus_width_bits=5120,
        tdp_watts=400.0,
        description="NVIDIA Ampere Architecture Data Center GPU with 3rd Gen Tensor Cores and TF32 precision.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=2039.0,  # 2.04 TB/s HBM2e
            hbm_capacity_gb=80.0,
            sram_per_core_kb=164.0,
            l2_cache_mb=40.0,
            num_compute_units=108,
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
        blueprint_blocks=[
            BlueprintBlock("108x Ampere SMs", "compute", "Streaming Multiprocessors with 3rd-Gen Tensor Cores and TF32 support", count=108, area_percent=58.0, specs={"SMs": "108", "SRAM/SM": "164 KB"}),
            BlueprintBlock("432x 3rd-Gen Tensor Cores", "tensor", "Hardware acceleration for BF16, FP16, and INT8 matrix multiplies", count=432, area_percent=18.0, specs={"BF16": "312 TFLOPs"}),
            BlueprintBlock("40 MB L2 Cache", "cache", "Crossbar routed L2 cache", count=1, area_percent=8.0, specs={"Capacity": "40 MB"}),
            BlueprintBlock("5x HBM2e Memory Stacks", "memory_ctrl", "2.04 TB/s HBM2e memory interface", count=5, area_percent=11.0, specs={"Bandwidth": "2,039 GB/s", "Capacity": "80 GB"}),
            BlueprintBlock("3rd-Gen NVLink", "interconnect", "600 GB/s multi-GPU interconnect", count=12, area_percent=5.0, specs={"Bandwidth": "600 GB/s"}),
        ],
        capabilities=[
            "Industry workhorse for LLM training and fine-tuning (Megatron-LM, DeepSpeed)",
            "Multi-Instance GPU (MIG) partitions one A100 into up to 7 isolated GPU instances",
            "High-throughput scientific computing and matrix decomposition (FP64 & TF32)",
        ],
    ),

    "nvidia_rtx_4090": HardwareSpec(
        name="NVIDIA GeForce RTX 4090",
        vendor="NVIDIA",
        hardware_type=HardwareType.GPU,
        architecture="Ada Lovelace (AD102)",
        target_device="Desktop Workstation & Enthusiast PC",
        process_node_nm=4.0,
        transistor_count_billion=76.3,
        die_size_mm2=608.5,
        packaging="TSMC 4N Monolithic",
        memory_type="GDDR6X",
        memory_bus_width_bits=384,
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
        blueprint_blocks=[
            BlueprintBlock("128x Ada Lovelace SMs", "compute", "16,384 CUDA Cores with 3rd-Gen RT Cores and Shader Execution Reordering", count=128, area_percent=60.0, specs={"CUDA Cores": "16,384", "FP32": "82.6 TFLOPs"}),
            BlueprintBlock("512x 4th-Gen Tensor Cores", "tensor", "FP8 Transformer Engine accelerators", count=512, area_percent=16.0, specs={"FP8": "660 TFLOPs", "FP16": "330 TFLOPs"}),
            BlueprintBlock("72 MB Ultra-Large L2 Cache", "cache", "Keeps high-frequency working sets on-chip to save memory bandwidth", count=1, area_percent=12.0, specs={"Capacity": "72 MB"}),
            BlueprintBlock("384-bit GDDR6X Memory Controller", "memory_ctrl", "Micron GDDR6X running at 21 Gbps (1,008 GB/s)", count=6, area_percent=8.0, specs={"Bandwidth": "1,008 GB/s", "VRAM": "24 GB"}),
            BlueprintBlock("Dual 8th-Gen NVENC with AV1", "media", "Dual encoders for simultaneous 8K60 AV1 livestreaming & recording", count=2, area_percent=4.0, specs={"AV1": "Dual 8K60 Encoders"}),
        ],
        capabilities=[
            "Premise workstation LLM inference: Llama-3 8B at 110+ tokens/sec and 70B (4-bit) at 15 tok/sec",
            "4K Ultra 144+ FPS AAA gaming with full Path Tracing and DLSS 3.5 Ray Reconstruction",
            "Stable Diffusion generation at over 40 images/min with TensorRT acceleration",
        ],
    ),

    "amd_mi300x": HardwareSpec(
        name="AMD Instinct MI300X",
        vendor="AMD",
        hardware_type=HardwareType.GPU,
        architecture="CDNA 3 (XCD Chiplets)",
        target_device="Hyperscale AI Cloud",
        process_node_nm=5.0,
        transistor_count_billion=153.0,
        die_size_mm2=1017.0,
        packaging="TSMC 3D Chiplet Stacking (XCDs on IODs with 3D Hybrid Bonding)",
        memory_type="HBM3",
        memory_bus_width_bits=8192,
        tdp_watts=750.0,
        description="AMD flagship generative AI accelerator featuring 192 GB HBM3 at 5.3 TB/s and 304 Compute Units.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=5300.0,  # 5.3 TB/s HBM3
            hbm_capacity_gb=192.0,
            sram_per_core_kb=128.0,
            l2_cache_mb=256.0,         # 256 MB Infinity Cache
            num_compute_units=304,
            register_file_kb_per_core=256.0,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP64: 81.7,
                Precision.FP32: 163.4,
                Precision.FP16: 1307.0,
                Precision.BF16: 1307.0,
                Precision.FP8: 2614.0,
                Precision.INT8: 2614.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("8x XCD Accelerator Complex Dies (304 CUs)", "compute", "304 CDNA 3 Compute Units stacked in 3D over base dies", count=304, area_percent=55.0, specs={"CUs": "304", "BF16": "1,307 TFLOPs"}),
            BlueprintBlock("Matrix Core Accelerators", "tensor", "Dedicated matrix multiply engines supporting FP8, BF16, and INT8", count=304, area_percent=18.0, specs={"FP8 Peak": "2,614 TFLOPs"}),
            BlueprintBlock("256 MB AMD Infinity Cache", "cache", "4 base IOD dies providing massive shared high-speed SRAM cache", count=4, area_percent=12.0, specs={"Capacity": "256 MB", "Bandwidth": "17 TB/s"}),
            BlueprintBlock("8x HBM3 Memory Stacks", "memory_ctrl", "192 GB HBM3 memory running at 5.3 TB/s bandwidth", count=8, area_percent=10.0, specs={"Bandwidth": "5,300 GB/s (5.3 TB/s)", "VRAM": "192 GB"}),
            BlueprintBlock("Infinity Fabric Links", "interconnect", "Coherent multi-GPU interconnect at 896 GB/s", count=8, area_percent=5.0, specs={"Bandwidth": "896 GB/s"}),
        ],
        capabilities=[
            "192 GB memory capacity fits 70B parameter models in 16-bit without multi-GPU sharding",
            "5.3 TB/s memory bandwidth provides class-leading token generation throughput in memory-bound phases",
            "Full open-source ROCm software stack with native PyTorch and vLLM integration",
        ],
    ),

    # --------------------------------------------------------------------------
    # GOOGLE TPUS
    # --------------------------------------------------------------------------
    "google_tpu_v5p": HardwareSpec(
        name="Google TPU v5p",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v5p (Performance Flagship)",
        target_device="Google Cloud AI Supercomputer",
        process_node_nm=4.0,
        transistor_count_billion=45.0,
        die_size_mm2=650.0,
        packaging="TSMC CoWoS 2.5D",
        memory_type="HBM3",
        memory_bus_width_bits=4096,
        tdp_watts=300.0,
        description="Google TPU v5p with high-bandwidth ICI (Inter-Chip Interconnect) and 459 BF16 TFLOPs.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=1600.0,  # 1.6 TB/s HBM3
            hbm_capacity_gb=95.0,
            sram_per_core_kb=32768.0,  # 32 MB VMEM per Core
            l2_cache_mb=0.0,
            num_compute_units=2,       # 2 Cores per chip (each with 4 MXUs)
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 57.0,
                Precision.BF16: 459.0,
                Precision.FP8: 918.0,
                Precision.INT8: 918.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("8x Matrix Multiply Units (MXUs)", "tensor", "Systolic 128x128 2D matrix multiply arrays running at 1.75 GHz", count=8, area_percent=42.0, specs={"BF16 Peak": "459 TFLOPs", "Architecture": "Systolic Array"}),
            BlueprintBlock("2x Vector Processing Units (VPUs)", "compute", "General-purpose SIMD vector units for activations (GELU, Softmax, RMSNorm)", count=2, area_percent=18.0, specs={"Precision": "FP32 / BF16"}),
            BlueprintBlock("64 MB Software-Managed VMEM (SRAM)", "cache", "Direct software-controlled scratchpad memory bypassing cache misses", count=2, area_percent=20.0, specs={"Capacity": "64 MB Total", "SRAM/Core": "32 MB"}),
            BlueprintBlock("HBM3 Memory Controller", "memory_ctrl", "95 GB HBM3 memory running at 1.6 TB/s", count=4, area_percent=12.0, specs={"Bandwidth": "1,600 GB/s", "Capacity": "95 GB"}),
            BlueprintBlock("Optical Inter-Chip Interconnect (ICI)", "interconnect", "4,800 Gbps 3D Torus networking fabric interconnecting up to 8,960 chips", count=6, area_percent=8.0, specs={"Speed": "4,800 Gbps", "Topology": "3D Torus"}),
        ],
        capabilities=[
            "Trained Google Gemini 1.5 Pro and Gemini Ultra foundation models",
            "Huge 32 MB VMEM scratchpad per core enables massive tile sizes with zero reload penalty",
            "Scales up to 8,960 TPU chips in a single optical pod with near-linear scaling",
        ],
    ),

    "google_tpu_v5e": HardwareSpec(
        name="Google TPU v5e",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v5e (ViperLite)",
        target_device="Google Cloud Cost-Efficient AI",
        process_node_nm=5.0,
        transistor_count_billion=22.0,
        die_size_mm2=340.0,
        packaging="TSMC CoWoS 2.5D",
        memory_type="HBM2e",
        memory_bus_width_bits=2048,
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
        blueprint_blocks=[
            BlueprintBlock("4x Systolic Matrix Units (MXUs)", "tensor", "128x128 systolic matrix engines", count=4, area_percent=45.0, specs={"BF16": "197 TFLOPs", "INT8": "394 TOPs"}),
            BlueprintBlock("16 MB On-Chip VMEM Scratchpad", "cache", "Fast on-die vector memory", count=1, area_percent=25.0, specs={"Capacity": "16 MB"}),
            BlueprintBlock("HBM2e Memory Controller", "memory_ctrl", "16 GB HBM2e at 819 GB/s", count=2, area_percent=18.0, specs={"Bandwidth": "819 GB/s"}),
            BlueprintBlock("Inter-Chip Interconnect (ICI)", "interconnect", "2D Torus network interconnecting 256 pods", count=4, area_percent=12.0, specs={"Bandwidth": "1,600 Gbps"}),
        ],
        capabilities=[
            "Cost-effective inference for Gemma, LLaMA, and BERT at ultra-low price per million tokens",
            "High power efficiency: 197 TFLOPs at only 150 Watts TDP",
        ],
    ),

    "google_tpu_v4": HardwareSpec(
        name="Google TPU v4",
        vendor="Google",
        hardware_type=HardwareType.TPU,
        architecture="TPU v4 (Systolic Matrix Multiply Unit)",
        target_device="Google Cloud Datacenter",
        process_node_nm=7.0,
        transistor_count_billion=32.0,
        die_size_mm2=500.0,
        packaging="TSMC CoWoS",
        memory_type="HBM2",
        memory_bus_width_bits=4096,
        tdp_watts=200.0,
        description="Google Tensor Processing Unit v4 with 128x128 3D Torus Interconnect & 2 Matrix Multiply Units (MXUs).",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=1200.0,  # 1.2 TB/s HBM2
            hbm_capacity_gb=32.0,
            sram_per_core_kb=16384.0,  # 16 MB Vector Memory (VMEM) / Scratchpad per TensorCore
            l2_cache_mb=0.0,
            num_compute_units=2,
        ),
        compute=ComputeCapacity(
            dense_tflops={
                Precision.FP32: 34.0,
                Precision.BF16: 275.0,
                Precision.INT8: 550.0,
            },
        ),
        blueprint_blocks=[
            BlueprintBlock("4x Systolic Matrix Engines", "tensor", "Systolic arrays for tensor GEMM operations", count=4, area_percent=46.0, specs={"BF16": "275 TFLOPs"}),
            BlueprintBlock("32 MB Vector Memory (VMEM)", "cache", "16 MB scratchpad per core", count=2, area_percent=24.0, specs={"Capacity": "32 MB"}),
            BlueprintBlock("HBM2 Controller", "memory_ctrl", "32 GB HBM2 at 1.2 TB/s", count=4, area_percent=18.0, specs={"Bandwidth": "1,200 GB/s"}),
            BlueprintBlock("Optical Circuit Switch (OCS) ICI", "interconnect", "Dynamic optical switching 3D torus interconnect", count=6, area_percent=12.0, specs={"Technology": "Optical Circuit Switch"}),
        ],
        capabilities=[
            "Foundation training engine for PaLM, Gemini 1.0, and Imagen",
            "Optical Circuit Switching allows reconfiguring pod topologies dynamically without recabling",
        ],
    ),

    # --------------------------------------------------------------------------
    # CUSTOM OPEN-SOURCE SILICON / ASICs
    # --------------------------------------------------------------------------
    "custom_open_tpu_130nm": HardwareSpec(
        name="OpenTPU-130 (SkyWater 130nm)",
        vendor="Open Silicon",
        hardware_type=HardwareType.CUSTOM_ASIC,
        architecture="16x16 Weight-Stationary Systolic Array",
        target_device="Open-Source Silicon / TinyTapeout Edge ASIC",
        process_node_nm=130.0,
        transistor_count_billion=0.005,  # 5 Million transistors
        die_size_mm2=10.0,
        packaging="QFN-64 Open Source Package",
        memory_type="HyperRAM / QSPI",
        memory_bus_width_bits=16,
        tdp_watts=0.5,
        description="Open-Source Silicon Matrix Multiply Accelerator targeted for SkyWater 130nm / TinyTapeout / Efabless.",
        memory=MemoryHierarchy(
            hbm_bandwidth_gbs=0.8,     # 800 MB/s HyperRAM or QSPI off-chip
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
        blueprint_blocks=[
            BlueprintBlock("16x16 Weight-Stationary Systolic Array", "tensor", "256 8-bit Multiply-Accumulate (MAC) processing elements", count=1, area_percent=55.0, specs={"GOPs": "51.2 GOPs", "Frequency": "100 MHz"}),
            BlueprintBlock("64 KB OpenRAM Scratchpad", "cache", "SkyWater 130nm open-source SRAM macro for activation tiling", count=1, area_percent=25.0, specs={"Capacity": "64 KB", "Type": "OpenRAM"}),
            BlueprintBlock("HyperRAM / Octal SPI Interface", "memory_ctrl", "Low-pin-count off-chip memory interface running at 800 MB/s", count=1, area_percent=12.0, specs={"Bandwidth": "800 MB/s"}),
            BlueprintBlock("RISC-V Control Core (VexRiscv)", "compute", "Lightweight RV32I controller executing command queues", count=1, area_percent=8.0, specs={"Arch": "RV32I"}),
        ],
        capabilities=[
            "Fully open-source tapeout verifiable via OpenLane and SkyWater 130nm PDK",
            "Ultra-low power (500 mW) edge keyword spotting and vibration anomaly detection",
            "Educational model for hardware-software co-design & systolic dataflow verification",
        ],
    ),
}

# Aliases for fast user lookup
ALIAS_MAP: Dict[str, str] = {
    # Mobile
    "a17": "apple_a17_pro",
    "a17pro": "apple_a17_pro",
    "apple_a17": "apple_a17_pro",
    "iphone": "apple_a17_pro",
    "iphone15pro": "apple_a17_pro",
    "snapdragon": "snapdragon_8_gen3",
    "snapdragon8gen3": "snapdragon_8_gen3",
    "sd8gen3": "snapdragon_8_gen3",
    "tensor_g4": "google_tensor_g4",
    "tensorg4": "google_tensor_g4",
    "pixel9": "google_tensor_g4",
    "pixel": "google_tensor_g4",
    "dimensity": "mediatek_dimensity_9300",
    "dimensity9300": "mediatek_dimensity_9300",
    # Laptop & PC
    "m4": "apple_m4",
    "applem4": "apple_m4",
    "macbook_m4": "apple_m4",
    "m3max": "apple_m3_max",
    "applem3max": "apple_m3_max",
    "macbook_m3max": "apple_m3_max",
    "lunarlake": "intel_lunar_lake_288v",
    "lunar_lake": "intel_lunar_lake_288v",
    "coreultra288v": "intel_lunar_lake_288v",
    "288v": "intel_lunar_lake_288v",
    "strixpoint": "amd_ryzen_ai_9_hx370",
    "ryzenai": "amd_ryzen_ai_9_hx370",
    "hx370": "amd_ryzen_ai_9_hx370",
    "rtx4060": "nvidia_rtx_4060_mobile",
    "rtx4060mobile": "nvidia_rtx_4060_mobile",
    "4060laptop": "nvidia_rtx_4060_mobile",
    # Cloud & GPUs
    "h100": "nvidia_h100_sxm",
    "h100_sxm": "nvidia_h100_sxm",
    "a100": "nvidia_a100_sxm",
    "a100_sxm": "nvidia_a100_sxm",
    "b200": "nvidia_b200",
    "blackwell": "nvidia_b200",
    "rtx4090": "nvidia_rtx_4090",
    "4090": "nvidia_rtx_4090",
    "mi300": "amd_mi300x",
    "mi300x": "amd_mi300x",
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
