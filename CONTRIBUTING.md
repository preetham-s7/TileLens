# Contributing to TileLens

Thank you for your interest in contributing to **TileLens**! TileLens is an open-source tool built to advance semiconductor hardware-software co-design, GPU/TPU profiling, and compiler optimization.

---

## 🛠️ How to Contribute

### 1. Adding New Hardware Profiles
We actively welcome accurate hardware profiles for new accelerators, including:
- **AMD Instinct** (MI300X, MI300A, MI250X)
- **Tenstorrent** (Wormhole, Blackhole)
- **Groq LPU** (Tensor Streaming Processor)
- **Apple Silicon** (M3/M4 Max)
- **Open-source ASICs** (Tiny Tapeout designs, RISC-V Vector cores)

To add a new accelerator:
1. Open `tilelens/hardware/database.py`.
2. Add a new `HardwareSpec` instance with verified peak TFLOPs across precisions, HBM bandwidth, SRAM per core, and L2 cache sizes.
3. Add convenient aliases to `ALIAS_MAP`.
4. Add a test in `tests/test_roofline.py`.

### 2. Enhancing Kernel Models
Currently, TileLens provides cycle and memory models for GEMM. High-value areas for expansion:
- **FlashAttention-2 and FlashAttention-3** ($Q, K, V$ tiling with online softmax rescale).
- **Conv2D / Winograd Convolution** modeling.
- **Sparse GEMM / 2:4 Structured Sparsity** tensor core emulation.
- **Triton / StableHLO AST parser** to automatically extract tile sizes from kernel code.

### 3. Improving the Visual Dashboard
- Add 3D WebGL memory cube rendering.
- Add real-time interactive sliders in HTML to dynamically tweak $T_M, T_N, T_K$ and watch roofline points move live.

---

## 🧪 Development Workflow

```bash
# 1. Fork & clone the repo
git clone https://github.com/preetham-s7/TileLens.git
cd TileLens

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev,viz]"

# 4. Run tests
python -m unittest discover -s tests

# Dashboard calculation and interaction regression tests (requires Node.js)
node --test tests/test_dashboard.cjs
```

---

## 📜 Code Style & Standards
- Write clean, type-annotated Python (`typing`, `dataclasses`).
- Include docstrings explaining architectural concepts.
- Ensure all tests pass before submitting a Pull Request.
