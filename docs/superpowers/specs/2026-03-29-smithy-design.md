# smithy -- Profile-Guided Triton Kernel Optimizer for AMD/RDNA3

**Date:** 2026-03-29
**Status:** Design approved, ready for implementation

## Summary

CLI tool that profiles Triton kernels on AMD GPUs via rocprofv3, classifies bottlenecks (bandwidth/occupancy/spill/launch-overhead), runs guided config sweeps, verifies correctness, and reports improvements. First kernel optimization tool targeting AMD/RDNA3 -- every existing tool (KernelSkill, CUDA Agent, KernelFoundry, TritonForge) targets NVIDIA exclusively.

## Architecture

```
smithy optimize kernel.py --entry my_kernel
    |
    v
Profile (rocprofv3) -> Analyze (bottleneck class) -> Sweep (guided configs)
    -> Verify (allclose + benchmark) -> Report (best config, speedup)
```

### Profile (smithy/profile.py)

Wraps the existing AEGIS `triton_prof` infrastructure for programmatic use. Runs a Triton kernel via rocprofv3, parses the kernel trace CSV, computes derived metrics.

Input: kernel function + launch args + data sizes
Output: ProfileResult with timing_ns, vgpr_count, sgpr_count, lds_bytes, scratch_bytes, occupancy_pct, bandwidth_gbs, limiting_factor

The existing triton_prof already handles:
- rocprofv3 subprocess management
- CSV parsing (kernel_trace.csv, agent_info.csv)
- RDNA3 hardware constants (gfx1100=960 GB/s, 96 CUs, etc.)
- Occupancy calculation (VGPR/LDS limiting)
- Bandwidth computation

smithy/profile.py provides a clean programmatic API on top of this.

### Analyze (smithy/analyze.py)

Classifies kernel bottleneck from profile data:

| Class | Condition | Optimization Direction |
|-------|-----------|----------------------|
| bandwidth_bound | BW util > 60% | Increase arithmetic intensity, reduce memory traffic |
| occupancy_limited_vgpr | VGPR limits waves | Reduce BLOCK_N, fewer warps, fewer stages |
| occupancy_limited_lds | LDS limits waves | Reduce shared memory usage |
| register_spill | scratch_bytes > 0 | Reduce register pressure (smaller blocks, fewer stages) |
| launch_overhead | duration < 10us | Fuse with adjacent kernels, increase work per launch |
| compute_bound | BW util < 30%, occupancy > 70% | Increase BLOCK_N/K for more parallelism |

Returns a BottleneckReport with the classification, severity, and recommended optimization directions.

### Optimize (smithy/optimize.py)

Two modes:

**Config Sweep (Phase 1):** Given a bottleneck classification, generates a targeted set of Triton autotuning configs. Not brute force -- the sweep is guided by the bottleneck class:

- occupancy_limited_vgpr: prioritize lower BLOCK_N, fewer warps
- bandwidth_bound: prioritize larger BLOCK_K, try SPLIT_K
- register_spill: prioritize smaller blocks across all dimensions
- compute_bound: prioritize larger blocks, more warps

Generates configs as `triton.Config` dicts. Each config is compiled, profiled, verified, and benchmarked.

**LLM Rewrite (Phase 2, not tonight):** Feed kernel source + profile + RDNA3 knowledge to LLM for code-level rewrites.

### Verify (smithy/verify.py)

For each candidate config:
1. Run kernel with candidate config on test input
2. Compare output to reference (torch.allclose with configurable tolerance)
3. If correct, benchmark (median of N runs with warmup)
4. Record: config, correctness, latency_us, bandwidth_gbs, speedup_vs_baseline

### RDNA3 Knowledge Base (smithy/rdna3.py)

Hardware constants and optimization heuristics:

```python
RDNA3_GFX1100 = {
    "peak_bandwidth_gbs": 960,
    "cu_count": 96,
    "simds_per_cu": 2,
    "max_waves_per_simd": 10,
    "vgpr_per_simd": 1536,
    "vgpr_granule": 8,
    "lds_per_cu_bytes": 98304,  # 96 KB
    "lds_granule_bytes": 512,
    "wave_size": 32,
    "l2_size_bytes": 6291456,  # 6 MB
}

# Proven heuristics from AEGIS kernel tournament results
RDNA3_HEURISTICS = [
    "Scalar inner loops (static_range) beat 2D tile patterns",
    "LUT dequant beats arithmetic dequant (dependency chains)",
    "Two-pass SPLIT_K beats atomic for SPLIT_K > 4",
    "Atomic SPLIT_K is faster for small shapes (less launch overhead)",
    "Row-major access pattern wins over tl.trans()",
    "VGPR allocation is in blocks of 8 registers",
    "Coalesced loads: adjacent threads should access sequential bytes",
    "Register spilling to scratch destroys performance",
]
```

### CLI (smithy/cli.py)

```bash
# Profile a kernel (requires a runner script that exercises the kernel)
smithy profile runner.py

# Guided config sweep
smithy sweep runner.py --configs 20

# Compare two kernel implementations
smithy compare runner_a.py runner_b.py
```

The "runner" is a Python script that:
1. Creates test input tensors
2. Calls the kernel
3. Returns output tensor(s)

smithy introspects the runner to find the kernel, input shapes, and expected output.

## Runner Script Convention

A runner script is a Python file that exposes:

```python
# runner.py
import torch
import triton

@triton.jit
def my_kernel(...):
    ...

def setup():
    """Create input tensors."""
    return {"x": torch.randn(4096, device="cuda"),
            "w": torch.randn(4096, 1024, device="cuda")}

def run(inputs):
    """Execute the kernel, return output."""
    output = torch.empty(1024, device="cuda")
    my_kernel[(grid,)](inputs["x"], inputs["w"], output, ...)
    return output

def reference(inputs):
    """Reference implementation for correctness check."""
    return inputs["x"] @ inputs["w"]

# Metadata
KERNEL = my_kernel
DATA_BYTES = 4096 * 1024 * 2  # for bandwidth calculation
```

## Project Structure

```
smithy/
├── smithy/
│   ├── __init__.py
│   ├── cli.py           # argparse CLI
│   ├── profile.py       # rocprofv3 wrapper (programmatic API)
│   ├── analyze.py       # Bottleneck classification
│   ├── sweep.py         # Guided config sweep
│   ├── verify.py        # Correctness + benchmark
│   └── rdna3.py         # Hardware constants + heuristics
├── tests/
│   ├── test_analyze.py
│   ├── test_sweep.py
│   └── test_verify.py
├── examples/
│   └── gemv_runner.py   # Example runner using AEGIS MXFP4 GEMV
├── pyproject.toml
├── LICENSE
└── README.md
```

## Phase 1 Scope (Tonight)

1. Project scaffolding
2. rdna3.py -- hardware constants and heuristics
3. analyze.py -- bottleneck classification from profile metrics
4. sweep.py -- guided config generation based on bottleneck class
5. verify.py -- correctness check + benchmarking
6. cli.py -- `smithy sweep` and `smithy compare` commands
7. profile.py -- programmatic rocprofv3 wrapper
8. Example runner + test on real AEGIS MXFP4 kernel
9. README with usage and example output

## Success Criteria

1. `smithy sweep example_runner.py` produces a ranked table of configs with speedups
2. Bottleneck classification matches manual analysis (bandwidth-bound kernel identified as such)
3. Correctness verification catches incorrect configs
4. Runs end-to-end on the 7900 XTX
5. README with real benchmark output

## Limitations

- Phase 1 is config-sweep only (no code rewrites)
- Requires rocprofv3 (ROCm 6.0+)
- RDNA3 focused (gfx1100/1101/1102); CDNA constants not included yet
- Runner script convention requires user to structure their kernel test
