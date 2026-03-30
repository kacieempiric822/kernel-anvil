# smithy Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Working CLI that profiles Triton kernels on RDNA3, classifies bottlenecks, runs guided config sweeps, and reports improvements

**Architecture:** Profile via rocprofv3 -> classify bottleneck -> generate targeted configs -> verify + benchmark -> report best

**Tech Stack:** Python 3.14, PyTorch 2.10+ROCm 7.1, Triton 3.6, rocprofv3, pytest

---

### Task 1: Project Scaffolding + RDNA3 Constants

**Files:**
- Create: `smithy/pyproject.toml`
- Create: `smithy/__init__.py`
- Create: `smithy/rdna3.py`
- Create: `tests/__init__.py`
- Create: `LICENSE`

Create the package with RDNA3 hardware constants and optimization heuristics. This is the knowledge base that drives all optimization decisions.

rdna3.py should contain:
- GPU specs for gfx1100/1101/1102 (bandwidth, CUs, VGPR, LDS, wave size, etc.)
- Proven optimization heuristics from AEGIS kernel tournament results
- Occupancy calculation functions (vgpr_waves, lds_waves)
- A function to detect the current GPU and return its spec

---

### Task 2: Bottleneck Analyzer

**Files:**
- Create: `smithy/analyze.py`
- Create: `tests/test_analyze.py`

Takes raw profile metrics (duration_ns, vgpr_count, lds_bytes, scratch_bytes, bandwidth_gbs, occupancy_pct) and classifies the bottleneck.

Classes: bandwidth_bound, occupancy_limited_vgpr, occupancy_limited_lds, register_spill, launch_overhead, compute_bound.

Returns a BottleneckReport dataclass with: classification, severity (0-1), limiting_factor, recommended_directions (list of strings).

Tests should cover each bottleneck class with synthetic profile data.

---

### Task 3: Config Sweep Generator

**Files:**
- Create: `smithy/sweep.py`
- Create: `tests/test_sweep.py`

Given a BottleneckReport, generates a set of Triton config dicts to try. The configs are GUIDED by the bottleneck -- not brute force.

For each bottleneck class, define which config parameters to vary and in which direction:
- occupancy_limited_vgpr: lower BLOCK_N (32, 64), fewer num_warps (1, 2), fewer num_stages (1, 2)
- bandwidth_bound: larger BLOCK_K (128, 256, 512), try SPLIT_K (2, 4, 8)
- register_spill: reduce all block dimensions, minimize stages
- compute_bound: larger BLOCK_N (128, 256), more warps (4, 8)
- launch_overhead: suggest fusion (return a message, can't auto-fuse)

Output: list of config dicts like {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2}

Tests should verify that each bottleneck class produces configs that move in the right direction (e.g., occupancy_limited_vgpr produces configs with lower BLOCK_N than the baseline).

---

### Task 4: Kernel Verifier + Benchmarker

**Files:**
- Create: `smithy/verify.py`
- Create: `tests/test_verify.py`

Takes a kernel function, a config, input tensors, and a reference output. Runs the kernel with the config, checks correctness (torch.allclose), benchmarks timing (median of N runs with warmup and cuda sync).

Returns a VerifyResult dataclass: correct (bool), latency_us (float), bandwidth_gbs (float or None), speedup_vs_baseline (float or None).

The benchmarking pattern:
```python
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
# warmup
for _ in range(warmup):
    kernel[grid](*args)
# timed runs
times = []
for _ in range(runs):
    start.record()
    kernel[grid](*args)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
median_ms = sorted(times)[len(times) // 2]
```

Tests can use a simple vector add Triton kernel as the test case (no GPU required for the correctness logic tests; GPU tests should skip if no CUDA).

---

### Task 5: Profiler Wrapper

**Files:**
- Create: `smithy/profile.py`
- Create: `tests/test_profile.py`

Programmatic wrapper around rocprofv3 (leveraging the triton_prof infrastructure from AEGIS). This is the bridge between rocprofv3's subprocess output and smithy's analyzer.

For Phase 1, a simpler approach: instead of running rocprofv3 (which requires subprocess management and CSV parsing), use torch.cuda.Event timing + Triton's built-in kernel metadata to extract:
- Duration (from CUDA events)
- Grid/block dimensions (from the kernel launch)
- VGPR/SGPR count (from triton.compile() metadata if available)

This avoids the rocprofv3 dependency for Phase 1 while still providing enough data for bottleneck classification. Full rocprofv3 integration is Phase 2.

Fallback for register counts: estimate from block dimensions and Triton's compiler output, or accept user-provided values.

---

### Task 6: CLI

**Files:**
- Create: `smithy/cli.py`
- Modify: `smithy/__init__.py` (add CLI entry point)
- Modify: `pyproject.toml` (add console_scripts)

Commands:
- `smithy sweep runner.py` -- run guided config sweep on a runner script
- `smithy compare runner_a.py runner_b.py` -- compare two kernel implementations

The runner script convention: a Python file with `setup()`, `run(inputs)`, `reference(inputs)`, and metadata (KERNEL, DATA_BYTES).

CLI uses argparse. Output is a Rich-formatted table showing configs, latency, speedup, and the winning config.

---

### Task 7: Example Runner + End-to-End Test

**Files:**
- Create: `examples/simple_gemv.py` -- simple FP16 GEMV runner (not MXFP4, keep it portable)
- Create: `tests/test_e2e.py` -- end-to-end test using the example runner

The example runner implements a basic Triton GEMV kernel (not MXFP4 -- too complex for an example). Uses standard FP16 matmul/GEMV pattern.

The e2e test runs `smithy sweep` on the example and verifies:
1. Multiple configs are tested
2. At least one passes correctness
3. Results are ranked by latency
4. Output format is correct

This test requires a GPU and should skip gracefully without one.

---

### Task 8: README

**Files:**
- Create: `README.md`

Tight README with:
- One-line description
- Install
- Quick start (3 commands)
- Example output (table of configs + speedups)
- How it works (profile -> analyze -> sweep -> verify)
- RDNA3 knowledge base section
- Link to papers (KernelSkill, CUDA Agent, KernelFoundry -- for context on what this is an AMD equivalent of)
- Limitations (Phase 1: config sweep only, no code rewrites)
