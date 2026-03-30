"""Vulkan backend sweep for kernel-anvil.

Instead of generating HIP kernels, this sweeps Vulkan's existing mul_mat_vec
shader with different specialization constants (workgroup size, NUM_ROWS).

The approach:
1. Build llama.cpp with Vulkan backend
2. Run llama-bench with different configs
3. Compare tok/s across configs
4. Write optimal config for Vulkan pipeline tuning

Vulkan's mul_mat_vec_q4_k.comp uses specialization constants:
  - spec_const[0] = local_size_x (workgroup size, usually subgroup_size * N)
  - spec_const[1] = NUM_ROWS (rows per workgroup, default rm_kq=2 on AMD)

These are set at pipeline creation time in ggml-vulkan.cpp.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from kernel_anvil.gguf import parse_gguf


@dataclass
class VulkanBenchResult:
    """Result of a Vulkan llama-bench run."""
    config_name: str
    pp_tok_s: float
    tg_tok_s: float


def run_vulkan_bench(
    llama_bench: str,
    model_path: str,
    prompt_tokens: int = 512,
    gen_tokens: int = 128,
    runs: int = 3,
) -> VulkanBenchResult:
    """Run llama-bench with Vulkan backend and return results."""
    cmd = [
        llama_bench,
        "-m", model_path,
        "-ngl", "999",
        "-t", "12",
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(runs),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
    )

    pp_tps = 0.0
    tg_tps = 0.0

    for line in result.stdout.split("\n"):
        if f"pp{prompt_tokens}" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                try:
                    val = float(p.split("\xb1")[0].strip())
                    if val > 10:
                        pp_tps = val
                except (ValueError, IndexError):
                    pass
        if f"tg{gen_tokens}" in line:
            parts = line.split("|")
            for p in parts:
                p = p.strip()
                try:
                    val = float(p.split("\xb1")[0].strip())
                    if val > 1:
                        tg_tps = val
                except (ValueError, IndexError):
                    pass

    return VulkanBenchResult(
        config_name="vulkan",
        pp_tok_s=pp_tps,
        tg_tok_s=tg_tps,
    )


def compare_backends(
    model_path: str,
    rocm_bench: str | None = None,
    vulkan_bench: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run head-to-head ROCm vs Vulkan comparison.

    Args:
        model_path: Path to GGUF model.
        rocm_bench: Path to ROCm llama-bench. Auto-detected if None.
        vulkan_bench: Path to Vulkan llama-bench. Auto-detected if None.
        verbose: Print progress.

    Returns:
        Dict with comparison results.
    """
    # Auto-detect binaries
    search_paths = [
        Path.home() / "Projects/llama-cpp-mainline/build-rocm/bin/llama-bench",
        Path.home() / "Projects/llama-cpp-turboquant/build/bin/llama-bench",
    ]
    if rocm_bench is None:
        for p in search_paths:
            if p.exists():
                rocm_bench = str(p)
                break

    search_paths_vk = [
        Path.home() / "Projects/llama-cpp-mainline/build-vulkan/bin/llama-bench",
        Path.home() / "Projects/llama-cpp-turboquant/build-vulkan/bin/llama-bench",
    ]
    if vulkan_bench is None:
        for p in search_paths_vk:
            if p.exists():
                vulkan_bench = str(p)
                break

    results = {}

    if rocm_bench and Path(rocm_bench).exists():
        if verbose:
            print("Running ROCm benchmark...")
        rocm = run_vulkan_bench(rocm_bench, model_path)
        results["rocm"] = {"pp": rocm.pp_tok_s, "tg": rocm.tg_tok_s}
        if verbose:
            print(f"  ROCm: pp={rocm.pp_tok_s:.1f} tg={rocm.tg_tok_s:.1f} tok/s")

    if vulkan_bench and Path(vulkan_bench).exists():
        if verbose:
            print("Running Vulkan benchmark...")
        vulkan = run_vulkan_bench(vulkan_bench, model_path)
        results["vulkan"] = {"pp": vulkan.pp_tok_s, "tg": vulkan.tg_tok_s}
        if verbose:
            print(f"  Vulkan: pp={vulkan.pp_tok_s:.1f} tg={vulkan.tg_tok_s:.1f} tok/s")

    if "rocm" in results and "vulkan" in results:
        tg_speedup = results["vulkan"]["tg"] / results["rocm"]["tg"] if results["rocm"]["tg"] > 0 else 0
        pp_speedup = results["rocm"]["pp"] / results["vulkan"]["pp"] if results["vulkan"]["pp"] > 0 else 0
        results["comparison"] = {
            "vulkan_tg_speedup": tg_speedup,
            "rocm_pp_speedup": pp_speedup,
            "recommendation": "vulkan" if tg_speedup > 1.1 else "rocm" if pp_speedup > 1.1 else "similar",
        }
        if verbose:
            print(f"\n  Decode: Vulkan is {tg_speedup:.1f}x faster")
            print(f"  Prefill: ROCm is {pp_speedup:.1f}x faster")
            if tg_speedup > 1.1:
                print(f"  Recommendation: Use Vulkan for interactive/chat workloads")
            elif pp_speedup > 1.1:
                print(f"  Recommendation: Use ROCm for batch/prefill-heavy workloads")

    return results
