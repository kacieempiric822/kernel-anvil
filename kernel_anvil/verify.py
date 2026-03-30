"""Kernel correctness verification and benchmarking."""
import time
from dataclasses import dataclass

import torch

HAS_CUDA = torch.cuda.is_available()


@dataclass
class VerifyResult:
    config: dict
    correct: bool
    max_diff: float         # max absolute difference from reference
    latency_us: float       # median latency in microseconds
    bandwidth_gbs: float | None  # effective bandwidth if data_bytes provided
    speedup: float | None   # vs baseline latency if provided


def _benchmark(fn, config: dict, warmup: int, runs: int) -> float:
    """Benchmark a kernel function and return median latency in microseconds.

    Uses CUDA events when a GPU is available, falls back to
    time.perf_counter on CPU-only machines (for testing).
    """
    if HAS_CUDA:
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn(config)
        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(config)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us
        return sorted(times)[len(times) // 2]
    else:
        for _ in range(warmup):
            fn(config)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(config)
            times.append((time.perf_counter() - t0) * 1e6)  # seconds -> us
        return sorted(times)[len(times) // 2]


def verify_and_bench(
    kernel_fn,
    reference_output: torch.Tensor,
    config: dict,
    warmup: int = 5,
    runs: int = 10,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    data_bytes: int | None = None,
    baseline_latency_us: float | None = None,
) -> VerifyResult:
    """Run kernel with config, verify correctness, benchmark timing.

    Args:
        kernel_fn: callable(config) -> output tensor
        reference_output: expected output tensor to compare against
        config: config dict passed to kernel_fn
        warmup: number of warmup iterations before timing
        runs: number of timed iterations
        atol: absolute tolerance for correctness check
        rtol: relative tolerance for correctness check
        data_bytes: total bytes moved (for bandwidth calculation)
        baseline_latency_us: baseline latency for speedup calculation
    """
    # Run the kernel to get output for correctness check
    output = kernel_fn(config)

    # Check correctness
    correct = torch.allclose(output, reference_output, atol=atol, rtol=rtol)
    max_diff = (output - reference_output).abs().max().item()

    # Benchmark
    latency_us = _benchmark(kernel_fn, config, warmup, runs)

    # Bandwidth: data_bytes / duration
    bandwidth_gbs = None
    if data_bytes is not None and latency_us > 0:
        duration_s = latency_us / 1e6
        bandwidth_gbs = data_bytes / duration_s / 1e9  # bytes/s -> GB/s

    # Speedup vs baseline
    speedup = None
    if baseline_latency_us is not None and latency_us > 0:
        speedup = baseline_latency_us / latency_us

    return VerifyResult(
        config=config,
        correct=correct,
        max_diff=max_diff,
        latency_us=latency_us,
        bandwidth_gbs=bandwidth_gbs,
        speedup=speedup,
    )
