"""Kernel profiling for bottleneck analysis."""
from kernel_anvil.analyze import ProfileMetrics
from kernel_anvil.rdna3 import GpuSpec
from kernel_anvil.verify import _benchmark


def _estimate_vgpr(config: dict) -> int:
    """Estimate VGPR usage from config parameters.

    Heuristic: BLOCK_N * num_warps * ~4 registers per element,
    clamped to a reasonable range. This is a rough estimate --
    Phase 2 will use rocprofv3 for actual register counts.
    """
    block_n = config.get("BLOCK_N", 64)
    num_warps = config.get("num_warps", 4)
    num_stages = config.get("num_stages", 1)
    # Base register pressure from tile size and pipeline depth
    estimated = block_n * num_warps * 4 // 32  # scale down by wave size
    # Pipeline stages add register overhead for buffering
    estimated += num_stages * 8
    # Clamp to valid RDNA3 range (0-1536 per SIMD, but per-wave max ~256 is typical)
    return max(16, min(estimated, 256))


def _estimate_lds(config: dict) -> int:
    """Estimate LDS usage from config parameters.

    Heuristic based on tile footprint: BLOCK_N * BLOCK_K * element_size * num_stages.
    """
    block_n = config.get("BLOCK_N", 64)
    block_k = config.get("BLOCK_K", 128)
    num_stages = config.get("num_stages", 1)
    element_bytes = config.get("element_bytes", 2)  # FP16 default
    return block_n * block_k * element_bytes * num_stages


def profile_kernel(
    kernel_fn,
    config: dict,
    data_bytes: int | None = None,
    gpu_spec: GpuSpec | None = None,
    warmup: int = 5,
    runs: int = 20,
) -> ProfileMetrics:
    """Profile a kernel and return metrics for bottleneck analysis.

    Collects timing via CUDA events (or CPU fallback) and estimates
    bandwidth. Register/LDS counts are estimated from config if not
    provided by rocprofv3 (Phase 2 will add full rocprofv3 integration).

    Args:
        kernel_fn: callable(config) -> output tensor
        config: config dict passed to kernel_fn
        data_bytes: total bytes moved (for bandwidth calculation)
        gpu_spec: GpuSpec for occupancy calculation
        warmup: number of warmup iterations
        runs: number of timed iterations
    """
    # Benchmark
    latency_us = _benchmark(kernel_fn, config, warmup, runs)
    duration_ns = latency_us * 1000  # us -> ns

    # Estimate register and LDS usage from config
    vgpr_count = _estimate_vgpr(config)
    lds_bytes = _estimate_lds(config)

    # Compute bandwidth
    bandwidth_gbs = 0.0
    if data_bytes is not None and latency_us > 0:
        duration_s = latency_us / 1e6
        bandwidth_gbs = data_bytes / duration_s / 1e9

    # Compute occupancy if gpu_spec provided
    occupancy_pct = 0.0
    limiting_factor = "unknown"
    threads_per_wg = config.get("num_warps", 4) * 32  # warps * wave_size
    if gpu_spec is not None:
        occupancy_pct, limiting_factor = gpu_spec.occupancy(
            vgpr_count, lds_bytes, threads_per_wg
        )

    return ProfileMetrics(
        duration_ns=duration_ns,
        vgpr_count=vgpr_count,
        lds_bytes=lds_bytes,
        scratch_bytes=0,  # can't estimate without rocprofv3
        bandwidth_gbs=bandwidth_gbs,
        occupancy_pct=occupancy_pct,
        limiting_factor=limiting_factor,
        threads_per_wg=threads_per_wg,
    )
