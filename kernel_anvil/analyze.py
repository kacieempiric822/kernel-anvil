"""Bottleneck classification from kernel profile metrics."""
from dataclasses import dataclass, field

from kernel_anvil.rdna3 import GpuSpec


@dataclass
class ProfileMetrics:
    """Raw metrics from profiling a kernel."""
    duration_ns: float
    vgpr_count: int = 0
    lds_bytes: int = 0
    scratch_bytes: int = 0
    bandwidth_gbs: float = 0.0
    occupancy_pct: float = 0.0
    limiting_factor: str = "unknown"
    threads_per_wg: int = 256


@dataclass
class BottleneckReport:
    """Classification of kernel's performance bottleneck."""
    classification: str  # bandwidth_bound, occupancy_limited_vgpr, etc.
    severity: float  # 0.0 (minor) to 1.0 (severe)
    limiting_factor: str
    directions: list[str] = field(default_factory=list)  # recommended optimization directions


def classify(metrics: ProfileMetrics, gpu_spec: GpuSpec) -> BottleneckReport:
    """Classify the kernel's bottleneck from profile metrics.

    Priority order (highest to lowest):
    1. register_spill - scratch usage always indicates a problem
    2. occupancy_limited_vgpr - low occupancy from register pressure
    3. occupancy_limited_lds - low occupancy from shared memory pressure
    4. bandwidth_bound - hitting memory bandwidth limits
    5. launch_overhead - kernel too short to be useful
    6. compute_bound - everything else (the good case)
    """
    # 1. Register spill -- always bad, highest priority
    if metrics.scratch_bytes > 0:
        return BottleneckReport(
            classification="register_spill",
            severity=1.0,
            limiting_factor="scratch",
            directions=[
                "Reduce BLOCK_N and BLOCK_K to lower register pressure",
                "Reduce num_warps to decrease per-wave register demand",
                "Set num_stages=1 to eliminate pipeline register overhead",
                "Check for unnecessary intermediate variables in kernel body",
                "Consider splitting complex kernels into multiple passes",
            ],
        )

    # Compute occupancy from gpu_spec if we have register/LDS info
    occ_pct = metrics.occupancy_pct
    factor = metrics.limiting_factor
    if metrics.vgpr_count > 0 or metrics.lds_bytes > 0:
        occ_pct, factor = gpu_spec.occupancy(
            metrics.vgpr_count, metrics.lds_bytes, metrics.threads_per_wg
        )

    # 2. Occupancy limited by VGPRs
    if occ_pct < 50.0 and factor == "vgpr":
        severity = 1.0 - (occ_pct / 50.0)  # 0% -> 1.0, 49% -> 0.02
        return BottleneckReport(
            classification="occupancy_limited_vgpr",
            severity=severity,
            limiting_factor="vgpr",
            directions=[
                "Reduce BLOCK_N to lower register usage per thread",
                "Reduce num_warps to decrease total register demand",
                "Set num_stages=1 or 2 to reduce pipeline register overhead",
                "Consider smaller BLOCK_K to reduce register-resident accumulators",
            ],
        )

    # 3. Occupancy limited by LDS
    if occ_pct < 50.0 and factor == "lds":
        severity = 1.0 - (occ_pct / 50.0)
        return BottleneckReport(
            classification="occupancy_limited_lds",
            severity=severity,
            limiting_factor="lds",
            directions=[
                "Reduce BLOCK_N and BLOCK_K to shrink LDS tile footprint",
                "Set num_stages=1 to minimize shared memory buffering",
                "Consider reducing threads_per_wg if workgroup is oversized",
            ],
        )

    # 4. Bandwidth bound
    bw_ratio = metrics.bandwidth_gbs / gpu_spec.peak_bandwidth_gbs if gpu_spec.peak_bandwidth_gbs > 0 else 0.0
    if bw_ratio > 0.6:
        severity = min(1.0, (bw_ratio - 0.6) / 0.4)  # 60% -> 0.0, 100% -> 1.0
        return BottleneckReport(
            classification="bandwidth_bound",
            severity=severity,
            limiting_factor="memory_bandwidth",
            directions=[
                "Increase BLOCK_K to amortize memory access overhead",
                "Use SPLIT_K to parallelize the reduction dimension",
                "Ensure coalesced memory access patterns (row-major preferred)",
                "Consider kernel fusion to reduce global memory round-trips",
            ],
        )

    # 5. Launch overhead
    if metrics.duration_ns < 10000:
        return BottleneckReport(
            classification="launch_overhead",
            severity=0.5,
            limiting_factor="launch",
            directions=[
                "Consider fusing with adjacent kernels",
                "Increase problem size per launch if possible",
            ],
        )

    # 6. Compute bound (the good case -- kernel is doing useful work)
    return BottleneckReport(
        classification="compute_bound",
        severity=0.2,
        limiting_factor="compute",
        directions=[
            "Increase BLOCK_N to maximize parallelism per wave",
            "Increase num_warps to 4 or 8 for more instruction-level parallelism",
            "Try num_stages=2 or 4 for software pipelining",
            "Consider algorithmic improvements (e.g., tiling, blocking strategies)",
        ],
    )
