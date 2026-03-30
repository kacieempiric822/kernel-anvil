"""Guided config sweep based on bottleneck classification."""
from itertools import product

from kernel_anvil.analyze import BottleneckReport


def generate_configs(
    report: BottleneckReport,
    baseline_config: dict | None = None,
    max_configs: int = 20,
) -> list[dict]:
    """Generate Triton configs guided by bottleneck classification.

    Each config is a dict like:
    {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2}

    Returns an empty list for launch_overhead (kernel should be fused, not tuned).
    If baseline_config is provided, it is included in the output for comparison.
    """
    param_space = _get_param_space(report.classification)

    if param_space is None:
        # launch_overhead -- nothing to sweep
        return []

    # Generate cartesian product of all parameter values
    keys = list(param_space.keys())
    values = list(param_space.values())
    configs: list[dict] = []

    for combo in product(*values):
        cfg = dict(zip(keys, combo))
        configs.append(cfg)

    # Include baseline if provided and not already present
    if baseline_config is not None:
        if baseline_config not in configs:
            configs.insert(0, baseline_config)

    # Cap at max_configs (keep baseline at front if present)
    if len(configs) > max_configs:
        configs = configs[:max_configs]

    return configs


def _get_param_space(classification: str) -> dict[str, list] | None:
    """Return parameter search space for a given bottleneck classification."""
    spaces = {
        "bandwidth_bound": {
            "BLOCK_K": [128, 256, 512],
            "SPLIT_K": [2, 4, 8],
            "BLOCK_N": [64, 128],
            "num_warps": [4, 8],
        },
        "occupancy_limited_vgpr": {
            "BLOCK_N": [32, 64],
            "num_warps": [1, 2, 4],
            "num_stages": [1, 2],
            "BLOCK_K": [64, 128, 256],
        },
        "occupancy_limited_lds": {
            "BLOCK_N": [32, 64],
            "num_stages": [1],
            "BLOCK_K": [64, 128],
        },
        "register_spill": {
            "BLOCK_N": [32],
            "BLOCK_K": [32, 64, 128],
            "num_warps": [1, 2],
            "num_stages": [1],
        },
        "compute_bound": {
            "BLOCK_N": [128, 256],
            "BLOCK_K": [128, 256],
            "num_warps": [4, 8],
            "num_stages": [2, 4],
        },
        "launch_overhead": None,
    }
    return spaces.get(classification)
