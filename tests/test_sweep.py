"""Tests for config sweep generation."""
from kernel_anvil.analyze import BottleneckReport
from kernel_anvil.sweep import generate_configs


def _make_report(classification: str) -> BottleneckReport:
    """Helper to create a minimal BottleneckReport."""
    return BottleneckReport(
        classification=classification,
        severity=0.5,
        limiting_factor="test",
        directions=["test direction"],
    )


def test_bandwidth_bound_configs():
    report = _make_report("bandwidth_bound")
    configs = generate_configs(report)
    assert len(configs) > 0
    for cfg in configs:
        assert cfg["BLOCK_K"] in [128, 256, 512]
        assert cfg["SPLIT_K"] in [2, 4, 8]
        assert cfg["BLOCK_N"] in [64, 128]
        assert cfg["num_warps"] in [4, 8]


def test_occupancy_limited_vgpr_configs():
    report = _make_report("occupancy_limited_vgpr")
    configs = generate_configs(report)
    assert len(configs) > 0
    for cfg in configs:
        assert cfg["BLOCK_N"] in [32, 64]
        assert cfg["num_warps"] in [1, 2, 4]
        assert cfg["num_stages"] in [1, 2]
        assert cfg["BLOCK_K"] in [64, 128, 256]


def test_occupancy_limited_lds_configs():
    report = _make_report("occupancy_limited_lds")
    configs = generate_configs(report)
    assert len(configs) > 0
    for cfg in configs:
        assert cfg["BLOCK_N"] in [32, 64]
        assert cfg["num_stages"] in [1]
        assert cfg["BLOCK_K"] in [64, 128]


def test_register_spill_configs():
    report = _make_report("register_spill")
    configs = generate_configs(report)
    assert len(configs) > 0
    for cfg in configs:
        assert cfg["BLOCK_N"] in [32]
        assert cfg["BLOCK_K"] in [32, 64, 128]
        assert cfg["num_warps"] in [1, 2]
        assert cfg["num_stages"] in [1]


def test_compute_bound_configs():
    report = _make_report("compute_bound")
    configs = generate_configs(report)
    assert len(configs) > 0
    for cfg in configs:
        assert cfg["BLOCK_N"] in [128, 256]
        assert cfg["BLOCK_K"] in [128, 256]
        assert cfg["num_warps"] in [4, 8]
        assert cfg["num_stages"] in [2, 4]


def test_launch_overhead_returns_empty():
    report = _make_report("launch_overhead")
    configs = generate_configs(report)
    assert configs == []


def test_baseline_included():
    report = _make_report("bandwidth_bound")
    baseline = {"BLOCK_K": 64, "SPLIT_K": 1, "BLOCK_N": 32, "num_warps": 2}
    configs = generate_configs(report, baseline_config=baseline)
    assert baseline in configs
    # Baseline should be first
    assert configs[0] == baseline


def test_baseline_not_duplicated():
    report = _make_report("register_spill")
    # This baseline matches one of the generated configs
    baseline = {"BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 1, "num_stages": 1}
    configs = generate_configs(report, baseline_config=baseline)
    count = sum(1 for c in configs if c == baseline)
    assert count == 1


def test_max_configs_respected():
    report = _make_report("bandwidth_bound")
    configs = generate_configs(report, max_configs=5)
    assert len(configs) <= 5


def test_max_configs_with_baseline():
    report = _make_report("bandwidth_bound")
    baseline = {"BLOCK_K": 64, "SPLIT_K": 1, "BLOCK_N": 32, "num_warps": 2}
    configs = generate_configs(report, baseline_config=baseline, max_configs=3)
    assert len(configs) <= 3
    assert configs[0] == baseline


def test_configs_are_dicts():
    report = _make_report("compute_bound")
    configs = generate_configs(report)
    for cfg in configs:
        assert isinstance(cfg, dict)
        for key in cfg:
            assert isinstance(key, str)
            assert isinstance(cfg[key], int)


def test_no_duplicate_configs():
    report = _make_report("bandwidth_bound")
    configs = generate_configs(report)
    # Convert to tuples of sorted items for comparison
    as_tuples = [tuple(sorted(c.items())) for c in configs]
    assert len(as_tuples) == len(set(as_tuples))
