"""Tests for bottleneck classification."""
from kernel_anvil.analyze import BottleneckReport, ProfileMetrics, classify
from kernel_anvil.rdna3 import GFX1100


def test_register_spill():
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=128,
        scratch_bytes=1024,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "register_spill"
    assert report.severity == 1.0
    assert report.limiting_factor == "scratch"
    assert len(report.directions) > 0


def test_register_spill_overrides_everything():
    """Register spill takes priority even with bandwidth and occupancy issues."""
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=768,
        lds_bytes=98304,
        scratch_bytes=256,
        bandwidth_gbs=900,
        occupancy_pct=10.0,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "register_spill"


def test_occupancy_limited_vgpr():
    # 512 VGPRs -> 1536/512 = 3 waves -> 30% occupancy, factor=vgpr
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=512,
        lds_bytes=0,
        scratch_bytes=0,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "occupancy_limited_vgpr"
    assert report.limiting_factor == "vgpr"
    assert report.severity > 0.0


def test_occupancy_limited_lds():
    # Use all 98304 bytes of LDS with small WG -> very low occupancy from LDS
    # vgpr_count=0 so vgpr waves = 10 (max)
    # lds: wgs_per_cu = 98304/98304 = 1, waves_per_wg = 64/32 = 2,
    #   total = 1*2 = 2, per_simd = 2/2 = 1 -> 10% occupancy, factor=lds
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=0,
        lds_bytes=98304,
        scratch_bytes=0,
        threads_per_wg=64,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "occupancy_limited_lds"
    assert report.limiting_factor == "lds"


def test_bandwidth_bound():
    # 700 GB/s on a 960 GB/s card = 72.9% -> bandwidth bound
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=128,
        lds_bytes=0,
        scratch_bytes=0,
        bandwidth_gbs=700,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "bandwidth_bound"
    assert report.limiting_factor == "memory_bandwidth"


def test_launch_overhead():
    metrics = ProfileMetrics(
        duration_ns=5000,  # 5us < 10us threshold
        vgpr_count=64,
        lds_bytes=0,
        scratch_bytes=0,
        bandwidth_gbs=100,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "launch_overhead"
    assert report.limiting_factor == "launch"


def test_compute_bound():
    # Good occupancy, moderate bandwidth, no spills, reasonable duration
    metrics = ProfileMetrics(
        duration_ns=100000,
        vgpr_count=128,
        lds_bytes=0,
        scratch_bytes=0,
        bandwidth_gbs=200,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "compute_bound"
    assert report.limiting_factor == "compute"


def test_severity_scales_with_occupancy():
    # Very low occupancy -> high severity
    metrics_low = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=768,  # 1536/768 = 2 waves -> 20%
        lds_bytes=0,
        scratch_bytes=0,
        threads_per_wg=256,
    )
    report_low = classify(metrics_low, GFX1100)

    # Moderate occupancy -> lower severity
    metrics_mod = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=384,  # 1536/384 = 4 waves -> 40%
        lds_bytes=0,
        scratch_bytes=0,
        threads_per_wg=256,
    )
    report_mod = classify(metrics_mod, GFX1100)

    assert report_low.severity > report_mod.severity


def test_directions_are_strings():
    metrics = ProfileMetrics(duration_ns=50000, vgpr_count=128, bandwidth_gbs=700)
    report = classify(metrics, GFX1100)
    for d in report.directions:
        assert isinstance(d, str)
        assert len(d) > 0


def test_bandwidth_bound_threshold_exactly_60pct():
    # Exactly at 60% -> 576 GB/s on 960 -> 0.6 exactly
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=128,
        lds_bytes=0,
        scratch_bytes=0,
        bandwidth_gbs=576,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    # 576/960 = 0.6 exactly -- not > 0.6, so should NOT be bandwidth_bound
    assert report.classification != "bandwidth_bound"


def test_bandwidth_bound_just_above_threshold():
    metrics = ProfileMetrics(
        duration_ns=50000,
        vgpr_count=128,
        lds_bytes=0,
        scratch_bytes=0,
        bandwidth_gbs=577,
        threads_per_wg=256,
    )
    report = classify(metrics, GFX1100)
    assert report.classification == "bandwidth_bound"
