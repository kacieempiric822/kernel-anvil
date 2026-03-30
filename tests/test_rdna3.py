"""Tests for RDNA3 hardware constants and occupancy calculations."""
from kernel_anvil.rdna3 import GFX1100, GFX1101, GFX1102, GPU_SPECS, GpuSpec


def test_max_vgpr_waves_128():
    # 128 VGPRs -> ceil(128/8)*8 = 128 -> 1536/128 = 12 -> capped at 10
    assert GFX1100.max_vgpr_waves(128) == 10


def test_max_vgpr_waves_256():
    # 256 VGPRs -> 1536/256 = 6
    assert GFX1100.max_vgpr_waves(256) == 6


def test_max_vgpr_waves_zero():
    # 0 VGPRs -> max_waves_per_simd
    assert GFX1100.max_vgpr_waves(0) == 10


def test_max_vgpr_waves_granularity():
    # 100 VGPRs -> ceil(100/8)*8 = 104 -> 1536/104 = 14 -> capped at 10
    assert GFX1100.max_vgpr_waves(100) == 10
    # 200 VGPRs -> ceil(200/8)*8 = 200 -> 1536/200 = 7
    assert GFX1100.max_vgpr_waves(200) == 7


def test_max_vgpr_waves_high_pressure():
    # 512 VGPRs -> 1536/512 = 3
    assert GFX1100.max_vgpr_waves(512) == 3
    # 768 VGPRs -> 1536/768 = 2
    assert GFX1100.max_vgpr_waves(768) == 2


def test_max_lds_waves_zero():
    assert GFX1100.max_lds_waves(0, 256) == 10


def test_max_lds_waves_moderate():
    # 16384 LDS bytes, 256 threads per WG
    # wgs_per_cu = 98304 / 16384 = 6
    # waves_per_wg = 256/32 = 8
    # total_waves = 6*8 = 48
    # per_simd = 48/2 = 24 -> capped at 10
    assert GFX1100.max_lds_waves(16384, 256) == 10


def test_max_lds_waves_high_usage():
    # 49152 LDS bytes (half of 98304), 256 threads
    # wgs_per_cu = 98304/49152 = 2
    # waves_per_wg = 256/32 = 8
    # total_waves = 2*8 = 16
    # per_simd = 16/2 = 8
    assert GFX1100.max_lds_waves(49152, 256) == 8


def test_occupancy_vgpr_limited():
    pct, factor = GFX1100.occupancy(vgpr_count=256, lds_bytes=0, threads_per_wg=128)
    assert pct == 60.0
    assert factor == "vgpr"


def test_occupancy_balanced():
    # No LDS, no VGPR pressure -> both at max
    pct, factor = GFX1100.occupancy(vgpr_count=0, lds_bytes=0, threads_per_wg=256)
    assert pct == 100.0
    assert factor == "balanced"


def test_occupancy_lds_limited():
    # Very high LDS, low VGPR
    # 98304 LDS bytes = all of it, 128 threads
    # wgs_per_cu = 98304/98304 = 1
    # waves_per_wg = 128/32 = 4
    # total_waves = 1*4 = 4
    # per_simd = 4/2 = 2
    # vgpr_w with 0 vgprs = 10
    # active = min(10, 2) = 2 -> 20%
    pct, factor = GFX1100.occupancy(vgpr_count=0, lds_bytes=98304, threads_per_wg=128)
    assert pct == 20.0
    assert factor == "lds"


def test_gpu_specs_dict():
    assert "gfx1100" in GPU_SPECS
    assert "gfx1101" in GPU_SPECS
    assert "gfx1102" in GPU_SPECS
    assert GPU_SPECS["gfx1100"] is GFX1100


def test_gpu_spec_frozen():
    import pytest
    with pytest.raises(AttributeError):
        GFX1100.cu_count = 48  # type: ignore


def test_all_specs_wave_size_32():
    for spec in GPU_SPECS.values():
        assert spec.wave_size == 32


def test_gfx1102_lower_bandwidth():
    assert GFX1102.peak_bandwidth_gbs < GFX1101.peak_bandwidth_gbs < GFX1100.peak_bandwidth_gbs
