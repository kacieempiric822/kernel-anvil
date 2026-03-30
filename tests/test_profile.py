"""Tests for kernel profiling."""
import pytest
import torch

from kernel_anvil.analyze import ProfileMetrics
from kernel_anvil.profile import _estimate_lds, _estimate_vgpr, profile_kernel
from kernel_anvil.rdna3 import GFX1100


def _dummy_kernel(config):
    """A trivial kernel_fn that returns a small tensor."""
    return torch.ones(64)


class TestProfileReturnsValidMetrics:
    def test_returns_profile_metrics(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64, "num_warps": 4},
            warmup=1,
            runs=3,
        )
        assert isinstance(result, ProfileMetrics)

    def test_duration_is_positive(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64},
            warmup=1,
            runs=3,
        )
        assert result.duration_ns > 0

    def test_vgpr_count_is_positive(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64, "num_warps": 4},
            warmup=1,
            runs=3,
        )
        assert result.vgpr_count > 0

    def test_lds_bytes_is_positive(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64, "BLOCK_K": 128},
            warmup=1,
            runs=3,
        )
        assert result.lds_bytes > 0


class TestBandwidthCalculation:
    def test_bandwidth_from_data_bytes(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64},
            data_bytes=1_000_000_000,  # 1 GB
            warmup=1,
            runs=3,
        )
        assert result.bandwidth_gbs > 0
        # Verify formula: bandwidth = data_bytes / duration_s / 1e9
        latency_us = result.duration_ns / 1000
        duration_s = latency_us / 1e6
        expected_bw = 1_000_000_000 / duration_s / 1e9
        assert result.bandwidth_gbs == pytest.approx(expected_bw, rel=1e-6)

    def test_zero_bandwidth_without_data_bytes(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64},
            warmup=1,
            runs=3,
        )
        assert result.bandwidth_gbs == 0.0


class TestOccupancy:
    def test_occupancy_computed_with_gpu_spec(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64, "num_warps": 4},
            gpu_spec=GFX1100,
            warmup=1,
            runs=3,
        )
        assert result.occupancy_pct > 0
        assert result.limiting_factor in ("vgpr", "lds", "balanced")

    def test_occupancy_zero_without_gpu_spec(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64},
            warmup=1,
            runs=3,
        )
        assert result.occupancy_pct == 0.0
        assert result.limiting_factor == "unknown"

    def test_threads_per_wg_from_config(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"num_warps": 8},
            gpu_spec=GFX1100,
            warmup=1,
            runs=3,
        )
        assert result.threads_per_wg == 8 * 32  # warps * wave_size

    def test_default_threads_per_wg(self):
        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={},
            gpu_spec=GFX1100,
            warmup=1,
            runs=3,
        )
        assert result.threads_per_wg == 4 * 32  # default 4 warps


class TestVGPREstimation:
    def test_larger_block_n_means_more_vgprs(self):
        low = _estimate_vgpr({"BLOCK_N": 32, "num_warps": 4, "num_stages": 1})
        high = _estimate_vgpr({"BLOCK_N": 256, "num_warps": 4, "num_stages": 1})
        assert high > low

    def test_more_warps_means_more_vgprs(self):
        low = _estimate_vgpr({"BLOCK_N": 64, "num_warps": 1, "num_stages": 1})
        high = _estimate_vgpr({"BLOCK_N": 64, "num_warps": 8, "num_stages": 1})
        assert high > low

    def test_more_stages_means_more_vgprs(self):
        low = _estimate_vgpr({"BLOCK_N": 64, "num_warps": 4, "num_stages": 1})
        high = _estimate_vgpr({"BLOCK_N": 64, "num_warps": 4, "num_stages": 4})
        assert high > low

    def test_vgpr_clamped_minimum(self):
        result = _estimate_vgpr({"BLOCK_N": 1, "num_warps": 1, "num_stages": 1})
        assert result >= 16

    def test_vgpr_clamped_maximum(self):
        result = _estimate_vgpr({"BLOCK_N": 1024, "num_warps": 16, "num_stages": 8})
        assert result <= 256


class TestLDSEstimation:
    def test_lds_from_tile_size(self):
        # BLOCK_N=64, BLOCK_K=128, element_bytes=2, num_stages=1
        lds = _estimate_lds({"BLOCK_N": 64, "BLOCK_K": 128, "element_bytes": 2, "num_stages": 1})
        assert lds == 64 * 128 * 2 * 1

    def test_lds_scales_with_stages(self):
        lds1 = _estimate_lds({"BLOCK_N": 64, "BLOCK_K": 128, "num_stages": 1})
        lds4 = _estimate_lds({"BLOCK_N": 64, "BLOCK_K": 128, "num_stages": 4})
        assert lds4 == 4 * lds1

    def test_lds_default_fp16(self):
        # Default element_bytes=2 (FP16)
        lds = _estimate_lds({"BLOCK_N": 64, "BLOCK_K": 128})
        assert lds == 64 * 128 * 2


class TestProfileForClassify:
    """Test that profile output is compatible with analyze.classify()."""

    def test_profile_output_classifiable(self):
        from kernel_anvil.analyze import classify

        result = profile_kernel(
            kernel_fn=_dummy_kernel,
            config={"BLOCK_N": 64, "num_warps": 4, "BLOCK_K": 128},
            gpu_spec=GFX1100,
            warmup=1,
            runs=3,
        )
        report = classify(result, GFX1100)
        assert report.classification in (
            "register_spill",
            "occupancy_limited_vgpr",
            "occupancy_limited_lds",
            "bandwidth_bound",
            "launch_overhead",
            "compute_bound",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm GPU available")
class TestGPUProfile:
    def test_gpu_profile(self):
        def fn(config):
            return torch.ones(1024, device="cuda")

        result = profile_kernel(
            kernel_fn=fn,
            config={"BLOCK_N": 64, "num_warps": 4},
            gpu_spec=GFX1100,
            data_bytes=1024 * 4,
            warmup=2,
            runs=5,
        )
        assert result.duration_ns > 0
        assert result.bandwidth_gbs > 0
        assert result.occupancy_pct > 0
