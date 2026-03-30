"""Tests for kernel verification and benchmarking."""
import pytest
import torch

from kernel_anvil.verify import VerifyResult, verify_and_bench


def _make_correct_fn(reference: torch.Tensor):
    """Return a kernel_fn that always returns the reference tensor."""
    def fn(config):
        return reference.clone()
    return fn


def _make_wrong_fn(reference: torch.Tensor, offset: float = 1.0):
    """Return a kernel_fn that returns wrong values."""
    def fn(config):
        return reference + offset
    return fn


class TestCorrectnessPass:
    def test_correct_output_passes(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={"BLOCK_N": 64},
            warmup=1,
            runs=3,
        )
        assert result.correct is True
        assert result.max_diff == 0.0

    def test_correct_within_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        # Return values within tolerance
        def fn(config):
            return ref + 0.005  # within atol=0.01
        result = verify_and_bench(
            kernel_fn=fn,
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            atol=1e-2,
            rtol=1e-2,
        )
        assert result.correct is True
        assert result.max_diff == pytest.approx(0.005, abs=1e-6)


class TestCorrectnessFail:
    def test_wrong_output_fails(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        result = verify_and_bench(
            kernel_fn=_make_wrong_fn(ref, offset=1.0),
            reference_output=ref,
            config={"BLOCK_N": 64},
            warmup=1,
            runs=3,
        )
        assert result.correct is False
        assert result.max_diff == pytest.approx(1.0, abs=1e-6)

    def test_barely_outside_tolerance(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        def fn(config):
            return ref + 0.02  # outside atol=0.01
        result = verify_and_bench(
            kernel_fn=fn,
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            atol=1e-2,
            rtol=0.0,
        )
        assert result.correct is False


class TestLatency:
    def test_latency_is_positive(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=5,
        )
        assert result.latency_us > 0

    def test_latency_type(self):
        ref = torch.tensor([1.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
        )
        assert isinstance(result.latency_us, float)


class TestBandwidth:
    def test_bandwidth_computed_when_data_bytes_provided(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            data_bytes=1_000_000,  # 1 MB
        )
        assert result.bandwidth_gbs is not None
        assert result.bandwidth_gbs > 0

    def test_bandwidth_none_when_no_data_bytes(self):
        ref = torch.tensor([1.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
        )
        assert result.bandwidth_gbs is None

    def test_bandwidth_formula(self):
        """Verify bandwidth = data_bytes / duration_s / 1e9."""
        ref = torch.tensor([1.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            data_bytes=1_000_000_000,  # 1 GB
        )
        # Recompute expected bandwidth from measured latency
        duration_s = result.latency_us / 1e6
        expected_bw = 1_000_000_000 / duration_s / 1e9
        assert result.bandwidth_gbs == pytest.approx(expected_bw, rel=1e-6)


class TestSpeedup:
    def test_speedup_computed_when_baseline_provided(self):
        ref = torch.tensor([1.0, 2.0, 3.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            baseline_latency_us=100.0,
        )
        assert result.speedup is not None
        assert result.speedup > 0

    def test_speedup_none_when_no_baseline(self):
        ref = torch.tensor([1.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
        )
        assert result.speedup is None

    def test_speedup_formula(self):
        """Verify speedup = baseline_latency / measured_latency."""
        ref = torch.tensor([1.0])
        baseline = 200.0
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
            baseline_latency_us=baseline,
        )
        expected_speedup = baseline / result.latency_us
        assert result.speedup == pytest.approx(expected_speedup, rel=1e-6)


class TestVerifyResult:
    def test_result_contains_config(self):
        ref = torch.tensor([1.0])
        cfg = {"BLOCK_N": 128, "num_warps": 4}
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config=cfg,
            warmup=1,
            runs=3,
        )
        assert result.config == cfg

    def test_result_dataclass_fields(self):
        ref = torch.tensor([1.0])
        result = verify_and_bench(
            kernel_fn=_make_correct_fn(ref),
            reference_output=ref,
            config={},
            warmup=1,
            runs=3,
        )
        assert isinstance(result, VerifyResult)
        assert hasattr(result, "config")
        assert hasattr(result, "correct")
        assert hasattr(result, "max_diff")
        assert hasattr(result, "latency_us")
        assert hasattr(result, "bandwidth_gbs")
        assert hasattr(result, "speedup")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm GPU available")
class TestGPU:
    def test_gpu_kernel_verification(self):
        ref = torch.ones(1024, device="cuda")
        def fn(config):
            return torch.ones(1024, device="cuda")
        result = verify_and_bench(
            kernel_fn=fn,
            reference_output=ref,
            config={},
            warmup=2,
            runs=5,
        )
        assert result.correct is True
        assert result.latency_us > 0

    def test_gpu_bandwidth(self):
        size = 1024 * 1024  # 1M elements
        ref = torch.ones(size, device="cuda")
        def fn(config):
            return torch.ones(size, device="cuda")
        result = verify_and_bench(
            kernel_fn=fn,
            reference_output=ref,
            config={},
            warmup=2,
            runs=5,
            data_bytes=size * 4,  # float32
        )
        assert result.bandwidth_gbs is not None
        assert result.bandwidth_gbs > 0
