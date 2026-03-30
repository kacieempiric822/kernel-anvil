"""Triton GEMV kernel for FP16 Linear replacement.

This kernel replaces nn.Linear.forward for the M=1 (single-token decode) case.
For M>1 (prefill/batched), fall back to torch.nn.functional.linear.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    M, N, K,
    stride_wn, stride_wk,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = W @ x (+ bias). W is [N, K], x is [M, K], out is [M, N].

    For decode (M=1), this is a GEMV. Each program instance handles
    BLOCK_N output rows, iterating over K in BLOCK_K chunks.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    # Accumulate dot products in FP32 for numerical stability
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        # Load input vector slice: x[0, k_start:k_start+BLOCK_K]
        x_vals = tl.load(x_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)
        # Load weight tile: W[rows, k_start:k_start+BLOCK_K]
        w_ptrs = w_ptr + rows[:, None] * stride_wn + k_off[None, :] * stride_wk
        w_vals = tl.load(
            w_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        acc += bias

    tl.store(out_ptr + rows, acc.to(tl.float16), mask=row_mask)


def triton_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: dict,
) -> torch.Tensor:
    """Launch the Triton GEMV kernel for a single input vector.

    Args:
        x: Input tensor, flattened to [K] (single vector).
        weight: Weight matrix [N, K] (PyTorch nn.Linear convention).
        bias: Optional bias vector [N].
        config: Dict with BLOCK_N, BLOCK_K, num_warps, num_stages.

    Returns:
        Output tensor [N] in FP16.
    """
    N, K = weight.shape
    out = torch.empty(N, device=x.device, dtype=torch.float16)

    BLOCK_N = config.get("BLOCK_N", 64)
    BLOCK_K = config.get("BLOCK_K", 128)
    num_warps = config.get("num_warps", 4)
    num_stages = config.get("num_stages", 1)

    grid = ((N + BLOCK_N - 1) // BLOCK_N,)

    gemv_kernel[grid](
        x, weight, out,
        bias if bias is not None else x,  # dummy pointer when no bias
        1, N, K,
        weight.stride(0), weight.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out
