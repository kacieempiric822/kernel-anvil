"""Simple FP16 GEMV runner for kernel_anvil.

Demonstrates the runner convention. A straightforward matrix-vector multiply
using a Triton kernel with configurable BLOCK_N and BLOCK_K.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    x_ptr, w_ptr, out_ptr,
    N, K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = W @ x where W is [N, K] and x is [K]."""
    row = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = row < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        x_vals = tl.load(x_ptr + k_off, mask=k_mask, other=0.0)
        w_vals = tl.load(w_ptr + row[:, None] * K + k_off[None, :],
                         mask=mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)
    tl.store(out_ptr + row, acc.to(tl.float16), mask=mask)


# Dimensions (typical for a small projection)
N, K = 2048, 4096


def setup():
    """Return input tensors as a dict."""
    return {
        "x": torch.randn(K, device="cuda", dtype=torch.float16),
        "w": torch.randn(N, K, device="cuda", dtype=torch.float16),
    }


def run(inputs, BLOCK_N=64, BLOCK_K=128, num_warps=4, **kwargs):
    """Run the kernel with given config, return output."""
    x, w = inputs["x"], inputs["w"]
    output = torch.empty(N, device=x.device, dtype=torch.float16)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    gemv_kernel[grid](x, w, output, N, K,
                      BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return output


def reference(inputs):
    """Reference implementation for correctness."""
    return (inputs["w"] @ inputs["x"]).to(torch.float16)


DATA_BYTES = N * K * 2 + K * 2 + N * 2  # W + x + y, all FP16
BASELINE_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4}
