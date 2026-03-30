"""Qwen3-8B decode GEMV runner for kernel_anvil.

Exercises the dominant kernel shapes from Qwen3-8B at decode time (batch=1).
Uses FP16 weights (Q4 dequant would be a separate kernel -- this tests the
GEMV shape optimization at Qwen3-8B's actual dimensions).

Shapes tested (configurable via SHAPE env var):
  gate_proj: x[4096] @ W[12288, 4096] -> y[12288]  (largest FFN projection)
  q_proj:    x[4096] @ W[4096, 4096]  -> y[4096]   (attention Q)
  kv_proj:   x[4096] @ W[1024, 4096]  -> y[1024]   (attention K or V, GQA)
  down_proj: x[12288] @ W[4096, 12288] -> y[4096]  (FFN down)
"""

import os
import torch
import triton
import triton.language as tl

# Select shape via env var, default to gate_proj (the biggest FFN kernel)
SHAPES = {
    "gate_proj": (12288, 4096),
    "q_proj": (4096, 4096),
    "kv_proj": (1024, 4096),
    "down_proj": (4096, 12288),
}
SHAPE_NAME = os.environ.get("SHAPE", "gate_proj")
N, K = SHAPES[SHAPE_NAME]


@triton.jit
def gemv_kernel(
    x_ptr, w_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = W @ x where W is [N, K] and x is [K]."""
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        x_vals = tl.load(x_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)

        w_ptrs = w_ptr + rows[:, None] * stride_wn + k_off[None, :] * stride_wk
        w_vals = tl.load(w_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    tl.store(out_ptr + rows, acc.to(tl.float16), mask=row_mask)


def setup():
    """Create input tensors matching Qwen3-8B dimensions."""
    return {
        "x": torch.randn(K, device="cuda", dtype=torch.float16),
        "w": torch.randn(N, K, device="cuda", dtype=torch.float16),
    }


def run(inputs, BLOCK_N=128, BLOCK_K=256, num_warps=4, num_stages=2, **kwargs):
    """Run the GEMV kernel with given config."""
    x, w = inputs["x"], inputs["w"]
    output = torch.empty(N, device=x.device, dtype=torch.float16)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    gemv_kernel[grid](
        x, w, output,
        N, K,
        w.stride(0), w.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output


def reference(inputs):
    """PyTorch reference implementation."""
    return (inputs["w"] @ inputs["x"]).to(torch.float16)


DATA_BYTES = N * K * 2 + K * 2 + N * 2  # W + x + y in FP16
BASELINE_CONFIG = {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2}
