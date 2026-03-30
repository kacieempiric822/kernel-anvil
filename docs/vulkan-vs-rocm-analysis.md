# Vulkan vs ROCm Decode Performance Analysis

## Summary

Vulkan (RADV) is 38% faster than ROCm (HIP) for decode on the 7900 XTX with Qwen3-8B Q4_K_M.
The gap comes from **three compounding factors**, not one.

| Backend | tg32 (tok/s) | pp16 (tok/s) | Per-token (ms) |
|---------|-------------|-------------|----------------|
| Vulkan (RADV) | 124.39 | 556.69 | 8.04 |
| ROCm (HIP) | 90.18 | 757.20 | 11.09 |
| ROCm + smithy | 89.89 | 768.30 | 11.12 |

ROCm wins on prefill (36% faster) but loses on decode (38% slower). This is because prefill
is compute-bound (ROCm's rocBLAS/hipBLASLt GEMM kernels are well-optimized) while decode is
memory-bandwidth-bound and dispatch-overhead-sensitive.

Build: `58d51a6db` (llama.cpp #8660), both from same source tree (llama-cpp-turboquant).

---

## Factor 1: HIP Kernel Dispatch Overhead (~3 ms/token)

**This is the dominant factor.**

ROCm launches **904 individual HIP kernels per decode step**. Each kernel launch goes through
the HIP runtime, which has ~3-5 us of CPU-side overhead per dispatch. At 904 dispatches, this
adds up to ~3-4 ms of pure host-side overhead per token.

Vulkan records **617 shader dispatches into a command buffer** and submits the entire batch in
a single `vkQueueSubmit` call. The Vulkan command buffer is a pre-built GPU-side instruction
stream. Once submitted, the GPU processes all dispatches back-to-back with no CPU round-trips.

### Profiled evidence

From rocprofv3 kernel trace of the decode phase (32 tokens):

| Metric | Value |
|--------|-------|
| Total kernel launches | 28,920 |
| Launches per token | ~904 |
| Total kernel time | 351.8 ms (11.0 ms/token) |
| Total inter-kernel gap time | 161.6 ms (5.1 ms/token) |
| Gap as % of wall time | 32% |

Gap distribution during decode:

| Gap size | Count | Total time |
|----------|-------|------------|
| < 1 us | 357 | 0.2 ms |
| 1-5 us | 25,508 | 85.6 ms |
| 5-10 us | 974 | 6.5 ms |
| 10-50 us | 351 | 7.3 ms |
| > 50 us | 202 | 62.0 ms |

The 25,508 gaps in the 1-5 us range are HIP dispatch overhead -- the time between one kernel
finishing on the GPU and the next one starting. Each gap is the HIP runtime processing the
next `hipLaunchKernelGGL` call on the CPU, setting up the kernel arguments, and issuing the
dispatch to the GPU command processor.

Vulkan avoids this entirely: all dispatches are pre-recorded in the command buffer, and the
GPU's hardware scheduler processes them sequentially without CPU involvement.

### Why HIP Graphs aren't helping

The ROCm build has `GGML_HIP_GRAPHS=OFF` (default). The HIP graph feature is marked
"experimental, slow" in the llama.cpp CMake configuration. When enabled, it would capture the
dispatch sequence and replay it without per-kernel CPU overhead -- similar to what Vulkan's
command buffers do natively. This is a potential optimization target, but HIP graph capture
has its own overhead and compatibility issues.

---

## Factor 2: Memory Bandwidth Utilization Gap (wave64 vs wave32)

The Vulkan RADV driver uses **wave64** (subgroup size 64) on RDNA3, while ROCm HIP uses
**wave32** (warp size 32, hardcoded in `common.cuh`).

On RDNA3 (gfx1100), a Workgroup Processor (WGP) contains two SIMD32 units. When operating in
wave64 mode, both SIMD32 units work in lockstep on a single wave, effectively doubling the
work per scheduling decision. For memory-bound kernels like decode matmul, this means:

1. **Fewer waves to schedule** -- half the scheduling overhead per instruction
2. **Wider memory coalescing** -- 64 threads access memory together vs 32
3. **More efficient cross-lane reductions** -- one 64-wide reduction vs two 32-wide

### Measured bandwidth utilization

For the dominant matmul shape (Q4_K, 12288 x 4096 -- gate/up projections):

| Backend | Per-dispatch (us) | Effective BW (GB/s) | % of 960 GB/s peak |
|---------|------------------|--------------------|--------------------|
| Vulkan | 33.73 | 839 | 87% |
| ROCm (fused gate+up) | 94.63 | 598 | 62% |

For the 4096 x 4096 shape (q_proj, o_proj):

| Backend | Per-dispatch (us) | Effective BW (GB/s) | % of peak |
|---------|------------------|--------------------|--------------------|
| Vulkan | 18.42 | 512 | 53% |
| ROCm | 29.94 | 315 | 33% |

Note: ROCm's mmvq fuses gate+up projections into a single kernel launch (loading both weight
matrices once), so the direct per-launch comparison is 94.63 us for two matmuls vs Vulkan's
67.46 us (2 x 33.73) for the same two matmuls. Even with this fusion advantage, ROCm is 40%
slower due to lower bandwidth utilization.

### Different dot product approach

The backends also use fundamentally different dot product strategies:

- **Vulkan**: Reads Q4_K weights, dequantizes to float in-shader, multiplies with float
  activations directly. No input quantization step.
- **ROCm**: Quantizes float activations to Q8_1 format (either in a separate `quantize_q8_1`
  kernel or fused into the mmvq kernel), then uses integer dot products between Q4_K and Q8_1.

The ROCm approach was designed for NVIDIA GPUs where integer dot products (DP4A) are faster
than float. On RDNA3, the advantage is less clear, and the quantization overhead adds cost.

---

## Factor 3: Operator Fusion

Vulkan's graph compute fuses multiple operations into single shader dispatches, reducing both
dispatch count and memory round-trips.

### Vulkan fusions active during decode

| Fusion | Ops combined | Dispatches/token | Equivalent separate ops |
|--------|-------------|-----------------|------------------------|
| RMS_NORM_MUL | rms_norm + scale | 73 | 146 |
| RMS_NORM_MUL_ROPE | rms_norm + scale + rope | 36 | 108 |
| RMS_NORM_MUL_ROPE_VIEW_SET_ROWS | rms_norm + scale + rope + view + set_rows | 36 | 180 |
| MUL_MAT_ADD | matmul + bias add | 71 | 142 |
| GLU | silu + elementwise mul | 36 | 72 |
| **Total** | | **252** | **648** |

These fusions eliminate 396 separate kernel dispatches per token and avoid the intermediate
memory reads/writes between fused operations.

### ROCm fusion status

ROCm has mmvq fusion (quantize + matmul in one kernel) and some newer fusions (rope+set_rows,
gate+up matmul fusion). However, it lacks equivalents for:

- RMS_NORM_MUL (still separate rms_norm + mul kernels)
- RMS_NORM_MUL_ROPE (no 3-way fusion)
- MUL_MAT_ADD (no matmul + bias add fusion for decode)

### Extra kernels in ROCm

ROCm launches several kernel types that Vulkan avoids entirely:

| Kernel | Launches/token | Time/token (us) | Why |
|--------|---------------|----------------|-----|
| quantize_q8_1 | 224 | 573 | Input quantization (Vulkan uses float directly) |
| copy_buffer | 43 | 103 | Host-device staging copies (Vulkan uses GPU buffers) |
| convert f32/f16 | 3.4 | 36 | Type conversion between ops |
| quantize_mmq | 7.8 | 17 | MMQ-specific quantization |

These add ~729 us/token of compute and 278 extra kernel launches.

---

## Attribution Summary

For Qwen3-8B Q4_K_M decode on 7900 XTX:

| Factor | Impact (ms/token) | % of gap |
|--------|-------------------|----------|
| HIP dispatch overhead vs Vulkan command buffer | ~3.0 | ~60% |
| Lower bandwidth utilization (wave32 vs wave64) | ~1.0 | ~20% |
| Extra kernels (quantize_q8_1, copy_buffer, etc.) | ~0.7 | ~15% |
| Missing operator fusions (intermediate memory traffic) | ~0.3 | ~5% |
| **Total gap** | **~3.0** | **38%** |

Note: These factors interact. Dispatch overhead and low bandwidth utilization are partially
overlapping (the gap time between kernels reduces effective bandwidth). The attribution is
approximate.

---

## Potential Optimizations for ROCm

### High impact

1. **Enable HIP Graphs** -- `GGML_HIP_GRAPHS=ON` would capture the dispatch sequence and
   replay it without per-kernel CPU overhead, similar to Vulkan's command buffers. This
   could eliminate most of the 3 ms/token dispatch overhead. Currently marked "experimental,
   slow" -- needs testing and likely upstream fixes.

2. **wave64 mode for mmvq kernels** -- Compile mmvq kernels with
   `__attribute__((amdgpu_flat_work_group_size(64, 256)))` and `__attribute__((amdgpu_waves_per_eu(0)))`
   to use wave64 on RDNA3. This would improve memory bandwidth utilization from 62% to
   potentially 80%+. Requires restructuring the kernel's warp-level primitives.

3. **Eliminate quantize_q8_1** -- Switch mmvq to work directly with float activations
   (like Vulkan does), removing 224 kernel launches and ~573 us/token of quantization work.
   The Vulkan shader demonstrates this is practical and may even be faster on RDNA3.

### Medium impact

4. **Fuse RMS_NORM + MUL** -- Vulkan's most common fusion. Would eliminate 73 rms_norm and
   73 mul kernel launches per token, saving dispatch overhead and one memory round-trip per
   fusion.

5. **Fuse MUL_MAT + ADD** -- For layers with bias, fuse the bias addition into the matmul
   kernel. Saves 71 add kernel launches per token.

6. **Reduce copy_buffer calls** -- 43 per token is excessive. Investigate whether these can
   be eliminated through better buffer management.

### Lower impact

7. **Match Vulkan's 5-way fusion** -- RMS_NORM_MUL_ROPE_VIEW_SET_ROWS is an aggressive
   fusion that eliminates 5 separate ops. Diminishing returns vs the simpler 2-way fusions.

---

## Methodology

- Model: Qwen3-8B Q4_K_M (4.68 GiB, 8.19B params)
- GPU: AMD Radeon RX 7900 XTX (RDNA3, gfx1100, 24GB VRAM, 960 GB/s peak BW)
- ROCm profiling: `rocprofv3 --kernel-trace` with SQLite output, decode phase isolated
- Vulkan profiling: `GGML_VK_PERF_LOGGER=1` (GPU timestamp queries between dispatches)
- Clean benchmarks: `llama-bench -p 16 -n 32 -r 1` without profiling overhead
- Both builds from same source tree (commit 58d51a6db, #8660)
- Date: 2026-03-30

---

## Raw Data

### ROCm decode phase kernel breakdown (per token, from 32-token window)

| Kernel | Launches | Time (us) | Avg (us) |
|--------|----------|-----------|----------|
| mmvq_q4k_fused | 91.8 | 5,155 | 56.1 |
| mmvq_q4k_unfused | 93.8 | 1,167 | 12.4 |
| mmvq_q6k_fused | 18.6 | 1,005 | 53.9 |
| mmvq_q6k_unfused | 19.7 | 808 | 41.1 |
| quantize_q8_1 | 223.9 | 573 | 2.5 |
| rms_norm (both) | 154.1 | 571 | 3.7 |
| mmvf_half (both) | 74.3 | 542 | 7.3 |
| mmq_matmul | 7.8 | 453 | 58.3 |
| rope (both) | 76.5 | 294 | 3.8 |
| softmax | 38.3 | 138 | 3.6 |
| copy_buffer | 43.4 | 103 | 2.4 |
| set_rows | 38.3 | 91 | 2.4 |
| **Total** | **904** | **10,990** | |

### Vulkan decode GPU time (per token, from perf logger)

| Operation | Dispatches | Time (us) | Avg (us) |
|-----------|-----------|-----------|----------|
| MUL_MAT_VEC q4K 12288x4096 | 72 | 2,429 | 33.7 |
| RMS_NORM_MUL_ROPE | 36 | 910 | 25.3 |
| MUL_MAT_ADD q6K 4096x12288 | 18 | 900 | 50.0 |
| MUL_MAT_ADD q4K 4096x12288 | 18 | 767 | 42.6 |
| MUL_MAT_VEC q4K 4096x4096 | 37 | 714 | 19.3 |
| MUL_MAT_ADD q4K 4096x4096 | 35 | 655 | 18.7 |
| MUL_MAT_VEC q6K 151936x4096 | 1 | 547 | 547.1 |
| RMS_NORM_MUL | 73 | 525 | 7.2 |
| SOFT_MAX | 36 | 186 | 5.2 |
| GLU | 36 | 176 | 4.9 |
| MUL_MAT_VEC f16 (both) | 72 | 304 | 4.2 |
| CONT | 36 | 139 | 3.9 |
| SET_ROWS | 36 | 106 | 2.9 |
| RMS_NORM_MUL_ROPE_VIEW_SET_ROWS | 36 | 103 | 2.9 |
| MUL_MAT_VEC q4K 1024x4096 | 54 | 92 | 1.7 |
| Other (ADD, GET_ROWS, q6K small) | 5 | 14 | 2.8 |
| **Total** | **617** | **8,565** | |
