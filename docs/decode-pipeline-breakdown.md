# Decode Pipeline Breakdown: Qwen3-8B Q4_K_M on 7900 XTX

**Date:** 2026-03-30
**Branch:** `smithy-shape-configs` @ `58d51a6db`
**Profiler:** rocprofv3 1.1.0, kernel trace mode
**Workload:** `llama-bench -m Qwen3-8B-Q4_K_M.gguf -ngl 999 -t 12 -p 16 -n 128 -r 1`
**Measured throughput:** 75.59 tok/s

## Summary

MMVQ dominates decode at 77.6% of GPU time. A 1.34x MMVQ speedup (the 940 GB/s
we measured in isolation) translates to ~25% end-to-end improvement via Amdahl's
law. But the small_k optimization that achieves 940 GB/s is **not triggering** on
this model due to an off-by-one in the threshold condition.

## Pipeline Breakdown (tg128 phase only)

| Category | Time (ms) | % of GPU | us/tok | Notes |
|---|---:|---:|---:|---|
| **MMVQ (weight matvec)** | 988.3 | 77.6% | 7721 | Q4_K + Q6_K dequant matvec |
| quantize_q8_1 | 72.3 | 5.7% | 565 | Input quantization for MMVQ |
| MMVF (attention matvec) | 69.5 | 5.5% | 543 | K*Q and V*attn (half precision) |
| rms_norm | 66.2 | 5.2% | 517 | RMSNorm + fused weight multiply |
| rope | 31.5 | 2.5% | 246 | Rotary position embedding |
| copy_buffer | 18.3 | 1.4% | 143 | Device-to-device copies |
| softmax | 15.1 | 1.2% | 118 | Attention softmax |
| set_rows (KV write) | 11.1 | 0.9% | 87 | Writing K/V to cache |
| get_rows | 1.0 | 0.1% | 8 | Embedding lookup |
| add/bcast | 0.3 | 0.0% | 3 | Residual add |
| **Total GPU** | **1273.6** | **100%** | | |
| Host/launch overhead | 459.5 | -- | 3590 | 26.5% of wall time |
| **Wall clock** | **1733.1** | -- | | |

GPU utilization: 73.5% (kernel time / wall time).

## MMVQ Sub-breakdown

| Variant | Time (ms) | % of MMVQ | % of Total | Avg (us) |
|---|---:|---:|---:|---:|
| Q4_K fused (gate+SiLU) | 621.6 | 62.9% | 48.8% | 54.1 |
| Q4_K plain (down, attn) | 145.7 | 14.7% | 11.4% | 12.4 |
| Q6_K fused | 121.4 | 12.3% | 9.5% | 52.3 |
| Q6_K plain | 99.5 | 10.1% | 7.8% | 40.6 |

All MMVQ dispatches use: `workgroup_size = 32x8` (warp_size=32, nwarps=8).
No `small_k=true` kernel variants appear in the trace.

## Critical Finding: small_k Not Triggering

The `small_k` template path (which enables `rows_per_block = nwarps`) is **never
activated** for Qwen3-8B Q4_K_M. The gate is in `mul_mat_vec_q_switch_ncols_dst`:

```cpp
const bool use_small_k = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
//                                                        ^ strict less-than
```

For Q4_K with K=4096 (the most common weight dimension in this model):
- `qk = 256`, `qi = 32`, `vdr = 2`
- `blocks_per_row_x = 4096 / 256 = 16`
- `blocks_per_iter_1warp = vdr * warp_size / qi = 2 * 32 / 32 = 2`
- `threshold = nwarps * blocks_per_iter_1warp = 8 * 2 = 16`
- `16 < 16 = false` -- **off by one**

For Q4_K with K=12288 (down_proj):
- `blocks_per_row_x = 12288 / 256 = 48`
- `48 < 16 = false` -- well above threshold, correctly excluded

The 940 GB/s benchmark result was on a synthetic small-K matrix that DID trigger
`small_k=true`. The actual model's K=4096 dimension sits exactly on the boundary.

### Fix Options

1. **Change `<` to `<=`** in the threshold condition. This activates `small_k` for
   K=4096, covering gate_proj, up_proj, q_proj, and o_proj weight matrices (the
   majority of MMVQ time). K=12288 remains unaffected.

2. **Raise the threshold** to also cover K=12288. This would require
   `blocks_per_row_x <= 2 * nwarps * blocks_per_iter_1warp` (threshold=32), which
   still excludes K=12288 (bpr=48). To include it: threshold >= 49, meaning
   ~3.06x the current value. This needs benchmarking -- rpb=8 with K=12288 may
   hurt because each warp only processes 6 blocks per iteration.

3. **Use smithy configs** to force rpb=2 for specific shapes. The smithy system
   works (verified) but needs model-specific config generation for Qwen3-8B.

## Amdahl's Law Projections

| MMVQ Speedup | MMVQ Time | Total GPU | Overall Speedup | Predicted tok/s |
|---|---:|---:|---:|---:|
| 1.34x (940/700 BW) | 737.5 ms | 1022.8 ms | 1.25x | 94.1 (+24.5%) |
| 1.5x (optimistic) | 658.9 ms | 944.1 ms | 1.35x | 102.0 (+34.9%) |
| 2.0x (theoretical) | 494.1 ms | 779.4 ms | 1.63x | 123.5 (+63.4%) |

The 940 GB/s result represents a 1.34x bandwidth improvement over stock's 700 GB/s.
If this speedup applied to all MMVQ dispatches, end-to-end decode would improve
~25% (75.6 -> 94.1 tok/s). The previous 12% (26.5 -> 29.9) measurement was likely
on a different model or config.

## Next Optimization Targets

### Tier 1: High Impact

1. **Fuse quantize_q8_1 into MMVQ** (5.7% of decode). Every MMVQ dispatch requires
   a preceding q8_1 quantization of the input vector. Fusing this into the MMVQ
   kernel eliminates 72.3 ms of standalone quantization and reduces launch overhead.
   Combined with MMVQ, this targets 83.3% of decode time.

2. **Flash attention for decode** (5.5% MMVF + 1.2% softmax = 6.7%). The decode path
   uses regular attention (separate K*Q matvec, softmax, V*attn matvec). Flash
   attention fuses these, eliminating softmax materialization and reducing memory
   traffic. ROCm composable_kernel has FA2 support.

### Tier 2: Medium Impact

3. **RMSNorm optimization** (5.2%). Already fused with weight multiply. Potential:
   vectorized loads, reduced precision intermediate computation.

4. **RoPE fusion** (2.5%). Could be fused into the attention kernel to eliminate a
   separate kernel launch per layer.

5. **Reduce host overhead** (26.5% of wall time). 459.5 ms is spent between kernel
   launches. CUDA/HIP graphs, persistent kernels, or reduced graph traversal
   overhead could help.

### Tier 3: Lower Impact

6. **Eliminate copy_buffer** (1.4%). Investigate whether device copies can be avoided
   through in-place operations or aliased tensors.

7. **set_rows optimization** (0.9%). KV cache writes are structural but could benefit
   from coalesced write patterns.

## Model Architecture Reference (Qwen3-8B)

| Parameter | Value |
|---|---|
| Layers | 36 |
| Hidden dim | 4096 |
| FFN dim | 12288 |
| Heads | 32 (GQA: 8 KV heads) |
| Head dim | 128 |
| Vocab | ~152K |

### Weight Matrix Shapes (MMVQ targets)

| Matrix | Shape (N x K) | Quant | Bucket (N,K) | small_k? |
|---|---|---|---|---|
| gate_proj | 12288 x 4096 | Q4_K | (3,2) | No (bpr=16, threshold=16) |
| up_proj | 12288 x 4096 | Q4_K | (3,2) | No |
| down_proj | 4096 x 12288 | Q4_K | (2,3) | No (bpr=48, threshold=16) |
| q_proj | 4096 x 4096 | Q6_K | (2,2) | No (bpr=16, threshold=16) |
| k_proj | 1024 x 4096 | Q6_K | (1,2) | No |
| v_proj | 1024 x 4096 | Q6_K | (1,2) | No |
| o_proj | 4096 x 4096 | Q6_K | (2,2) | No |

## Smithy Config Status

The default smithy config (`~/.cache/smithy/default.json`) was loaded (10 entries)
but was generated for `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M`.
None of its bucket combinations (3,4), (3,3), (3,0), (3,1), (4,3) match Qwen3-8B's
actual matrix dimensions which all fall in buckets 1-2 for K and 1-3 for N.
The configs had no effect on this run.

## mmvq.cu Patch Verification

1. **calc_rows_per_block**: Correctly returns `nwarps` for `small_k` shapes on RDNA3.
   The generic path at line 199-200 handles all RDNA table IDs.

2. **smithy_lookup**: Working. Lazy-loads on first call, returns `{0,0}` for missing
   entries, properly validates bucket bounds and type indices.

3. **Shared memory reduction**: Line 347 allocates `nwarps-1` slots (compile-time max).
   Line 388 iterates `actual_nwarps-1` (runtime `blockDim.y-1`). This is correct --
   when smithy reduces nwarps at launch time, the reduction loop only reads
   initialized slots.

## Reproduction

```bash
# Profile command (disable smithy to get pure stock behavior):
SMITHY_CONFIG=/dev/null rocprofv3 --kernel-trace -o /tmp/profile_results -- \
    ~/Projects/llama-cpp-turboquant/build/bin/llama-bench \
    -m ~/Models/Qwen3-8B-Q4_K_M.gguf -ngl 999 -t 12 -p 16 -n 128 -r 1

# Analyze results:
sqlite3 /tmp/profile_results_results.db "
    SELECT ks.display_name, COUNT(*), SUM(kd.end - kd.start)/1e3 as total_us
    FROM rocpd_kernel_dispatch kd
    JOIN rocpd_info_kernel_symbol ks ON kd.kernel_id = ks.id
    GROUP BY ks.display_name ORDER BY total_us DESC;"
```

Note: the profile above was run WITH smithy defaults loaded (they had no effect on
this model's dimensions). To get identical results, use `SMITHY_CONFIG=/dev/null`.
