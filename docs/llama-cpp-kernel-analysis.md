# llama.cpp HIP Kernel Analysis for Smithy

Analysis of the quantized mat-vec (MMVQ) kernel dispatch in llama.cpp,
focused on RDNA3 (7900 XTX) and how smithy can inject shape-specific configs.

Source: `~/Projects/llama-cpp-turboquant/`

---

## 1. MMVQ Kernel Architecture Overview

The MMVQ (mul_mat_vec_q) path handles quantized matrix-vector products for
batch sizes 1-8. This is the hot path during autoregressive token generation
(batch=1). The call chain is:

```
ggml_cuda_mul_mat()                          [ggml-cuda.cu:2244]
  -> ggml_cuda_mul_mat_vec_q()               [mmvq.cu:727]
    -> mul_mat_vec_q_switch_type()            [mmvq.cu:586]  (dispatches by ggml_type)
      -> mul_mat_vec_q_switch_ncols_dst<T>()  [mmvq.cu:457]  (dispatches by batch size)
        -> calc_launch_params<T>()            [mmvq.cu:417]  (computes grid/block dims)
        -> mul_mat_vec_q_switch_fusion<T>()   [mmvq.cu:428]  (launches kernel)
          -> mul_mat_vec_q<T, ncols, ...>()   [mmvq.cu:200]  (the actual GPU kernel)
```

### Entry condition (ggml-cuda.cu:2257-2259)
MMVQ is used when:
- `src0` is quantized
- `src1` is F32
- `dst` is F32
- `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE` (8)
- No bad padding from compute buffers

Priority order: mmvf > mmf > **mmvq** > mmq > cuBLAS

---

## 2. Tunable Parameters

### 2.1 nwarps (mmvq.cu:100-176)

`calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id)`

The number of warps per thread block. Each warp processes a strided portion of
the K dimension, and partial sums are reduced across warps via shared memory.

**RDNA3_0 table (7900 XTX):** (mmvq.cu:155-174)

| ncols_dst | nwarps (whitelisted types) | nwarps (other types) |
|-----------|---------------------------|---------------------|
| 1         | 8 (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, IQ4_NL) | 1 |
| 2-8       | 1                          | 1                   |

Types NOT on the RDNA3 whitelist: Q2_K, Q3_K, Q5_K, IQ4_XS, IQ2_*, IQ3_*, IQ1_*.
The comment says "Q2_K / Q5_K / IQ4_XS regress in full quant sweeps" on RDNA3.

**Critical observation:** nwarps is determined ONLY by (type, ncols_dst, arch).
It does NOT consider the matrix dimensions (N=nrows_x, K=ncols_x). A 4096x4096
and a 128x4096 matrix get the same nwarps.

### 2.2 rows_per_block (mmvq.cu:178-196)

`calc_rows_per_block(int ncols_dst, int table_id, bool small_k, int nwarps)`

For RDNA3/RDNA4, this always returns 1 (line 195). The generic/GCN tables
return higher values for multi-column cases.

For the small_k optimization (see below), rows_per_block is set to nwarps on
the generic table, but still 1 on RDNA3.

### 2.3 small_k optimization (mmvq.cu:496-521)

When K (ncols_x) is small enough that a single thread block iteration covers all
K blocks, the kernel switches to `small_k=true` mode which bumps rows_per_block
to process multiple output rows per block. Detection:

```cpp
const int blocks_per_row_x = ncols_x / qk;         // e.g., 4096/256 = 16
const int blocks_per_iter_1warp = vdr * warp_size / qi;  // e.g., 2*32/8 = 8
const bool use_small_k = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
```

For Q4_K on RDNA3 (nwarps=8, warp_size=32, vdr=2, qi=8=QI4_K):
- blocks_per_iter_1warp = 2*32/8 = 8
- Threshold: blocks_per_row_x < 8*8 = 64
- K threshold: K < 64*256 = 16384

So small_k kicks in for K < 16384, which covers most attention matrices
(head_dim * n_heads typically 4096). But since RDNA3's calc_rows_per_block
always returns 1, this optimization is a no-op on RDNA3!

### 2.4 VDR (Vector Dot Result) - per-type constant

The VDR controls how many q8_1 blocks each thread processes per iteration.
This is a compile-time constant per quant type, not tunable at dispatch time.

Key values (from vecdotq.cuh):

| Type  | VDR_MMVQ | qk  | qi  | qr |
|-------|----------|-----|-----|----|
| Q4_0  | 2        | 32  | 4   | 2  |
| Q4_K  | 2        | 256 | 8   | 2  |
| Q6_K  | 1        | 256 | 8   | 1  |
| Q3_K  | 1        | 256 | 4   | 1  |
| Q8_0  | 2        | 32  | 8   | 2  |

### 2.5 Launch configuration (mmvq.cu:416-426)

```cpp
template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(...) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}
```

Grid: (ceil(N/rpb), nchannels, nsamples)
Block: (warp_size, nwarps, 1)

Total threads per block: warp_size * nwarps = 32 * 8 = 256 on RDNA3 (for whitelisted types).

---

## 3. The Kernel Inner Loop (mmvq.cu:293-322)

```cpp
float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);
    const int kqs = vdr * (tid % (qi/vdr));

    for (int j = 0; j < ncols_dst; ++j) {
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby],
                                         kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }
}
```

Each thread computes partial dot products along K, strided by `blocks_per_iter`.
After the loop, warps reduce via shared memory (lines 324-371), then warp_reduce_sum
(line 371) collapses within each warp.

For Q4_K on RDNA3 (ncols_dst=1, nwarps=8, warp_size=32, vdr=2, qi=8):
- blocks_per_iter = 2 * 8*32 / 8 = 64
- Each thread processes every 64th block along K
- For K=4096: blocks_per_row_x = 16, so each iteration covers all K blocks
- For K=14336: blocks_per_row_x = 56, still < 64, one iteration

---

## 4. Architecture Detection

### Compile-time (common.cuh:337-343)
```cpp
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;   // GCN/CDNA
#else
    return 32;   // RDNA (all generations)
#endif
}
```

### Compile-time arch macros (common.cuh, used in mmvq.cu:70-82)
Defined per compilation unit by the build system:
`RDNA4`, `RDNA3_0`, `RDNA2`, `RDNA3_5`, `GCN`, `CDNA`

### Runtime CC values (common.cuh:63-90)
```
GGML_CUDA_CC_RDNA3   = OFFSET_AMD + 0x1100  // RX 7000
GGML_CUDA_CC_RDNA3_5 = OFFSET_AMD + 0x1150  // AI 370/395
GGML_CUDA_CC_RDNA4   = OFFSET_AMD + 0x1200  // RX 9000
```

`GGML_CUDA_CC_IS_RDNA3_0(cc)` matches 7900 XTX (gfx1100).

### mmvq parameter table (mmvq.cu:62-68)
```cpp
enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};
```

The table_id is resolved both at compile time (device function, line 70) and
runtime (host function, line 84).

---

## 5. Q4_K vec_dot Inner Loop (vecdotq.cuh:502-524)

```cpp
static float vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm4, d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    for (int i = 0; i < QR4_K; ++i) {    // QR4_K = 2
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = dp4a(v1i, u[2*i+1], dp4a(v0i, u[2*i+0], 0));
        const int dot2 = dp4a(0x01010101, u[2*i+1], dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}
```

This uses dp4a (4-element dot product with accumulate) available on RDNA2+.
Two dp4a chains per iteration, 2 iterations = 4 dp4a instructions per call.
Each call processes 8 bytes of quantized data (2 int32 values = 8 Q4 values).

---

## 6. TurboQuant Integration (turbo-quant.cuh)

TurboQuant is a KV cache compression system (turbo2/turbo3/turbo4), NOT a weight
quantization format. It implements:

- **WHT rotation**: Walsh-Hadamard transform with randomized signs (O(n log n))
- **Lloyd-Max centroids**: Optimal quantization bins for Gaussian distributions
- **InnerQ equalization**: Per-channel variance normalization before WHT
- **3-bit PolarQuant** (turbo3): 8 centroids, 2-bit low + 1-bit sign, block_size=32
- **4-bit** (turbo4): 16 centroids, 4-bit direct, block_size=128
- **2-bit** (turbo2): 4 centroids, 2-bit direct, block_size=32

TurboQuant integrates via set_rows kernels, NOT through the MMVQ dispatch.
KV cache quantized tensors go through a different path (dequant -> matmul).
This is orthogonal to the weight GEMV optimization smithy targets.

---

## 7. Key Findings for Smithy

### 7.1 No autotuning exists

There is zero autotuning or config caching in the MMVQ path. The parameter
tables are hardcoded switch statements evaluated at compile time. No runtime
profiling, no shape-dependent lookup.

### 7.2 Parameters that can be tuned per-shape

| Parameter       | Currently depends on          | Could also depend on  |
|----------------|-------------------------------|----------------------|
| nwarps          | type, ncols_dst, arch         | N (nrows_x), K (ncols_x) |
| rows_per_block  | ncols_dst, arch, small_k      | N, K |
| small_k flag    | nwarps, K, type               | (derived from above) |

The VDR is a per-type constant baked into the vec_dot function signatures.
Changing it would require multiple kernel variants per type, which is a much
larger change.

### 7.3 How to make nwarps shape-dependent

`calc_nwarps` currently takes `(type, ncols_dst, table_id)`. To make it
shape-aware, you need to:

1. **Add N and K parameters** to `calc_nwarps` signature
2. **Replace the switch statement** with a lookup table indexed by (type, N_bucket, K_bucket)
3. **Propagate N and K** through `calc_launch_params` (they're already available
   as `nrows_x` and `ncols_x` in the caller)

The tricky part: `calc_nwarps` is also used as a `constexpr` inside the kernel
itself (mmvq.cu:212) to size shared memory arrays. The kernel template already
has `ncols_dst` as a compile-time parameter but NOT N or K. Making nwarps
runtime-variable inside the kernel would require dynamic shared memory allocation
or template expansion over N/K buckets.

### 7.4 The launch_bounds problem

Line 199:
```cpp
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*warp_size, 1)
```

`__launch_bounds__` is a compile-time attribute. If nwarps becomes shape-dependent,
this attribute would need to use the maximum possible nwarps for the type, or be
removed (losing occupancy hints).

---

## 8. Recommended Approach for Shape-Specific Optimization

### Option A: External config table (minimal patch, smithy's approach)

Instead of modifying `calc_nwarps` to be shape-aware at compile time, use a
**runtime config table** that overrides the constexpr defaults:

```cpp
// In mmvq.cu, new addition:
struct mmvq_shape_config {
    int nwarps;
    int rows_per_block;
};

// Indexed by [type][N_bucket][K_bucket], generated by smithy
static mmvq_shape_config g_shape_configs[GGML_TYPE_COUNT][N_BUCKETS][K_BUCKETS];

// Modified calc_launch_params:
static std::pair<dim3, dim3> calc_launch_params(..., int nrows_x, int ncols_x) {
    auto cfg = lookup_shape_config(type, nrows_x, ncols_x, table_id);
    // Use cfg.nwarps and cfg.rows_per_block instead of calc_nwarps/calc_rows_per_block
    ...
}
```

The kernel template still uses the compile-time max nwarps for shared memory
sizing, but only `cfg.nwarps` warps actually do work. Warps beyond cfg.nwarps
early-exit. This wastes some shared memory but requires no kernel recompilation.

**Pros:** Minimal code change, no recompilation needed, smithy generates configs externally.
**Cons:** Wasted shared memory, can't reduce threads below compile-time max.

### Option B: Runtime nwarps with dynamic shared memory

Replace the constexpr shared memory sizing with dynamic shared memory:

```cpp
// Change kernel signature to accept nwarps at runtime
template <ggml_type type, int ncols_dst, int max_nwarps, ...>
__global__ void mul_mat_vec_q(...) {
    const int nwarps = blockDim.y;  // runtime from launch config
    extern __shared__ float smem[];
    // Partition smem manually
}
```

This is a larger refactor but gives full control over thread block dimensions
at launch time without kernel recompilation.

### Option C: Smithy generates a C header (recommended)

Smithy profiles all (type, N, K) combinations on the target GPU, finds optimal
(nwarps, rows_per_block) for each, and emits a C header:

```c
// Generated by smithy for gfx1100 (7900 XTX)
// smithy_mmvq_config_gfx1100.h

static constexpr mmvq_shape_config SMITHY_CONFIGS_Q4_K[] = {
    // {N_min, N_max, K_min, K_max, nwarps, rows_per_block}
    {1, 128, 1, 4096, 4, 1},      // small N, small K: fewer warps
    {1, 128, 4097, 16384, 8, 1},   // small N, large K: max warps
    {129, 4096, 1, 4096, 2, 2},    // large N, small K: fewer warps, more rows
    {129, 4096, 4097, 16384, 4, 1},
    // ... etc
};
```

Then a small patch to llama.cpp's `calc_launch_params` to consult this table.

**Pros:** Full control, smithy's core value proposition, clean integration.
**Cons:** Requires recompilation to apply new configs.

---

## 9. Minimal Patch to Make llama.cpp RDNA3 MMVQ Shape-Aware

The simplest useful patch touches only `mmvq.cu`:

### Step 1: Add shape config lookup (host-side only)

```cpp
// After line 196 in mmvq.cu:

struct smithy_override {
    int nwarps;
    int rows_per_block;
};

// Returns {0,0} if no override exists
static smithy_override lookup_smithy_config(
    ggml_type type, int nrows_x, int ncols_x, mmvq_parameter_table_id table_id);
```

### Step 2: Modify calc_launch_params

```cpp
template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int ncols_x,
        const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id,
        const bool small_k = false) {

    int nwarps = calc_nwarps(type, ncols_dst, table_id);
    int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);

    // Smithy override: only constrain nwarps downward (can't exceed template max)
    auto override = lookup_smithy_config(type, nrows_x, ncols_x, table_id);
    if (override.nwarps > 0) {
        nwarps = min(override.nwarps, nwarps);  // never exceed compiled max
    }
    if (override.rows_per_block > 0) {
        rpb = override.rows_per_block;
    }

    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}
```

### Step 3: Early warp exit in kernel

The kernel already handles variable warp participation through shared memory
reduction (lines 332-349). Warps with threadIdx.y >= actual_nwarps will contribute
zero to the sum, which is harmless but wastes cycles. A cleaner approach:

```cpp
// At the top of mul_mat_vec_q kernel, after computing constexpr nwarps:
// The blockDim.y from launch config may be < constexpr nwarps
if (threadIdx.y >= blockDim.y) return;  // early exit for unused warps
```

Wait -- this isn't needed. `calc_launch_params` sets `block_dims.y = nwarps`, so
the kernel only launches the warps we want. The constexpr nwarps in the kernel
only affects shared memory sizing, which will be oversized but functional. The
kernel reads `nwarps-1` slots from shared memory (line 363), but those extra
slots just contain zeros from initialization.

Actually, there's a subtlety: the constexpr `nwarps` in the kernel (line 212)
controls the loop bound at line 363. If the launched nwarps < constexpr nwarps,
warp 0 will read uninitialized shared memory from the missing warps. This needs
a fix:

```cpp
// Line 363: change nwarps-1 to blockDim.y-1
for (int l = 0; l < int(blockDim.y)-1; ++l) {
```

Or, more conservatively, zero-init the shared memory at kernel start.

### Summary: 3 files touched, ~50 lines changed

1. `mmvq.cu`: Add `smithy_override` struct, `lookup_smithy_config()`, modify
   `calc_launch_params` signature to include ncols_x, guard the shared memory
   reduction loop
2. A new `smithy-config.h` header: Generated config tables, included by mmvq.cu
3. Callers of `calc_launch_params` in mmvq.cu: Pass ncols_x through

---

## 10. Key Constants Reference

| Constant | Value | Source |
|----------|-------|--------|
| QK_K | 256 | ggml-common.h:89 |
| QI4_K | 32 | ggml-common.h:126 (QK_K / (4*QR4_K) = 256/8) |
| QR4_K | 2 | ggml-common.h:127 |
| VDR_Q4_K_Q8_1_MMVQ | 2 | vecdotq.cuh:498 |
| MMVQ_MAX_BATCH_SIZE | 8 | mmvq.cuh:3 |
| MMVQ_MMID_MAX_BATCH_SIZE | 4 | mmvq.cuh:4 |
| WARP_SIZE (RDNA3) | 32 | common.cuh:43, 341 |
| WARP_SIZE (GCN/CDNA) | 64 | common.cuh:339 |
| GGML_CUDA_CC_RDNA3 | OFFSET+0x1100 | common.cuh:74 |

---

## 11. Shapes That Matter (for a 7B Q4_K model)

Typical tensor shapes in Qwen3-7B / Mistral-7B class models:

| Layer | N (nrows) | K (ncols) | Notes |
|-------|-----------|-----------|-------|
| embed/lm_head | 32000-152000 | 4096 | Vocab projection, very tall |
| q_proj/k_proj/v_proj | 4096 | 4096 | Square-ish |
| o_proj | 4096 | 4096 | Square |
| gate_proj/up_proj | 14336 | 4096 | FFN, tall |
| down_proj | 4096 | 14336 | FFN, wide |
| GQA k/v | 1024 | 4096 | Grouped query attention |

For MMVQ (batch=1), each of these is a single row of src1 (K elements)
multiplied against N rows of the weight matrix. The kernel launches N/rpb blocks,
each with nwarps*warp_size threads, iterating over K/qk blocks.

The nwarps=8 default may be suboptimal for:
- **Small N (GQA k/v, 1024 rows):** Plenty of blocks for occupancy, more warps
  just add reduction overhead
- **Large K (down_proj, 14336):** More warps help divide the work
- **Huge N (vocab, 32000+):** Block count is already massive, nwarps mainly
  affects per-block throughput vs memory bandwidth

Smithy's value: profile each (N, K) shape and find the optimal nwarps.
