# llama.cpp Patch for kernel-anvil Runtime Config

This patch adds runtime shape-specific MMVQ kernel configuration to llama.cpp.
When kernel-anvil generates optimized configs for your model + GPU, this patch
lets llama.cpp load them at startup.

## Quick Apply

```bash
cd kernel-anvil/patches
./apply.sh /path/to/llama.cpp
```

## Manual Apply

If the patch doesn't apply cleanly (llama.cpp version mismatch), make these changes by hand:

### 1. Copy `smithy-config.h`

Copy `patches/smithy-config.h` to `ggml/src/ggml-cuda/smithy-config.h` in your llama.cpp tree.

### 2. Edit `ggml/src/ggml-cuda/mmvq.cu`

**Add the include** (top of file, after other includes):
```cpp
#include "smithy-config.h"
```

**In `calc_rows_per_block`**, add before the final `return 1;`:
```cpp
    // RDNA: rpb=2 when kernel-anvil profile says so (via small_k trigger)
    if (small_k && ncols_dst == 1) {
        return 2;
    }
```

**In the `should_use_small_k` lambda** (inside `mul_mat_vec_q_switch_ncols_dst`),
add before `return use;`:
```cpp
        // kernel-anvil override: if smithy config says rpb>1 for this shape, force small_k
        if (!use && c_ncols_dst == 1) {
            smithy_shape_config scfg = smithy_lookup(type, nrows_x, ncols_x);
            if (scfg.rows_per_block > 1) {
                use = true;
            }
        }
```

### 3. Rebuild

```bash
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

## How It Works

The patch is ~15 lines of actual code change:

1. `smithy-config.h` loads a JSON config file at first kernel dispatch
2. For decode (batch=1), it checks if kernel-anvil profiled this shape
3. If the config says `rows_per_block > 1`, it triggers the `small_k` kernel
   variant which processes 2 rows per block instead of 1
4. This halves the number of kernel launches and improves occupancy

Without a config file, behavior is identical to stock llama.cpp.

## Config Loading

Checked in order:
1. `SMITHY_CONFIG` environment variable (explicit path)
2. `~/.cache/smithy/default.json` (fallback)

Generate configs with:
```bash
kernel-anvil gguf-optimize model.gguf
# or for full benchmarking:
kernel-anvil autoforge model.gguf --llama-cpp-path /path/to/llama.cpp
```
