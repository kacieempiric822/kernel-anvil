#!/bin/bash
# Apply kernel-anvil runtime config patch to llama.cpp
# Usage: ./apply.sh /path/to/llama.cpp

set -e

LLAMA_CPP="${1:?Usage: $0 /path/to/llama.cpp}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$LLAMA_CPP/ggml/src/ggml-cuda/mmvq.cu" ]; then
    echo "Error: $LLAMA_CPP doesn't look like a llama.cpp source tree"
    echo "Expected: ggml/src/ggml-cuda/mmvq.cu"
    exit 1
fi

# Copy smithy-config.h
cp "$SCRIPT_DIR/smithy-config.h" "$LLAMA_CPP/ggml/src/ggml-cuda/smithy-config.h"
echo "Copied smithy-config.h"

# Apply mmvq.cu patch
cd "$LLAMA_CPP"
if git apply --check "$SCRIPT_DIR/mmvq-smithy.patch" 2>/dev/null; then
    git apply "$SCRIPT_DIR/mmvq-smithy.patch"
    echo "Applied mmvq-smithy.patch"
else
    echo "Patch doesn't apply cleanly (llama.cpp version mismatch)."
    echo "Manual steps:"
    echo "  1. smithy-config.h is already copied"
    echo "  2. Add '#include \"smithy-config.h\"' to the top of ggml/src/ggml-cuda/mmvq.cu"
    echo "  3. See patches/README.md for the two small code changes needed"
    exit 1
fi

echo ""
echo "Done! Now rebuild llama.cpp with HIP:"
echo "  cd $LLAMA_CPP"
echo "  cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build --config Release -j\$(nproc)"
echo ""
echo "Then run with kernel-anvil configs:"
echo "  kernel-anvil gguf-optimize model.gguf"
echo "  SMITHY_CONFIG=~/.cache/smithy/model.json ./build/bin/llama-server -m model.gguf -ngl 999"
