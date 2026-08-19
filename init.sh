#!/usr/bin/env bash
# init.sh — nanoserve environment check + build + test.
# Run this at the start of every coding session. Exits non-zero if the tree is unhealthy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== nanoserve init.sh ==="

# 1. Toolchain
if command -v cmake &>/dev/null; then
    CMAKE=cmake
elif [ -x /opt/homebrew/bin/cmake ]; then
    CMAKE=/opt/homebrew/bin/cmake
else
    echo "ERROR: cmake not found. Install with: brew install cmake"
    exit 1
fi
echo "cmake:    $($CMAKE --version | head -1)"

if ! command -v c++ &>/dev/null && ! command -v clang++ &>/dev/null; then
    echo "ERROR: no C++ compiler found. Install Xcode command line tools."
    exit 1
fi
echo "compiler: $(c++ --version | head -1)"

# 2. Model weights (informational — tests skip cleanly without them)
MODEL_DIR="models/qwen2.5-0.5b-instruct"
if [ -s "$MODEL_DIR/model.safetensors" ]; then
    echo "model:    present ($MODEL_DIR, $(du -h "$MODEL_DIR/model.safetensors" | cut -f1))"
else
    echo "model:    ABSENT — run scripts/download_model.sh (model-dependent tests will skip)"
fi

# 4. Configure + build (warnings-as-errors is enforced by CMakeLists)
echo ""
echo "--- configure ---"
$CMAKE -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
echo "--- build ---"
$CMAKE --build build -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

# 5. Tests
echo "--- ctest ---"
ctest --test-dir build --output-on-failure

echo ""
echo "=== init OK: build green, tests green. ==="
