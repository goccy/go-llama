#!/bin/bash
# bench-compare.sh — put go-llama's pp/tg throughput next to native
# llama.cpp's, on the same machine and model.
#
#   GO_LLAMA_NATIVE_REF=/path/to/llama.cpp/build/bin \
#     scripts/bench-compare.sh [model.gguf]
#
# Native numbers come from llama-bench (pp512 / tg64, one thread, to
# match the engine's single-threaded wasm build). go-llama numbers come
# from this package's benchmarks. Both are printed as one table; the
# ratio is the honest current cost of running fully in Go.
set -euo pipefail

MODEL="${1:-testdata/stories260K.gguf}"
: "${GO_LLAMA_NATIVE_REF:?set GO_LLAMA_NATIVE_REF to a native llama.cpp build/bin directory}"

echo "== native (llama-bench, 1 thread) =="
"$GO_LLAMA_NATIVE_REF/llama-bench" -m "$MODEL" -p 512 -n 64 -t 1 -r 3 2>/dev/null | grep -E 'pp512|tg64|model'

echo
echo "== go-llama (go test -bench, 1 thread) =="
GO_LLAMA_TEST_MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")" \
  go test -run '^$' -bench 'BenchmarkDecode|BenchmarkPromptEval' -benchtime 64x .
