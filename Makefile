# Version of goccy/llama-wasm whose release assets this package is built from.
# Bump it, run `make llama`, and commit the regenerated bridge.
LLAMA_WASM_REPO     ?= goccy/llama-wasm
LLAMA_WASM_VERSION  ?= v0.3.2
LLAMA_WASM_WORKFLOW ?= goccy/llama-wasm/.github/workflows/release.yml

# The generated wasm2go bridge. It is a release artifact of llama-wasm, not
# hand-written code, so it is downloaded and attestation-verified rather than
# edited here.
BRIDGE_ASSET := llama_wasm2go.go
BRIDGE_FILE  := internal/llama.go

RELEASE_URL     = https://github.com/$(LLAMA_WASM_REPO)/releases/download/$(LLAMA_WASM_VERSION)
ATTESTATION_API = https://api.github.com/repos/$(LLAMA_WASM_REPO)/attestations

# Tiny model the tests run against by default. Small enough to commit, big
# enough to exercise every path.
TEST_MODEL_URL := https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
TEST_MODEL     := testdata/stories260K.gguf

.PHONY: llama download verify testdata testdata-wikitext toolchain-check test test-arm64-native verify-native help

llama: download verify

download:
	mkdir -p $(dir $(BRIDGE_FILE))
	curl -fSL --proto '=https' --tlsv1.2 -o $(BRIDGE_FILE) $(RELEASE_URL)/$(BRIDGE_ASSET)

verify:
	@set -eu; \
	digest=$$(shasum -a 256 $(BRIDGE_FILE) | awk '{print $$1}'); \
	tmpdir=$$(mktemp -d); trap 'rm -rf $$tmpdir' EXIT; \
	curl -fsSL --proto '=https' --tlsv1.2 "$(ATTESTATION_API)/sha256:$$digest" \
	  | jq -c '.attestations[].bundle' > $$tmpdir/bundle.jsonl; \
	GH_TOKEN= GITHUB_TOKEN= gh attestation verify $(BRIDGE_FILE) -R $(LLAMA_WASM_REPO) \
	  --bundle $$tmpdir/bundle.jsonl --signer-workflow $(LLAMA_WASM_WORKFLOW)

testdata: $(TEST_MODEL)

# wikitext-2 (test split) for TestWikitextPerplexityParity; the archive
# is the one llama.cpp's own perplexity scripts fetch.
WIKITEXT_URL := https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
WIKITEXT     := testdata/wiki.test.raw

testdata-wikitext: $(WIKITEXT)

$(WIKITEXT):
	mkdir -p testdata
	curl -fSL --proto '=https' --tlsv1.2 -o testdata/wikitext-2-raw-v1.zip $(WIKITEXT_URL)
	unzip -o -q -j testdata/wikitext-2-raw-v1.zip wikitext-2-raw/wiki.test.raw -d testdata
	rm -f testdata/wikitext-2-raw-v1.zip

$(TEST_MODEL):
	mkdir -p testdata
	curl -fSL --proto '=https' --tlsv1.2 -o $@ $(TEST_MODEL_URL)

# toolchain-check refuses the silent wrong-engine run: an amd64 Go
# toolchain under Rosetta on an Apple Silicon host builds GOARCH=amd64
# at GOAMD64=v1, where the engine bundle has no asm at all — every test
# then exercises the pure-Go scalar fallback (~1000x slower; the big-
# model suite times out) instead of the arm64 engine this machine
# actually runs. That mode once burned an hour diagnosing a "hang" that
# was the wrong toolchain. Cross-building and running natively is the
# supported way on such hosts (see test-arm64-native).
toolchain-check:
	@hostarch=$$(go env GOHOSTARCH); goarch=$$(go env GOARCH); \
	echo "toolchain: $$(go version) GOARCH=$$goarch GOAMD64=$$(go env GOAMD64) GOHOSTARCH=$$hostarch"; \
	if [ "$$(uname -s)" = Darwin ] && [ "$$hostarch" = amd64 ] \
	   && [ "$$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then \
	  echo "go-llama: amd64 Go under Rosetta on an Apple Silicon host — tests would run the pure-Go" >&2; \
	  echo "go-llama: GOAMD64=v1 fallback, not the arm64 engine. Install a native arm64 Go, or run" >&2; \
	  echo "go-llama: 'make test-arm64-native' (cross-builds the test binaries and runs them natively)." >&2; \
	  exit 1; \
	fi

test: testdata toolchain-check
	go test ./...

# test-arm64-native: for an amd64 Go toolchain on an Apple Silicon host —
# `go test` will not execute a cross-GOARCH binary, so build the test
# binaries for arm64 and run them through the arch(1) shim.
test-arm64-native: testdata
	@set -eu; dir=$$(mktemp -d); trap 'rm -rf $$dir' EXIT; \
	for pkg in . ./internal; do \
	  name=$$(basename $$(cd $$pkg && pwd)); \
	  GOARCH=arm64 go test -c -o $$dir/$$name.test $$pkg; \
	  ( cd $$pkg && /usr/bin/arch -arm64 $$dir/$$name.test -test.timeout 20m ); \
	done

# Verify against a NATIVE llama.cpp build of the same commit: exact
# tokenizer IDs, greedy equality on the F32 model, metadata equality,
# and the quantized-divergence report. See verify_native_test.go.
verify-native: testdata
	@test -n "$(GO_LLAMA_NATIVE_REF)" || { echo "set GO_LLAMA_NATIVE_REF to a native llama.cpp build/bin directory"; exit 1; }
	GO_LLAMA_NATIVE_REF=$(GO_LLAMA_NATIVE_REF) go test -run Native -v ./...

help:
	@echo 'Targets:'
	@echo '  llama      Download + verify the generated bridge from $(LLAMA_WASM_REPO) $(LLAMA_WASM_VERSION)'
	@echo '  testdata   Fetch the tiny GGUF model the tests use'
	@echo '  test       go test ./... (fetches the model first; refuses a Rosetta amd64 toolchain)'
	@echo '  test-arm64-native  cross-build the test binaries for arm64 and run them natively (Rosetta hosts)'
