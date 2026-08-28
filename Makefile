# Version of goccy/llama-wasm whose release assets this package is built from.
# Bump it, run `make llama`, and commit the regenerated bridge.
LLAMA_WASM_REPO     ?= goccy/llama-wasm
LLAMA_WASM_VERSION  ?= v0.2.8
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

.PHONY: llama download verify testdata test verify-native help

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

$(TEST_MODEL):
	mkdir -p testdata
	curl -fSL --proto '=https' --tlsv1.2 -o $@ $(TEST_MODEL_URL)

test: testdata
	go test ./...

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
	@echo '  test       go test ./... (fetches the model first)'
