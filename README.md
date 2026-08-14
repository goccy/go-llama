# go-llama

[![Go Reference](https://pkg.go.dev/badge/github.com/goccy/go-llama.svg)](https://pkg.go.dev/github.com/goccy/go-llama)
[![CI](https://github.com/goccy/go-llama/actions/workflows/ci.yml/badge.svg)](https://github.com/goccy/go-llama/actions/workflows/ci.yml)

**llama.cpp in pure Go — run GGUF models anywhere Go runs. No cgo, no shared
library, one static binary.**

The inference engine is
[llama.cpp compiled to WebAssembly](https://github.com/goccy/llama-wasm) and
then [translated to Go](https://github.com/goccy/wasm2go) — no wasm runtime is
involved at run time — built on the
[`llamawasm2go`](https://github.com/goccy/llamawasm2go) module.

```go
package main

import (
	"fmt"

	llama "github.com/goccy/go-llama"
)

func main() {
	inst, err := llama.New()
	if err != nil {
		panic(err)
	}
	defer inst.Close()

	model, err := inst.LoadModel("model.gguf")
	if err != nil {
		panic(err)
	}
	defer model.Close()

	ctx, err := model.NewContext(llama.ContextParams{NCtx: 2048})
	if err != nil {
		panic(err)
	}
	defer ctx.Close()

	res, err := ctx.Generate("Once upon a time", llama.Params{
		NPredict:    128,
		Temperature: 0.8,
		TopP:        0.95,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Text)
}
```

## Features

- **Pure Go**: works anywhere Go compiles; `CGO_ENABLED=0` friendly.
- **Independent instances**: `llama.New` builds an engine with its own linear
  memory; create several and run them concurrently and in isolation.
- **One model, many contexts**: contexts share the weights and keep their own
  KV cache, which is how to serve independent conversations from one model. One
  instance can also hold several models — what speculative decoding needs.
- **Sampling**: temperature, top-k, top-p, min-p, typical-p, repetition /
  presence / frequency penalties, seeds, and GBNF grammars.
- **Streaming**: `Context.Stream` calls you back with each piece of text as it
  is decoded.
- **Interruptible**: `Context.Interrupt` stops a running generation from
  another goroutine.
- **Speculative decoding**, **LoRA adapters**, **chat templates**,
  **embeddings**, **scoring**, and **state save / load**.
- **Configurable sandbox**: options on `New` scope the guest to one directory,
  hand it an in-memory filesystem, cap its memory, and capture its stdio.

```go
inst, err := llama.New(
	llama.WithPreopenDir("/srv/models"), // the only directory the guest can see
	llama.WithMaxMemory(6<<30),          // fail inside the guest, not in the host
)
```

## Instances, models and contexts

`llama.New` returns a `*Llama` — one engine instance, with its own linear
memory and C heap. It is fully independent of any other instance, so several
can run concurrently on separate goroutines.

Within an instance you load one or more models with `LoadModel`; each model
spawns contexts with `NewContext` that share its weights and keep their own KV
cache. Because a whole instance carries the engine's memory, the common shape
is one instance with as many models and contexts as you need — reach for a
second instance when you want hard isolation between them.

```go
inst, _ := llama.New()
defer inst.Close()

target, _ := inst.LoadModel("qwen2.5-3b.gguf")
draft, _ := inst.LoadModel("qwen2.5-0.5b.gguf") // same instance: speculative decoding
```

Close contexts, then models, then the instance — using any handle after its
owner is closed is a use-after-free in the engine, and closed handles are
refused.

## Streaming and interruption

The engine is a single translated module with one C stack, so the goroutine
running `Generate` is the only one that can be inside it. Streaming and
interruption reach a running generation from opposite directions around that
constraint:

```go
res, err := ctx.Stream("Once upon a time", llama.Params{NPredict: 512},
	func(piece string) { fmt.Print(piece) })
```

`Stream` calls `onPiece` once per decoded token, on the generating goroutine
itself — so there is no concurrency and nothing to drop, but the callback must
be short and must not call back into the engine. It returns the same complete
`Result` as `Generate` (the pieces concatenate to `Result.Text`; a `Params.Stop`
string is delivered as decoded and only then trimmed, so the stream can run a
few characters past the returned text). A nil `onPiece` makes `Stream` exactly
`Generate`.

`Interrupt` goes the other way: it writes one aligned word straight into linear
memory (never calling into the engine), which the generation loop reads once per
token. It is safe to call from any goroutine while a generation runs; `Generate`
then returns what it has with `Reason == StopInterrupted`.

## Memory

wasm32 caps linear memory at 4 GiB, and the model weights plus every context's
KV cache live inside it. Target quantized models comfortably under that —
roughly 3B parameters at Q4 — and size `ContextParams.NCtx` accordingly.
`WithMaxMemory` caps growth so an oversized model fails in the guest instead of
growing the host process, and `WithMemoryReserve` reserves up front so a large
load does not repeatedly grow and copy.

## Performance

The generated Go is compiled by the Go compiler, and on amd64/arm64 most of it
ships as assembly derived from that compilation. The SIMD kernels ggml relies
on are native NEON on arm64 and SSE on amd64 (the latter under `GOAMD64=v2` or
higher — set it, or the vector helpers fall back to scalar Go). arm64 is the
flagship target, where the dot-product kernels lower to SDOT/SMMLA.

## Supply-chain verification

`internal/llama.go` in this repository is a release artifact of
[llama-wasm](https://github.com/goccy/llama-wasm), not hand-written code. It is
refreshed with:

```sh
make llama LLAMA_WASM_VERSION=v0.1.0
```

which downloads it and verifies its build-provenance attestation against
llama-wasm's release workflow. CI re-runs that verification (`make verify`) on
every push.

## Testing

```sh
make test        # fetches a tiny GGUF into testdata/ and runs the suite
```

Point the suite at your own model with `GO_LLAMA_TEST_MODEL=/path/to.gguf`.

## License

MIT (see LICENSE). llama.cpp is MIT; the embedded engine is a derivative work
of it.
