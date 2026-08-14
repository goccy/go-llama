# go-llama

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
	model, err := llama.LoadModel("model.gguf")
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
- **One model, many contexts**: contexts share the weights and keep their own
  KV cache, which is how to serve independent conversations from one model.
- **Sampling**: temperature, top-k, top-p, min-p, typical-p, repetition /
  presence / frequency penalties, seeds, and GBNF grammars.
- **Streaming**: `Context.Stream` calls you back with each piece of text as it
  is decoded.
- **Interruptible**: `Context.Interrupt` stops a running generation from
  another goroutine.
- **Chat templates** and **embeddings**.
- **Configurable sandbox**: `Init` can scope the guest to one directory, hand
  it an in-memory filesystem, cap its memory, and capture its stdio.

```go
llama.Init(llama.Config{
	PreopenDir:     "/srv/models", // the only directory the guest can see
	MaxMemoryBytes: 6 << 30,       // fail inside the guest, not in the host
})
```

## Streaming and interruption

Both work while a generation is running, and neither calls into the engine to
do it — they exchange a word and a ring buffer with the guest through linear
memory. That is not an optimisation: the engine is a single translated module
with one C stack, so the goroutine running `Generate` is the only one that can
be inside it.

```go
res, err := ctx.Stream("Once upon a time", llama.Params{NPredict: 512},
	func(piece string) { fmt.Print(piece) })
```

`Stream` returns the same complete `Result` as `Generate`. If the callback
cannot keep up, the engine drops pieces and `Stream` returns
`ErrStreamOverrun` alongside the (still complete) result.

## One engine per process

The engine has one linear memory and one C heap, so there is one of it per
process. Several models can be loaded at once and each gets its own contexts —
that is llama.cpp's own model. `Init` configures the engine and takes effect
once; `LoadModel` starts it with the defaults if `Init` has not run.

## Memory

wasm32 caps linear memory at 4 GiB, and the model weights plus every context's
KV cache live inside it. Target quantized models comfortably under that —
roughly 3B parameters at Q4 — and size `ContextParams.NCtx` accordingly.
`Config.MaxMemoryBytes` caps growth so an oversized model fails in the guest
instead of growing the host process, and `Config.MemoryReserveBytes` reserves
up front so a large load does not repeatedly grow and copy.

## Performance

The generated Go is compiled by the Go compiler, and on amd64/arm64 most of it
ships as assembly derived from that compilation. The SIMD kernels ggml relies
on are native NEON on arm64 and SSE on amd64 (the latter under `GOAMD64=v2` or
higher — set it, or the vector helpers fall back to scalar Go).

## Supply-chain verification

`internal/bridge/llama.go` in this repository is a release artifact of
[llama-wasm](https://github.com/goccy/llama-wasm), not hand-written code. It is
refreshed with:

```sh
make llama LLAMA_WASM_VERSION=v0.1.0
```

which downloads it and verifies its build-provenance attestation against
llama-wasm's release workflow.

## Testing

```sh
make test        # fetches a tiny GGUF into testdata/ and runs the suite
```

Point the suite at your own model with `GO_LLAMA_TEST_MODEL=/path/to.gguf`.

## License

MIT (see LICENSE). llama.cpp is MIT; the embedded engine is a derivative work
of it.
