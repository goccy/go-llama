package llama_test

// Self-contained verification: properties that need no native
// reference, only the engine's own invariants.

import (
	"sync"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestPromptBatchSizeInvariance: how the prompt is fed through the
// batch pipeline must not change what the model computes. Processing
// with a large batch and with a tiny one has to produce the same greedy
// continuation — a mismatch means batching leaks into the math.
func TestPromptBatchSizeInvariance(t *testing.T) {
	m := load(t)
	prompt := "Once upon a time there was a little girl who loved to play outside"
	params := llama.Params{NPredict: 12, Temperature: 0}

	var baseline string
	for _, nb := range []uint32{0, 8, 1} {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256, NBatch: nb, NUBatch: nb})
		if err != nil {
			t.Fatalf("NewContext(NBatch=%d): %v", nb, err)
		}
		res, err := ctx.Generate(prompt, params)
		ctx.Close()
		if err != nil {
			t.Fatalf("Generate(NBatch=%d): %v", nb, err)
		}
		if baseline == "" {
			baseline = res.Text
			continue
		}
		if res.Text != baseline {
			t.Errorf("NBatch=%d diverges:\n  got:      %q\n  baseline: %q", nb, res.Text, baseline)
		}
	}
}

// TestConcurrentContexts drives several contexts from separate
// goroutines at once. The engine serialises entries internally; the
// property under test is that concurrency cannot corrupt state — every
// goroutine's greedy output must equal the serial baseline.
func TestConcurrentContexts(t *testing.T) {
	m := load(t)
	params := llama.Params{NPredict: 8, Temperature: 0}

	base, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	want, err := base.Generate("Once upon a time", params)
	base.Close()
	if err != nil {
		t.Fatal(err)
	}

	const n = 4
	var wg sync.WaitGroup
	errs := make([]error, n)
	texts := make([]string, n)
	for i := 0; i < n; i++ {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Close()
		wg.Add(1)
		go func(i int, ctx *llama.Context) {
			defer wg.Done()
			res, err := ctx.Generate("Once upon a time", params)
			errs[i], texts[i] = err, res.Text
		}(i, ctx)
	}
	wg.Wait()
	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("goroutine %d: %v", i, errs[i])
		}
		if texts[i] != want.Text {
			t.Errorf("goroutine %d diverged:\n  got:  %q\n  want: %q", i, texts[i], want.Text)
		}
	}
}

// TestGenerationIsRepeatableAcrossContexts: a fresh context must always
// reproduce the same greedy output — the cross-run determinism that the
// FMA-fusion fix in wasm2go established. A regression here means the
// engine's floats are drifting again.
func TestGenerationIsRepeatableAcrossContexts(t *testing.T) {
	m := load(t)
	params := llama.Params{NPredict: 16, Temperature: 0}
	var want string
	for i := 0; i < 3; i++ {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
		if err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate("Once upon a time", params)
		ctx.Close()
		if err != nil {
			t.Fatal(err)
		}
		if want == "" {
			want = res.Text
		} else if res.Text != want {
			t.Fatalf("run %d diverged:\n  got:  %q\n  want: %q", i, res.Text, want)
		}
	}
}
