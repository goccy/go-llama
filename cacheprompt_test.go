package llama_test

import (
	"strings"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestCachePromptMatchesFreshDecode pins the CachePrompt contract: reusing
// the KV prefix must change nothing about the output — a cached generation
// is byte-identical to the same prompt decoded from scratch — while actually
// reusing the shared prefix (NCached reports it).
func TestCachePromptMatchesFreshDecode(t *testing.T) {
	m := load(t)
	defer m.Close()

	preamble := strings.Repeat("Once upon a time there was a tiny model that routed every request to the right place. ", 6)
	p1 := preamble + "The fox asked about the moon."
	p2 := preamble + "The bear asked about the sea."
	params := func(cache bool) llama.Params {
		return llama.Params{NPredict: 12, Temperature: 0, CachePrompt: cache}
	}

	// Reference outputs: each prompt decoded from scratch on its own context.
	fresh := func(p string) string {
		t.Helper()
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Close()
		res, err := ctx.Generate(p, params(false))
		if err != nil {
			t.Fatal(err)
		}
		return res.Text
	}
	want1, want2 := fresh(p1), fresh(p2)

	// Cached sequence on ONE context: first call is cold, the second reuses
	// the shared preamble, the third replays the first prompt.
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	r1, err := ctx.Generate(p1, params(true))
	if err != nil {
		t.Fatal(err)
	}
	if r1.NCached != 0 {
		t.Errorf("cold cache_prompt generate reports NCached=%d; want 0", r1.NCached)
	}
	r2, err := ctx.Generate(p2, params(true))
	if err != nil {
		t.Fatal(err)
	}
	if r2.NCached == 0 {
		t.Error("second cache_prompt generate reused nothing; want a shared preamble prefix")
	}
	r3, err := ctx.Generate(p1, params(true))
	if err != nil {
		t.Fatal(err)
	}
	if r3.NCached == 0 {
		t.Error("replayed cache_prompt generate reused nothing")
	}

	if r1.Text != want1 {
		t.Errorf("cold cached output diverges from fresh decode:\n  got  %q\n  want %q", r1.Text, want1)
	}
	if r2.Text != want2 {
		t.Errorf("prefix-reusing output diverges from fresh decode:\n  got  %q\n  want %q", r2.Text, want2)
	}
	if r3.Text != want1 {
		t.Errorf("replayed output diverges:\n  got  %q\n  want %q", r3.Text, want1)
	}
}

// TestStateBlobCarriesPrefixHistory pins the composition of the two caching
// mechanisms: a state blob carries the prefix history, so a context restored
// with LoadState immediately reuses the shared prefix under CachePrompt
// instead of rebuilding the cache from scratch.
func TestStateBlobCarriesPrefixHistory(t *testing.T) {
	m := load(t)
	defer m.Close()

	preamble := strings.Repeat("Once upon a time there was a tiny model that routed every request to the right place. ", 6)
	p1 := preamble + "The fox asked about the moon."
	p2 := preamble + "The bear asked about the sea."
	params := llama.Params{NPredict: 12, Temperature: 0, CachePrompt: true}

	fresh := func(p string) string {
		t.Helper()
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Close()
		res, err := ctx.Generate(p, llama.Params{NPredict: 12, Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		return res.Text
	}
	want2 := fresh(p2)

	// Warm one context, snapshot it with the history inside.
	ctx1, err := m.NewContext(llama.ContextParams{NCtx: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx1.Close()
	if _, err := ctx1.Generate(p1, params); err != nil {
		t.Fatal(err)
	}
	blob, err := ctx1.SaveState()
	if err != nil {
		t.Fatal(err)
	}

	// A fresh context restored from the blob must reuse the preamble on its
	// very first CachePrompt generate — no full rebuild.
	ctx2, err := m.NewContext(llama.ContextParams{NCtx: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx2.Close()
	if err := ctx2.LoadState(blob); err != nil {
		t.Fatal(err)
	}
	r, err := ctx2.Generate(p2, params)
	if err != nil {
		t.Fatal(err)
	}
	if r.NCached == 0 {
		t.Error("restored context reused nothing on its first CachePrompt generate; want the shared preamble")
	}
	if r.Text != want2 {
		t.Errorf("restored-context output diverges from fresh decode:\n  got  %q\n  want %q", r.Text, want2)
	}
}
