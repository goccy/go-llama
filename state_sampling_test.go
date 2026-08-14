package llama_test

import (
	"math"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestStateSaveLoadRoundTrip pins the state API's contract: a state
// saved after decoding a prefix restores into a FRESH context, and
// generation from the restored state matches generation from the
// original — the KV cache and RNG really did travel.
func TestStateSaveLoadRoundTrip(t *testing.T) {
	m := load(t)
	ctx1, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx1.Close()

	if _, err := ctx1.Eval("Once upon a time there was", true, false); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	state, err := ctx1.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	if len(state) == 0 {
		t.Fatal("SaveState returned an empty state")
	}

	p := llama.Params{NPredict: 12, Temperature: 0}
	r1, err := ctx1.Generate(" a little", p)
	if err != nil {
		t.Fatal(err)
	}

	ctx2, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx2.Close()
	if err := ctx2.LoadState(state); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	r2, err := ctx2.Generate(" a little", p)
	if err != nil {
		t.Fatal(err)
	}
	if r1.Text != r2.Text {
		t.Errorf("generation from restored state diverges:\n  original: %q\n  restored: %q", r1.Text, r2.Text)
	}
}

// TestEvalPrefill pins Eval's bookkeeping: token counts accumulate
// across calls, and a Reset starts the sequence over.
func TestEvalPrefill(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	e1, err := ctx.Eval("Once upon a time", true, false)
	if err != nil {
		t.Fatal(err)
	}
	if e1.NTokens <= 0 || e1.NPast != e1.NTokens {
		t.Errorf("first eval: NTokens=%d NPast=%d, want equal and positive", e1.NTokens, e1.NPast)
	}
	e2, err := ctx.Eval(" there was a girl", false, false)
	if err != nil {
		t.Fatal(err)
	}
	if e2.NPast != e1.NTokens+e2.NTokens {
		t.Errorf("positions did not continue: NPast=%d, want %d", e2.NPast, e1.NTokens+e2.NTokens)
	}
	if err := ctx.Reset(); err != nil {
		t.Fatal(err)
	}
	e3, err := ctx.Eval("Once upon a time", true, false)
	if err != nil {
		t.Fatal(err)
	}
	if e3.NPast != e3.NTokens {
		t.Errorf("Reset did not clear the sequence: NPast=%d, want %d", e3.NPast, e3.NTokens)
	}
}

// TestEmbedTokensMatchesEmbed: embedding from token ids is the same
// computation as embedding the text they tokenize from.
func TestEmbedTokensMatchesEmbed(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128, Embeddings: true})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	const text = "Once upon a time"
	fromText, err := ctx.Embed(text, true)
	if err != nil {
		t.Fatal(err)
	}
	// Embed tokenizes with add_special and parse_special, so the token
	// path must feed the identical ids to land on the identical vector.
	toks, err := m.Tokenize(text, true, true)
	if err != nil {
		t.Fatal(err)
	}
	fromToks, err := ctx.EmbedTokens(toks, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(fromText) != len(fromToks) {
		t.Fatalf("embedding sizes differ: %d vs %d", len(fromText), len(fromToks))
	}
	for i := range fromText {
		if fromText[i] != fromToks[i] {
			t.Fatalf("embeddings diverge at [%d]: %v vs %v", i, fromText[i], fromToks[i])
		}
	}
}

// eosToken resolves the model's end-of-sequence token id through the
// public API: parse_special tokenizes the EOS string to its id.
func eosToken(t *testing.T, m *llama.Model) int32 {
	t.Helper()
	for _, s := range []string{"</s>", "<|im_end|>", "<|endoftext|>"} {
		toks, err := m.Tokenize(s, false, true)
		if err == nil && len(toks) == 1 {
			return toks[0]
		}
	}
	t.Skip("cannot resolve an EOS token id via special-token parsing")
	return 0
}

// TestLogitBiasAndIgnoreEOS drives sampling through the bias plumbing:
// a huge positive bias on EOS forces generation to stop immediately,
// and IgnoreEOS overrides exactly that.
func TestLogitBiasAndIgnoreEOS(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	eos := eosToken(t, m)

	gen := func(p llama.Params) llama.Result {
		t.Helper()
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate("Once upon a time", p)
		if err != nil {
			t.Fatal(err)
		}
		return res
	}

	// +Inf bias (clamped on the wire) makes EOS the argmax: generation
	// ends before producing anything.
	forced := gen(llama.Params{NPredict: 8, Temperature: 0,
		LogitBias: map[int32]float32{eos: float32(math.Inf(1))}})
	if forced.Reason != llama.StopEOS || forced.NDecoded != 0 {
		t.Errorf("EOS bias did not force an immediate stop: reason=%q n=%d", forced.Reason, forced.NDecoded)
	}

	// IgnoreEOS excludes EOS from sampling even against that bias, so
	// the same call now runs to its token budget.
	ignored := gen(llama.Params{NPredict: 8, Temperature: 0, IgnoreEOS: true,
		LogitBias: map[int32]float32{eos: float32(math.Inf(1))}})
	if ignored.Reason != llama.StopLength || ignored.NDecoded != 8 {
		t.Errorf("IgnoreEOS did not run to budget: reason=%q n=%d (%q)", ignored.Reason, ignored.NDecoded, ignored.Text)
	}

	// A -Inf bias forbids a specific token: the previous greedy pick
	// cannot reappear in first position.
	base := gen(llama.Params{NPredict: 1, Temperature: 0})
	if len(base.Tokens) != 1 {
		t.Fatalf("greedy base run produced %d tokens", len(base.Tokens))
	}
	banned := base.Tokens[0]
	rerun := gen(llama.Params{NPredict: 1, Temperature: 0,
		LogitBias: map[int32]float32{banned: float32(math.Inf(-1))}})
	if len(rerun.Tokens) == 1 && rerun.Tokens[0] == banned {
		t.Errorf("token %d sampled despite a -Inf bias", banned)
	}
}

// TestMirostatContract pins what mirostat guarantees regardless of
// float error: it samples (successfully), respects the budget, and a
// fixed seed replays identically.
func TestMirostatContract(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	gen := func() llama.Result {
		t.Helper()
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate("Once upon a time", llama.Params{
			NPredict: 16, Temperature: 0.9, Mirostat: 2, Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	r1, r2 := gen(), gen()
	if r1.NDecoded == 0 || r1.Text == "" {
		t.Errorf("mirostat produced nothing: %+v", r1)
	}
	if r1.NDecoded > 16 {
		t.Errorf("mirostat overran NPredict: %d", r1.NDecoded)
	}
	if r1.Text != r2.Text {
		t.Errorf("mirostat with a fixed seed not reproducible:\n  %q\n  %q", r1.Text, r2.Text)
	}
}
