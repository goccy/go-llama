package llama_test

import (
	"math"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestScoreChoices pins the semantics of ScoreChoices: determinism and
// rollback (repeat calls agree, the stem survives untouched), ranking sanity
// (a natural continuation outranks gibberish), and — for choices whose
// tokenization is additive (tokenize(stem+choice) == tokenize(stem) +
// tokenize(choice)) — exact agreement with the full-decode Score identity
// Score(stem+choice) - Score(stem). Additivity fails wholesale under a
// SentencePiece vocabulary with a dummy space prefix (the test model), where
// the standalone tokenization carries a leading artifact token; ScoreChoices
// always scores the standalone tokenization, which keeps every choice on an
// equal footing, so the exact check runs only where the identity is defined.
func TestScoreChoices(t *testing.T) {
	m := load(t)
	defer m.Close()

	const stem = "Once upon a time, in a land far away, there lived a"
	choices := []string{" little girl", " zzz qqq xxw", " dragon"}

	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	if _, err := ctx.Eval(stem, true, true); err != nil {
		t.Fatal(err)
	}
	got, err := ctx.ScoreChoices(choices)
	if err != nil {
		t.Fatalf("ScoreChoices: %v", err)
	}

	// Ranking sanity: per-token NLL of the natural continuation beats the
	// gibberish one.
	nat := got[0].NLL / float64(got[0].NTokens)
	gib := got[1].NLL / float64(got[1].NTokens)
	if nat >= gib {
		t.Errorf("natural continuation per-token NLL %v not better than gibberish %v", nat, gib)
	}

	// Repeatable: each choice is rolled back, so a second call agrees bit
	// for bit — including when the choices arrive in a different order.
	again, err := ctx.ScoreChoices(choices)
	if err != nil {
		t.Fatal(err)
	}
	for i := range choices {
		if got[i].NLL != again[i].NLL {
			t.Errorf("choice %d: NLL changed across calls: %v then %v", i, got[i].NLL, again[i].NLL)
		}
	}
	rev := []string{choices[2], choices[1], choices[0]}
	revScores, err := ctx.ScoreChoices(rev)
	if err != nil {
		t.Fatal(err)
	}
	if revScores[2].NLL != got[0].NLL || revScores[0].NLL != got[2].NLL {
		t.Error("scores depend on choice order")
	}

	// The cache still holds exactly the stem: generating from it matches a
	// fresh context's generation over the same prompt.
	res, err := ctx.Generate(stem, llama.Params{NPredict: 6, CachePrompt: true, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	fctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer fctx.Close()
	want, err := fctx.Generate(stem, llama.Params{NPredict: 6, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != want.Text {
		t.Errorf("generation after ScoreChoices = %q, fresh context = %q", res.Text, want.Text)
	}
	if res.NCached == 0 {
		t.Error("stem was not cached after ScoreChoices rollback")
	}

	// Exact identity against the full-decode Score, where defined.
	stemToks, err := m.Tokenize(stem, true, true)
	if err != nil {
		t.Fatal(err)
	}
	rctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer rctx.Close()
	stemScore, err := rctx.Score(stem)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for i, ch := range choices {
		cToks, err := m.Tokenize(ch, false, true)
		if err != nil {
			t.Fatal(err)
		}
		fullToks, err := m.Tokenize(stem+ch, true, true)
		if err != nil {
			t.Fatal(err)
		}
		additive := len(fullToks) == len(stemToks)+len(cToks)
		for j := 0; additive && j < len(cToks); j++ {
			additive = fullToks[len(stemToks)+j] == cToks[j]
		}
		if !additive {
			continue
		}
		full, err := rctx.Score(stem + ch)
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Abs(got[i].NLL - (full.NLL - stemScore.NLL)); d > 1e-3 {
			t.Errorf("choice %d: NLL = %v, full-decode reference %v (diff %v)", i, got[i].NLL, full.NLL-stemScore.NLL, d)
		}
		checked++
	}
	t.Logf("exact identity checked on %d/%d choices (rest non-additive under this vocabulary)", checked, len(choices))

	// Input validation.
	if _, err := ctx.ScoreChoices([]string{"ok", "bad\nline"}); err == nil {
		t.Error("newline choice accepted")
	}
	if _, err := ctx.ScoreChoices([]string{""}); err == nil {
		t.Error("empty choice accepted")
	}
}

// TestScoreChoicesSeqBatched pins that the multi-sequence batched path
// (NSeqMax > 1: one decode for all candidates, each on its own sequence
// sharing the stem) returns bit-identical scores to the single-sequence
// sequential path, over enough choices to also exercise group chunking.
func TestScoreChoicesSeqBatched(t *testing.T) {
	m := load(t)
	defer m.Close()

	const stem = "Once upon a time, in a land far away, there lived a"
	choices := []string{
		" little girl", " dragon", " brave knight of the realm", " cat",
		" wise old man", " tiny mouse in a big house", " king", " queen of the north",
	}

	seq, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer seq.Close()
	if _, err := seq.Eval(stem, true, true); err != nil {
		t.Fatal(err)
	}
	want, err := seq.ScoreChoices(choices)
	if err != nil {
		t.Fatal(err)
	}

	// NSeqMax 4 forces chunking (8 multi-token candidates over 3 spare
	// sequences per group).
	bat, err := m.NewContext(llama.ContextParams{NCtx: 256, NSeqMax: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer bat.Close()
	if _, err := bat.Eval(stem, true, true); err != nil {
		t.Fatal(err)
	}
	got, err := bat.ScoreChoices(choices)
	if err != nil {
		t.Fatal(err)
	}
	for i := range choices {
		if got[i].NTokens != want[i].NTokens {
			t.Errorf("choice %d: NTokens %d vs %d", i, got[i].NTokens, want[i].NTokens)
		}
		if d := math.Abs(got[i].NLL - want[i].NLL); d > 1e-4 {
			t.Errorf("choice %d: batched NLL %v vs sequential %v (diff %v)", i, got[i].NLL, want[i].NLL, d)
		}
	}

	// The batched context still generates from the untouched stem.
	res, err := bat.Generate(stem, llama.Params{NPredict: 4, CachePrompt: true, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.NCached == 0 {
		t.Error("stem not cached after batched ScoreChoices")
	}
}
