package llama_test

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestWikitextPerplexityParity reproduces llama-perplexity's protocol on
// wikitext-2 and compares the result with the value RECORDED from
// native llama.cpp built at the pinned submodule commit
// (testdata/goldens/native_ppl_wikitext.json): tokenize the whole
// corpus once, cut it into n_ctx-token chunks, and score positions
// n_ctx/2+1 .. n_ctx-1 of each chunk given their prefix. Score reports
// whole-text sums, so a chunk's second-half NLL is
// Score(chunk) - Score(first half).
//
// The tolerance is tight on purpose: native's own paths (flash
// attention on/off, micro-batch 1) move the number by 0.04%, and the
// engine sits within 0.03% of native on the same hardware class. A
// kernel that rounds wrongly shows up at the percent level; the older
// TestNativeNLLParity's 10% q8 envelope cannot see that.
//
// Needs the q8 qwen model and the corpus (`make testdata-wikitext`);
// skips when either is missing.
func TestWikitextPerplexityParity(t *testing.T) {
	raw, err := os.ReadFile("testdata/goldens/native_ppl_wikitext.json")
	if err != nil {
		t.Fatalf("golden: %v", err)
	}
	var golden struct {
		Model     string  `json:"model"`
		Corpus    string  `json:"corpus"`
		NCtx      int     `json:"n_ctx"`
		Chunks    int     `json:"chunks"`
		NativePPL float64 `json:"native_ppl"`
		RelTol    float64 `json:"rel_tol"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("golden: %v", err)
	}
	if verifyModelPath(t, golden.Model) == "" {
		t.Skipf("%s missing under testdata", golden.Model)
	}
	corpus, err := os.ReadFile(filepath.Join("testdata", golden.Corpus))
	if err != nil {
		t.Skipf("corpus missing (run `make testdata-wikitext`): %v", err)
	}
	m, err := testInst.LoadModel(golden.Model)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	toks, err := m.Tokenize(string(corpus), true, false)
	if err != nil {
		t.Fatal(err)
	}
	nctx := golden.NCtx
	if len(toks) < golden.Chunks*nctx {
		t.Fatalf("corpus has %d tokens, need %d", len(toks), golden.Chunks*nctx)
	}
	score := func(text string) llama.ScoreResult {
		t.Helper()
		ctx, err := m.NewContext(llama.ContextParams{NCtx: uint32(nctx + 64), NBatch: 512, NUBatch: 512})
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Close()
		r, err := ctx.Score(text)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	first := nctx / 2
	var nll float64
	var count, skipped int
	for i := 0; i < golden.Chunks; i++ {
		chunk := toks[i*nctx : (i+1)*nctx]
		full, err := m.Detokenize(chunk, false)
		if err != nil {
			t.Fatal(err)
		}
		pre, err := m.Detokenize(chunk[:first+1], false)
		if err != nil {
			t.Fatal(err)
		}
		rf, rp := score(full), score(pre)
		// A chunk whose text does not re-tokenize to the same token
		// counts cannot be scored by subtraction; wikitext round-trips,
		// so more than a couple would mean the tokenizer changed.
		if int(rf.NTokens) != nctx || int(rp.NTokens) != first+1 {
			skipped++
			continue
		}
		nll += rf.NLL - rp.NLL
		count += nctx - 1 - first
	}
	if skipped > 2 {
		t.Fatalf("%d of %d chunks did not re-tokenize to their token counts", skipped, golden.Chunks)
	}
	ppl := math.Exp(nll / float64(count))
	rel := math.Abs(ppl-golden.NativePPL) / golden.NativePPL
	t.Logf("wikitext-2 PPL over %d chunks x %d: go %.4f, native %.4f (rel %.2e)", golden.Chunks-skipped, nctx, ppl, golden.NativePPL, rel)
	if rel > golden.RelTol {
		t.Fatalf("PPL diverges from native: go %.4f, native %.4f (rel %.2e > %.0e)", ppl, golden.NativePPL, rel, golden.RelTol)
	}
}
