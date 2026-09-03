package llama_test

import (
	"math"
	"strings"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestScoreLongerThanBatch scores a text longer than the context's
// n_batch. The bridge used to hand the whole text to one llama_decode,
// whose n_batch assertion trapped the wasm instance; Score now decodes
// in n_batch slices. The slice boundaries change how the prompt
// matmuls batch rows, so under the engine's fast-math kernels the sum
// moves by float accumulation (measured 0.2% on the q8 model, 0.01% on
// the f32 one); a wrong slice would be off by whole tokens.
func TestScoreLongerThanBatch(t *testing.T) {
	m := load(t)
	defer m.Close()

	text := strings.Repeat("Once upon a time, in a land far away, there lived a little girl who loved to read books. ", 12)

	score := func(nBatch uint32) llama.ScoreResult {
		t.Helper()
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 1024, NBatch: nBatch, NUBatch: nBatch})
		if err != nil {
			t.Fatal(err)
		}
		defer ctx.Close()
		r, err := ctx.Score(text)
		if err != nil {
			t.Fatalf("Score with n_batch %d: %v", nBatch, err)
		}
		return r
	}
	whole := score(1024)
	sliced := score(32)
	if whole.NTokens <= 32 {
		t.Fatalf("text tokenizes to %d tokens; the test needs more than n_batch (32)", whole.NTokens)
	}
	if sliced.NTokens != whole.NTokens || sliced.NScored != whole.NScored {
		t.Fatalf("token counts: sliced %d/%d, whole %d/%d", sliced.NTokens, sliced.NScored, whole.NTokens, whole.NScored)
	}
	if rel := math.Abs(sliced.NLL-whole.NLL) / whole.NLL; rel > 1e-2 {
		t.Fatalf("NLL: sliced %.6f, whole %.6f (rel %.2e)", sliced.NLL, whole.NLL, rel)
	}
	t.Logf("n_tokens %d: NLL sliced %.6f, whole %.6f", whole.NTokens, sliced.NLL, whole.NLL)
}
