package llama_test

import (
	"math"
	"path/filepath"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestScoreResultsFinite pins that Score and ScoreChoices return finite
// negative log-likelihoods on every model in testdata. The tolerance
// comparisons in the other score tests are silent on NaN (|NaN - x| > tol
// is false), so a kernel that poisons the logits passes them: the
// llamawasm2go v0.3.3 engine did exactly that on AVX-512 VNNI hosts,
// where the q8_0x4 repack GEMV clobbered one column group's accumulator
// per pass (goccy/llama-wasm#21) and every choice scored NaN. The
// quantized models are the ones that take that kernel; the f32 story
// model covers the plain path.
func TestScoreResultsFinite(t *testing.T) {
	const stem = "Once upon a time, in a land far away, there lived a"
	choices := []string{" little girl", " dragon", " brave knight of the realm"}

	models := []string{"stories260K.gguf"}
	for _, name := range []string{"qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"} {
		if verifyModelPath(t, name) != "" {
			models = append(models, name)
		}
	}
	// ScoreChoices scores every token of a choice and reports only
	// NTokens; NScored is Score's count of teacher-forced positions.
	finite := func(t *testing.T, what string, r llama.ScoreResult) {
		t.Helper()
		if r.NTokens <= 0 {
			t.Errorf("%s: no tokens: %+v", what, r)
		}
		if math.IsNaN(r.NLL) || math.IsInf(r.NLL, 0) {
			t.Errorf("%s: NLL = %v, want finite", what, r.NLL)
		}
	}
	for _, name := range models {
		t.Run(name, func(t *testing.T) {
			if verifyModelPath(t, name) == "" {
				t.Skipf("model missing: %s", name)
			}
			m, err := testInst.LoadModel(filepath.Base(name))
			if err != nil {
				t.Fatalf("LoadModel: %v", err)
			}
			defer m.Close()

			ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
			if err != nil {
				t.Fatal(err)
			}
			defer ctx.Close()
			full, err := ctx.Score(stem + choices[0])
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			finite(t, "Score", full)
			if full.NScored <= 0 {
				t.Errorf("Score: no positions scored: %+v", full)
			}

			cctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
			if err != nil {
				t.Fatal(err)
			}
			defer cctx.Close()
			if _, err := cctx.Eval(stem, true, true); err != nil {
				t.Fatal(err)
			}
			scores, err := cctx.ScoreChoices(choices)
			if err != nil {
				t.Fatalf("ScoreChoices: %v", err)
			}
			for i, r := range scores {
				finite(t, "ScoreChoices "+choices[i], r)
			}
		})
	}
}
