package llama_test

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestModernBertParity loads a ModernBERT encoder GGUF and compares its
// tokenization and CLS-pooled embedding with the PyTorch values recorded in
// testdata/modernbert-ja-130m-reference.json (sbintuitions/modernbert-ja-130m,
// transformers AutoModel, last_hidden_state[:, 0]). The texts cover newline,
// tab, full-width space and emoji, which the UGM tokenizer can only match
// through byte fallback.
//
// Opt-in via MODERNBERT_GGUF (file name under testdata; convert the public
// checkpoint with llama-wasm's bench/encoder/convert.py). Token ids must be
// identical. The CLS vector's largest element-wise difference from PyTorch,
// relative to the largest magnitude in the reference vector, must stay within
// MODERNBERT_REL_TOL (default 1e-2, raise for quantized weights), and its
// cosine similarity must stay at or above MODERNBERT_COS_MIN (default 0.999).
func TestModernBertParity(t *testing.T) {
	name := os.Getenv("MODERNBERT_GGUF")
	if name == "" {
		t.Skip("set MODERNBERT_GGUF=modernbert-ja-130m-f32.gguf (under testdata) to run")
	}
	relTol := envFloat(t, "MODERNBERT_REL_TOL", 1e-2)
	cosMin := envFloat(t, "MODERNBERT_COS_MIN", 0.999)
	var ref struct {
		Model string `json:"model"`
		NEmbd int    `json:"n_embd"`
		Tests []struct {
			Text   string    `json:"text"`
			Tokens []int32   `json:"tokens"`
			CLS    []float32 `json:"cls"`
		} `json:"tests"`
	}
	b, err := os.ReadFile("testdata/modernbert-ja-130m-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := testInst.LoadModel(name)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	info, err := m.Info()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: n_embd=%d n_layer=%d n_vocab=%d (reference %s)", name, info.NEmbd, info.NLayer, info.NVocab, ref.Model)
	if int(info.NEmbd) != ref.NEmbd {
		t.Fatalf("n_embd %d, reference %d", info.NEmbd, ref.NEmbd)
	}
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 1024, NBatch: 1024, NUBatch: 1024, NThreads: 4, Embeddings: true})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()
	for _, tc := range ref.Tests {
		toks, err := m.Tokenize(tc.Text, true, true)
		if err != nil {
			t.Fatal(err)
		}
		same := len(toks) == len(tc.Tokens)
		for i := 0; same && i < len(toks); i++ {
			same = toks[i] == tc.Tokens[i]
		}
		if !same {
			t.Errorf("tokens differ for %q: got %v, reference %v", tc.Text, toks, tc.Tokens)
		}
		vec, err := ctx.EmbedTokens(tc.Tokens, false)
		if err != nil {
			t.Fatalf("EmbedTokens: %v", err)
		}
		if len(vec) != len(tc.CLS) {
			t.Fatalf("CLS length %d, reference %d", len(vec), len(tc.CLS))
		}
		var maxd, maxr, dot, na, nb float64
		for i := range vec {
			a, b := float64(vec[i]), float64(tc.CLS[i])
			maxd = math.Max(maxd, math.Abs(a-b))
			maxr = math.Max(maxr, math.Abs(b))
			dot += a * b
			na += a * a
			nb += b * b
		}
		cos := dot / math.Sqrt(na*nb)
		rel := maxd / maxr
		t.Logf("%3d tokens  cls max|diff| %.2e (%.2e of max|ref| %.2e)  cos %.6f  %q",
			len(toks), maxd, rel, maxr, cos, tc.Text[:min(24, len(tc.Text))])
		if rel > relTol {
			t.Errorf("CLS embedding diverges from PyTorch: relative max|diff| %.3e > %.1e", rel, relTol)
		}
		if cos < cosMin {
			t.Errorf("CLS embedding diverges from PyTorch: cosine %.6f < %.4f", cos, cosMin)
		}
	}
}

func envFloat(t *testing.T, key string, def float64) float64 {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("%s: %v", key, err)
	}
	return v
}
