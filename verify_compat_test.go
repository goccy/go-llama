package llama_test

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"strings"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestTokenizerRoundTrip asserts Detokenize(Tokenize(text)) == text
// across the multilingual corpus, and that per-token pieces
// concatenate to the same string — llama.cpp's tokenizer is lossless
// over ordinary text, so any divergence is a real compatibility bug.
func TestTokenizerRoundTrip(t *testing.T) {
	model := verifyModelPath(t, "qwen2.5-0.5b-instruct-q8_0.gguf")
	if model == "" {
		t.Skip("model missing")
	}
	m, err := testInst.LoadModel("qwen2.5-0.5b-instruct-q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, text := range tokenizerCorpus {
		if text == "" || strings.Contains(text, "<|") {
			continue // specials render differently by design
		}
		toks, err := m.Tokenize(text, false, false)
		if err != nil {
			t.Fatalf("Tokenize(%q): %v", text, err)
		}
		back, err := m.Detokenize(toks, false)
		if err != nil {
			t.Fatalf("Detokenize(%q): %v", text, err)
		}
		if back != text {
			t.Errorf("round trip diverges:\n  in:  %q\n  out: %q", text, back)
		}
		var pieces strings.Builder
		for _, tok := range toks {
			p, err := m.TokenToPiece(tok, false)
			if err != nil {
				t.Fatalf("TokenToPiece(%d): %v", tok, err)
			}
			// Byte-fallback tokens carry partial UTF-8 sequences; the
			// bridge's base64 channel transports them byte-exactly, so
			// concatenating raw pieces must reproduce the text.
			pieces.WriteString(p)
		}
		if pieces.String() != back {
			t.Errorf("piece concatenation diverges from Detokenize for %q:\n  pieces: %q\n  detok:  %q",
				text, pieces.String(), back)
		}
	}
}

// TestSamplerContract pins the parameter behaviors go-llama forwards
// to llama.cpp's sampler chain: the properties below hold in native
// llama.cpp and must hold here regardless of float error.
func TestSamplerContract(t *testing.T) {
	model := verifyModelPath(t, "qwen2.5-0.5b-instruct-q8_0.gguf")
	if model == "" {
		t.Skip("model missing")
	}
	m, err := testInst.LoadModel("qwen2.5-0.5b-instruct-q8_0.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	gen := func(p llama.Params) llama.Result {
		t.Helper()
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate("The quick brown fox", p)
		if err != nil {
			t.Fatal(err)
		}
		return res
	}

	// NPredict is an exact cap.
	if res := gen(llama.Params{NPredict: 7, Temperature: 0}); res.NDecoded != 7 {
		t.Errorf("NPredict=7 produced %d tokens (%q)", res.NDecoded, res.Reason)
	}

	// Greedy decoding is deterministic.
	g1 := gen(llama.Params{NPredict: 24, Temperature: 0})
	g2 := gen(llama.Params{NPredict: 24, Temperature: 0})
	if g1.Text != g2.Text {
		t.Errorf("greedy not deterministic:\n  %q\n  %q", g1.Text, g2.Text)
	}

	// TopK=1 collapses sampling to greedy for any temperature — the
	// truncation samplers run before the temperature draw, exactly as
	// in llama.cpp's chain.
	k1 := gen(llama.Params{NPredict: 24, Temperature: 0.8, TopK: 1, Seed: 42})
	if k1.Text != g1.Text {
		t.Errorf("TopK=1 diverges from greedy:\n  topk1:  %q\n  greedy: %q", k1.Text, g1.Text)
	}

	// A fixed seed replays identically; a different seed is free to
	// (and here does) diverge.
	s1 := gen(llama.Params{NPredict: 24, Temperature: 0.9, TopP: 0.9, Seed: 7})
	s2 := gen(llama.Params{NPredict: 24, Temperature: 0.9, TopP: 0.9, Seed: 7})
	if s1.Text != s2.Text {
		t.Errorf("seeded sampling not reproducible:\n  %q\n  %q", s1.Text, s2.Text)
	}

	// Stop strings end generation and are trimmed from the text.
	// llama.cpp's server semantics cut at the stop string's first
	// occurrence ANYWHERE in the new text, including mid-piece.
	st := gen(llama.Params{NPredict: 64, Temperature: 0, Stop: []string{"."}})
	if strings.Contains(st.Text, ".") || st.NDecoded >= 64 {
		t.Errorf("boundary stop string not honored: %q (%d tokens)", st.Text, st.NDecoded)
	}

	// A GBNF grammar constrains the output alphabet.
	gr := gen(llama.Params{NPredict: 12, Temperature: 0,
		Grammar: "root ::= [0-9]+"})
	if gr.Text == "" || !regexp.MustCompile(`^[0-9]+$`).MatchString(gr.Text) {
		t.Errorf("grammar-constrained output not all digits: %q", gr.Text)
	}

	// A stop string that lands mid-piece (a leading space is part of the
	// next word's piece in BPE vocabularies) must still cut the text.
	if mid := gen(llama.Params{NPredict: 16, Temperature: 0, Stop: []string{" "}}); strings.Contains(mid.Text, " ") {
		t.Errorf("mid-piece stop string not honored: %q", mid.Text)
	}
}

// TestNativeNLLParity compares Score — the teacher-forced negative
// log-likelihood of a text, the quantity llama.cpp's perplexity tool
// averages — against values RECORDED from native llama.cpp at the
// pinned commit (testdata/goldens/native_nll.json). Token counts must
// match exactly (the tokenizer is integer-exact); the NLL sum must
// agree within a relative tolerance covering float accumulation
// (f32 model) plus the quantized kernels' rounding envelope (q8).
func TestNativeNLLParity(t *testing.T) {
	raw, err := os.ReadFile("testdata/goldens/native_nll.json")
	if err != nil {
		t.Fatalf("golden: %v", err)
	}
	var golden struct {
		NLL map[string]map[string]struct {
			NTokens int32   `json:"n_tokens"`
			NScored int32   `json:"n_scored"`
			NLL     float64 `json:"nll"`
		} `json:"nll"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("golden: %v", err)
	}
	// The f32 model agrees to float-rounding precision (measured rel
	// error <= 4e-5). The q8 model's NLL against native moves inside a
	// few-percent envelope: native NEON kernels, the wasm-exact order
	// and the fast-math transpile each accumulate q8 dot products in a
	// different order (measured: <= 3.2% exact-math, <= 6.5%
	// fast-math). The gate exists to catch structural breakage — a
	// wrong kernel corrupts NLL by orders of magnitude, not percent.
	cases := []struct {
		file   string
		relTol float64
	}{
		{"stories260K.gguf", 0.005},
		{"qwen2.5-0.5b-instruct-q8_0.gguf", 0.10},
	}
	for _, tc := range cases {
		model := verifyModelPath(t, tc.file)
		if model == "" {
			continue
		}
		m, err := testInst.LoadModel(tc.file)
		if err != nil {
			t.Fatal(err)
		}
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
		if err != nil {
			t.Fatal(err)
		}
		for text, want := range golden.NLL[tc.file] {
			if err := ctx.Reset(); err != nil {
				t.Fatal(err)
			}
			got, err := ctx.Score(text)
			if err != nil {
				t.Fatalf("Score(%q): %v", text, err)
			}
			if got.NTokens != want.NTokens || got.NScored != want.NScored {
				t.Errorf("%s: token counts diverge for %q: go %d/%d, native %d/%d",
					tc.file, text, got.NTokens, got.NScored, want.NTokens, want.NScored)
				continue
			}
			rel := math.Abs(got.NLL-want.NLL) / want.NLL
			t.Logf("%s nll(%q) = %.6f (native %.6f, rel %.2e)", tc.file, text, got.NLL, want.NLL, rel)
			if rel > tc.relTol {
				t.Errorf("%s: NLL diverges for %q: go %.6f, native %.6f (rel %.2e > %.0e)",
					tc.file, text, got.NLL, want.NLL, rel, tc.relTol)
			}
		}
		ctx.Close()
		m.Close()
	}
}

// TestNativeEmbeddingParity compares Embed against llama-embedding
// output RECORDED at the pinned llama.cpp commit
// (testdata/goldens/native_embeddings.json — regenerate with
// `llama-embedding --embd-output-format json --embd-normalize 2` and
// take data[0]): direction must agree to high cosine similarity, the
// float-tolerant form of "the encoder computes the same thing".
func TestNativeEmbeddingParity(t *testing.T) {
	raw, err := os.ReadFile("testdata/goldens/native_embeddings.json")
	if err != nil {
		t.Fatalf("golden: %v", err)
	}
	var golden struct {
		Embeddings map[string]map[string][]float64 `json:"embeddings"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("golden: %v", err)
	}
	model := verifyModelPath(t, "qwen2.5-0.5b-instruct-q8_0.gguf")
	if model == "" {
		t.Skip("model missing")
	}
	cases := []struct {
		file string
		min  float64
	}{
		// The f32 model must agree to float-rounding precision; the
		// quantized model tolerates its kernels' legitimate rounding
		// differences (measured cosine: >= 0.997 with the exact-math
		// transpile, >= 0.985 with fast-math). A structural bug drops
		// the cosine towards zero, so the gate keeps its power.
		{"stories260K.gguf", 0.9999},
		{"qwen2.5-0.5b-instruct-q8_0.gguf", 0.98},
	}
	for _, tc := range cases {
		model = verifyModelPath(t, tc.file)
		if model == "" {
			continue
		}
		m, err := testInst.LoadModel(tc.file)
		if err != nil {
			t.Fatal(err)
		}
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256, Embeddings: true})
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"Hello", "Hello world", "日本の首都は東京です。"} {
			got, err := ctx.Embed(text, true)
			if err != nil {
				t.Fatalf("Embed(%q): %v", text, err)
			}
			want := golden.Embeddings[tc.file][text]
			if len(want) == 0 {
				t.Fatalf("no golden embedding for %s %q", tc.file, text)
			}
			if len(want) != len(got) {
				t.Fatalf("embedding size: go %d, native %d", len(got), len(want))
			}
			var dot, ng, nw float64
			for i := range got {
				dot += float64(got[i]) * want[i]
				ng += float64(got[i]) * float64(got[i])
				nw += want[i] * want[i]
			}
			cos := dot / math.Sqrt(ng*nw)
			t.Logf("%s cosine(%q) = %.6f", tc.file, text, cos)
			if cos < tc.min {
				t.Errorf("%s: embedding direction diverges for %q: cosine %.6f < %.4f",
					tc.file, text, cos, tc.min)
			}
		}
		ctx.Close()
		m.Close()
	}
}
