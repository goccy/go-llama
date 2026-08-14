package llama_test

import (
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	llama "github.com/goccy/go-llama"
)

// TestNativeGreedyEvenBatch drives an even-token-count prompt so the
// nrc==2 (paired rows/columns) matmul path runs during prompt
// processing, and compares the greedy continuation against native.
func TestNativeGreedyEvenBatch(t *testing.T) {
	dir := nativeRef(t)
	model := verifyModelPath(t, "qwen2.5-0.5b-instruct-q8_0.gguf")
	if model == "" {
		t.Skip("model missing")
	}
	m, err := testInst.LoadModel(filepath.Base(model))
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	// 75 prompt tokens + BOS = 76 (even).
	prompt := strings.Repeat("Once upon a time there was a little girl who loved to play outside. ", 5)
	res, err := ctx.Generate(prompt, llama.Params{NPredict: 32, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.NDecoded == 0 || !utf8.ValidString(res.Text) {
		t.Fatalf("degenerate generation %q", res.Text)
	}
	native := nativeGreedy(t, dir, model, prompt, 32)
	goFull := prompt + res.Text
	common := 0
	gr, nr := []rune(goFull), []rune(native)
	for common < len(gr) && common < len(nr) && gr[common] == nr[common] {
		common++
	}
	promptRunes := len([]rune(prompt))
	t.Logf("common prefix %d runes (prompt %d)\n  go:     %q\n  native: %q",
		common, promptRunes, res.Text, strings.TrimPrefix(native, string(nr[:min(len(nr), promptRunes)])))
	if common < promptRunes+8 {
		t.Errorf("continuation diverges from native almost immediately (common %d, prompt %d)", common, promptRunes)
	}
}
