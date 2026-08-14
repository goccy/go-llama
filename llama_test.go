package llama_test

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// modelPath is the GGUF the tests run against. GO_LLAMA_TEST_MODEL points at a
// model of your choosing; otherwise the tiny stories260K model in testdata/ is
// used, which is small enough to commit and fast enough for CI while still
// exercising every code path.
func modelPath() string {
	if p := os.Getenv("GO_LLAMA_TEST_MODEL"); p != "" {
		return p
	}
	return filepath.Join("testdata", "stories260K.gguf")
}

// TestMain scopes the engine's filesystem to the directory the model lives in,
// which is both how an embedder should configure it and the only chance to
// cover Config: the engine is process-wide, so this is the one place a test
// binary can configure it.
// testInst is the engine instance the suite loads models into. One
// instance, scoped to the model directory, shared across tests — each test
// loads its own model handle into it.
var testInst *llama.Llama

func TestMain(m *testing.M) {
	dir, err := filepath.Abs(filepath.Dir(modelPath()))
	if err != nil {
		panic(err)
	}
	inst, err := llama.New(llama.WithPreopenDir(dir))
	if err != nil {
		panic(err)
	}
	testInst = inst
	os.Exit(m.Run())
}

// load opens the test model. The engine's filesystem is rooted at the model's
// directory (see TestMain), so the guest path is just the file name.
func load(t *testing.T) *llama.Model {
	t.Helper()
	p := modelPath()
	if _, err := os.Stat(p); err != nil {
		t.Skipf("test model missing (%v); run `make testdata` or set GO_LLAMA_TEST_MODEL", err)
	}
	m, err := testInst.LoadModel(filepath.Base(p))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("Model.Close: %v", err)
		}
	})
	return m
}

func TestBuildInfo(t *testing.T) {
	info, err := testInst.BuildInfo()
	if err != nil {
		t.Fatalf("BuildInfo: %v", err)
	}
	// Exceptions are non-negotiable: llama.cpp throws, and a build without
	// them would abort instead of reporting a bad model.
	if !info.Exceptions {
		t.Error("the embedded engine was built without exception handling")
	}
	if !info.SIMD {
		t.Error("the embedded engine was built without SIMD")
	}
	t.Logf("build: simd=%v threads=%v exceptions=%v", info.SIMD, info.Threads, info.Exceptions)
}

func TestModelInfo(t *testing.T) {
	m := load(t)
	info, err := m.Info()
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if info.NVocab <= 0 || info.NEmbd <= 0 || info.NLayer <= 0 {
		t.Errorf("implausible model info: %+v", info)
	}
	if !info.HasDecoder {
		t.Error("a causal LM must report a decoder")
	}
	t.Logf("model: %s n_vocab=%d n_embd=%d n_layer=%d n_ctx_train=%d",
		info.Desc, info.NVocab, info.NEmbd, info.NLayer, info.NCtxTrain)
}

func TestTokenizeRoundTrip(t *testing.T) {
	m := load(t)
	const text = "Once upon a time"

	toks, err := m.Tokenize(text, true, true)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(toks) == 0 {
		t.Fatal("Tokenize returned no tokens")
	}

	back, err := m.Detokenize(toks, false)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	// The BOS token and the tokenizer's leading-space convention mean the
	// round trip is not byte-identical; containment is the real invariant.
	if !strings.Contains(back, "upon a time") {
		t.Errorf("round trip lost the text: got %q, want it to contain %q", back, "upon a time")
	}

	// Every token must render to something, and the pieces must concatenate
	// to what Detokenize produced.
	var joined strings.Builder
	for _, tok := range toks {
		piece, err := m.TokenToPiece(tok, false)
		if err != nil {
			t.Fatalf("TokenToPiece(%d): %v", tok, err)
		}
		joined.WriteString(piece)
	}
	if joined.String() != back {
		t.Errorf("pieces do not reassemble: %q vs %q", joined.String(), back)
	}
}

func TestGenerateGreedyIsDeterministic(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	// Temperature 0 selects greedy decoding, so two runs of the same prompt
	// must agree exactly — the check that catches sampler state leaking
	// between generations.
	params := llama.Params{NPredict: 16, Temperature: 0}
	first, err := ctx.Generate("Once upon a time", params)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if first.Text == "" {
		t.Fatal("Generate produced no text")
	}
	if first.NDecoded != 16 || first.Reason != llama.StopLength {
		t.Errorf("expected 16 tokens and reason %q, got %d and %q",
			llama.StopLength, first.NDecoded, first.Reason)
	}
	if len(first.Tokens) != first.NDecoded {
		t.Errorf("NDecoded=%d but %d tokens returned", first.NDecoded, len(first.Tokens))
	}
	t.Logf("generated: %q", first.Text)

	if err := ctx.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	second, err := ctx.Generate("Once upon a time", params)
	if err != nil {
		t.Fatalf("Generate (second): %v", err)
	}
	if first.Text != second.Text {
		t.Errorf("greedy generation is not reproducible:\n first: %q\nsecond: %q", first.Text, second.Text)
	}
}

func TestGenerateStopString(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	// Generate once to learn what this model actually produces, then ask for a
	// stop string taken from the middle of it — a fixed literal would make the
	// test depend on the model.
	base, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 16, Temperature: 0})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	fields := strings.Fields(base.Text)
	if len(fields) < 3 {
		t.Skipf("model produced too little text to derive a stop string: %q", base.Text)
	}
	stop := fields[1]

	if err := ctx.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	got, err := ctx.Generate("Once upon a time", llama.Params{
		NPredict: 16, Temperature: 0, Stop: []string{stop},
	})
	if err != nil {
		t.Fatalf("Generate with stop: %v", err)
	}
	if got.Reason != llama.StopString {
		t.Errorf("expected reason %q, got %q (text %q)", llama.StopString, got.Reason, got.Text)
	}
	if strings.HasSuffix(got.Text, stop) {
		t.Errorf("the matched stop string %q should have been trimmed from %q", stop, got.Text)
	}
	if got.NDecoded >= base.NDecoded {
		t.Errorf("stopping early should decode fewer tokens: %d vs %d", got.NDecoded, base.NDecoded)
	}
}

func TestStreamDeliversTextWhileGenerating(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	const want = 64
	var (
		got     strings.Builder
		pieces  []string
		firstAt time.Duration
	)
	start := time.Now()
	res, err := ctx.Stream("Once upon a time", llama.Params{NPredict: want, Temperature: 0},
		func(piece string) {
			if len(pieces) == 0 {
				firstAt = time.Since(start)
			}
			pieces = append(pieces, piece)
			got.WriteString(piece)
		})
	total := time.Since(start)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	// The streamed bytes must be exactly what the call returned: no dropped,
	// duplicated or reordered pieces.
	if got.String() != res.Text {
		t.Errorf("streamed text differs from the returned text:\n stream: %q\n return: %q", got.String(), res.Text)
	}
	// One callback per decoded token, so delivery is genuinely incremental
	// rather than one dump at the end.
	if len(pieces) != res.NDecoded {
		t.Errorf("expected one piece per decoded token: %d pieces, %d tokens", len(pieces), res.NDecoded)
	}
	if res.NDecoded != want {
		t.Errorf("expected %d tokens, got %d", want, res.NDecoded)
	}
	if firstAt >= total {
		t.Errorf("first piece arrived at %v, at the very end of a %v generation", firstAt, total)
	}
	t.Logf("%d pieces, first after %v, total %v", len(pieces), firstAt, total)
}

func TestStreamNilCallbackMatchesGenerate(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	params := llama.Params{NPredict: 12, Temperature: 0}
	want, err := ctx.Generate("Once upon a time", params)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if err := ctx.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	got, err := ctx.Stream("Once upon a time", params, nil)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if got.Text != want.Text {
		t.Errorf("Stream with no callback should equal Generate:\n got: %q\nwant: %q", got.Text, want.Text)
	}
}

func TestInterrupt(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	// Ask for far more tokens than the interrupt should let it produce, and
	// trip the flag from another goroutine while it runs.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(20 * time.Millisecond)
		if err := ctx.Interrupt(); err != nil {
			t.Errorf("Interrupt: %v", err)
		}
	}()

	res, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 400, Temperature: 0})
	wg.Wait()
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !res.Interrupted || res.Reason != llama.StopInterrupted {
		t.Errorf("expected an interrupted generation, got reason %q after %d tokens",
			res.Reason, res.NDecoded)
	}
	if res.NDecoded >= 400 {
		t.Errorf("interrupt did not stop generation early: %d tokens", res.NDecoded)
	}
	t.Logf("interrupted after %d tokens", res.NDecoded)
}

func TestChatTemplate(t *testing.T) {
	m := load(t)
	msgs := []llama.Message{
		{Role: "system", Content: "You are terse."},
		{Role: "user", Content: "Hello"},
	}

	// A model without a template must fail cleanly rather than inventing one.
	info, err := m.Info()
	if err != nil {
		t.Fatalf("Info: %v", err)
	}
	if info.ChatTemplate == "" {
		if _, err := m.ApplyChatTemplate(msgs, "", true); err == nil {
			t.Error("a model with no chat template should reject ApplyChatTemplate without an override")
		}
	}

	// With an override it must render, and the rendering must contain what
	// went in.
	prompt, err := m.ApplyChatTemplate(msgs, "chatml", true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate: %v", err)
	}
	for _, want := range []string{"You are terse.", "Hello", "assistant"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("rendered prompt %q is missing %q", prompt, want)
		}
	}
}

func TestEmbeddings(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256, Embeddings: true})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	info, err := m.Info()
	if err != nil {
		t.Fatalf("Info: %v", err)
	}

	vec, err := ctx.Embed("Once upon a time", true)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != info.NEmbd {
		t.Fatalf("embedding has %d dimensions, model reports n_embd=%d", len(vec), info.NEmbd)
	}
	// Normalised: the L2 norm must be 1 (within float tolerance), which also
	// proves the values are not all zero.
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("normalised embedding should have unit norm, got %v", sum)
	}
}

func TestErrorPaths(t *testing.T) {
	// A missing model must be a Go error, not a panic — this is the wasm
	// exception path, since llama.cpp throws on a bad file.
	if _, err := testInst.LoadModel("definitely-not-a-model.gguf"); err == nil {
		t.Error("loading a missing model should fail")
	}
	// The engine's filesystem is scoped to the model directory, so a path
	// outside it must not resolve.
	if _, err := testInst.LoadModel("../go.mod"); err == nil {
		t.Error("a path outside the preopened directory should not load")
	}

	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()
	// An empty prompt has two llama.cpp-compatible outcomes: models
	// whose tokenizer adds BOS generate unconditionally from it
	// (one prompt token), and BOS-less models (e.g. qwen2) tokenize
	// to nothing, which the engine rejects with a clean error rather
	// than decoding an empty batch.
	res, err := ctx.Generate("", llama.Params{NPredict: 4})
	switch {
	case err != nil && strings.Contains(err.Error(), "empty prompt"):
		// BOS-less model: clean rejection.
	case err != nil:
		t.Errorf("empty prompt failed with an unexpected error: %v", err)
	case res.NPrompt != 1 || res.Text == "":
		t.Errorf("expected 1 prompt token and some text, got NPrompt=%d text=%q", res.NPrompt, res.Text)
	}

	// A closed context must refuse the memory-poking calls rather than write
	// into a freed handle's address.
	other, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	if err := other.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := other.Interrupt(); err == nil {
		t.Error("Interrupt on a closed context should fail")
	}
	if _, err := other.Stream("hi", llama.Params{NPredict: 1}, func(string) {}); err == nil {
		t.Error("Stream on a closed context should fail")
	}
}

// Every handle is a raw C++ pointer, so a call after Close has to be refused
// in Go: passing it on would be a use-after-free inside the engine, which
// surfaces as corruption rather than an error.
func TestClosedHandlesAreRefused(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	if err := ctx.Close(); err != nil {
		t.Fatalf("Context.Close: %v", err)
	}
	// Closing twice is a no-op, not a double free.
	if err := ctx.Close(); err != nil {
		t.Errorf("second Context.Close: %v", err)
	}
	for name, call := range map[string]func() error{
		"Generate": func() error { _, err := ctx.Generate("hi", llama.Params{NPredict: 1}); return err },
		"Embed":    func() error { _, err := ctx.Embed("hi", true); return err },
		"Reset":    ctx.Reset,
	} {
		if err := call(); err == nil {
			t.Errorf("%s on a closed context should fail", name)
		}
	}

	// A context whose model has been closed is dangling too, so the model's
	// state has to be part of the check.
	live, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer live.Close()
	closed := load(t)
	orphan, err := closed.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	if err := orphan.Close(); err != nil {
		t.Fatalf("Context.Close: %v", err)
	}
	if err := closed.Close(); err != nil {
		t.Fatalf("Model.Close: %v", err)
	}
	for name, call := range map[string]func() error{
		"Info":              func() error { _, err := closed.Info(); return err },
		"Tokenize":          func() error { _, err := closed.Tokenize("hi", true, false); return err },
		"Detokenize":        func() error { _, err := closed.Detokenize([]int32{1}, false); return err },
		"TokenToPiece":      func() error { _, err := closed.TokenToPiece(1, false); return err },
		"ApplyChatTemplate": func() error { _, err := closed.ApplyChatTemplate(nil, "chatml", true); return err },
		"NewContext":        func() error { _, err := closed.NewContext(llama.ContextParams{}); return err },
	} {
		if err := call(); err == nil {
			t.Errorf("%s on a closed model should fail", name)
		}
	}
	// The other model's context is untouched by that.
	if _, err := live.Generate("Once upon a time", llama.Params{NPredict: 2}); err != nil {
		t.Errorf("closing one model broke another model's context: %v", err)
	}
}

func TestMultipleContextsShareOneModel(t *testing.T) {
	m := load(t)
	a, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("NewContext(a): %v", err)
	}
	defer a.Close()
	b, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatalf("NewContext(b): %v", err)
	}
	defer b.Close()

	params := llama.Params{NPredict: 8, Temperature: 0}
	ra, err := a.Generate("Once upon a time", params)
	if err != nil {
		t.Fatalf("Generate(a): %v", err)
	}
	rb, err := b.Generate("Once upon a time", params)
	if err != nil {
		t.Fatalf("Generate(b): %v", err)
	}
	// Independent KV caches, same weights and same greedy sampling: the two
	// contexts must produce the same text. A mismatch means state leaked
	// between them.
	if ra.Text != rb.Text {
		t.Errorf("contexts disagree:\n a: %q\n b: %q", ra.Text, rb.Text)
	}
}

func TestMultipleModels(t *testing.T) {
	// Two independent models in one engine: llama.cpp supports it, and the
	// handles must not collide.
	a, b := load(t), load(t)
	ia, err := a.Info()
	if err != nil {
		t.Fatalf("Info(a): %v", err)
	}
	ib, err := b.Info()
	if err != nil {
		t.Fatalf("Info(b): %v", err)
	}
	if ia != ib {
		t.Errorf("the same file loaded twice reports different metadata:\n a: %+v\n b: %+v", ia, ib)
	}
}
