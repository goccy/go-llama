package llama_test

import (
	"path/filepath"
	"testing"

	llama "github.com/goccy/go-llama"
)

// newSharedTestInst brings up a fresh instance scoped to the test model's
// directory — the multi-instance pattern the copy-on-write machinery exists
// for, as opposed to the suite-wide testInst.
func newSharedTestInst(t *testing.T) *llama.Llama {
	t.Helper()
	dir, err := filepath.Abs(filepath.Dir(modelPath()))
	if err != nil {
		t.Fatal(err)
	}
	inst, err := llama.New(llama.WithPreopenDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	return inst
}

func greedy24(t *testing.T, m *llama.Model) string {
	t.Helper()
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	res, err := ctx.Generate("The quick brown fox", llama.Params{NPredict: 24, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text == "" {
		t.Fatal("empty generation")
	}
	return res.Text
}

// TestSharedModelSnapshot pins the copy-on-write sharing contract: instances
// that load the same model with the same options share one loaded image, and
// each instance still generates independently and identically — including
// after the instance that populated the snapshot is gone.
func TestSharedModelSnapshot(t *testing.T) {
	name := filepath.Base(modelPath())

	l1 := newSharedTestInst(t)
	m1, err := l1.LoadModel(name)
	if err != nil {
		t.Fatal(err)
	}
	want := greedy24(t, m1)

	l2 := newSharedTestInst(t)
	m2, err := l2.LoadModel(name)
	if err != nil {
		t.Fatal(err)
	}
	if !llama.EngineImageBacked(l1) || !llama.EngineImageBacked(l2) {
		t.Errorf("instances are not image-backed (l1=%v l2=%v) — the model snapshot was not shared",
			llama.EngineImageBacked(l1), llama.EngineImageBacked(l2))
	}
	if got := greedy24(t, m2); got != want {
		t.Errorf("shared-model instance diverges from the loader:\n  got  %q\n  want %q", got, want)
	}

	// The first instance's teardown must not disturb the second: the image is
	// process-owned, not owned by whoever happened to populate it.
	if err := m1.Close(); err != nil {
		t.Fatal(err)
	}
	if err := l1.Close(); err != nil {
		t.Fatal(err)
	}
	if got := greedy24(t, m2); got != want {
		t.Errorf("survivor instance diverges after the loader closed:\n  got  %q\n  want %q", got, want)
	}
	if err := m2.Close(); err != nil {
		t.Fatal(err)
	}
	if err := l2.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestCloseIsSafeNotFatal pins the failure mode of use-after-close: an error,
// never a fault on unmapped memory.
func TestCloseIsSafeNotFatal(t *testing.T) {
	l := newSharedTestInst(t)
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if err := l.Close(); err != nil { // idempotent
		t.Fatal(err)
	}
	if _, err := l.BuildInfo(); err == nil {
		t.Error("BuildInfo on a closed instance succeeded; want an error")
	}
}
