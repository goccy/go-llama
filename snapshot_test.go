package llama_test

import (
	"strings"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestSnapshotFork checks the prepare-once / fork-per-request path: a fork
// starts with the builder's context — prefix cached, no re-decode — and
// generates exactly what a conventionally built instance generates; a fork
// that skips the threadpool (nThreads 1) still works; forks are independent.
func TestSnapshotFork(t *testing.T) {
	prefix := "Once upon a time, in a land far away, there lived a little girl who loved"
	const suffix = " to sing"

	// Baseline: a conventional instance, greedy.
	inst := load(t)
	ctx, err := inst.NewContext(llama.ContextParams{NCtx: 512, NThreads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	want, err := ctx.Generate(prefix+suffix, llama.Params{NPredict: 8, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}

	snap, err := llama.NewSnapshot(func(b *llama.SnapshotBuilder) error {
		m, err := b.LoadModel(modelPath())
		if err != nil {
			return err
		}
		c, err := m.NewContext(llama.ContextParams{NCtx: 512, NThreads: 2})
		if err != nil {
			return err
		}
		if _, err := c.Generate(prefix, llama.Params{NPredict: 1, CachePrompt: true, Temperature: 0}); err != nil {
			return err
		}
		return b.Register("main", c)
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, nThreads := range []uint32{2, 1} {
		f, err := snap.Fork(nThreads)
		if err != nil {
			t.Fatalf("Fork(%d): %v", nThreads, err)
		}
		c := f.Context("main")
		if c == nil {
			t.Fatal("kept context missing in fork")
		}
		res, err := c.Generate(prefix+suffix, llama.Params{NPredict: 8, CachePrompt: true, Temperature: 0})
		if err != nil {
			t.Fatalf("fork(%d) generate: %v", nThreads, err)
		}
		if res.NCached == 0 {
			t.Fatalf("fork(%d): prefix not cached (n_cached=0)", nThreads)
		}
		if res.Text != want.Text {
			t.Fatalf("fork(%d) text = %q, want %q", nThreads, res.Text, want.Text)
		}
		if err := f.Close(); err != nil {
			t.Fatalf("close fork: %v", err)
		}
	}

	// Two live forks answer independently.
	f1, err := snap.Fork(2)
	if err != nil {
		t.Fatal(err)
	}
	defer f1.Close()
	f2, err := snap.Fork(2)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	r1, err := f1.Context("main").Generate(prefix+" to dance", llama.Params{NPredict: 8, CachePrompt: true, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := f2.Context("main").Generate(prefix+suffix, llama.Params{NPredict: 8, CachePrompt: true, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if r2.Text != want.Text {
		t.Fatalf("fork2 text = %q, want %q", r2.Text, want.Text)
	}
	if strings.TrimSpace(r1.Text) == "" {
		t.Fatal("fork1 produced nothing")
	}
}
