package llama_test

import (
	"path/filepath"
	"sync"
	"testing"

	llama "github.com/goccy/go-llama"
)

// Two Llama instances are fully independent: each owns its own wasm module
// (own linear memory), so they load models and generate on separate
// goroutines concurrently. Greedy decoding is deterministic, so both
// instances running the same prompt must produce identical text — proving
// the instances neither share state nor interfere.
func TestInstancesAreIndependent(t *testing.T) {
	dir, err := filepath.Abs(filepath.Dir(modelPath()))
	if err != nil {
		t.Fatal(err)
	}
	name := filepath.Base(modelPath())

	newInst := func() (*llama.Llama, *llama.Context) {
		inst, err := llama.New(llama.WithPreopenDir(dir))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		m, err := inst.LoadModel(name)
		if err != nil {
			t.Fatalf("LoadModel: %v", err)
		}
		c, err := m.NewContext(llama.ContextParams{NCtx: 512})
		if err != nil {
			t.Fatalf("NewContext: %v", err)
		}
		return inst, c
	}

	instA, ctxA := newInst()
	defer instA.Close()
	instB, ctxB := newInst()
	defer instB.Close()

	const prompt = "Once upon a time"
	params := llama.Params{NPredict: 24, Temperature: 0}

	var wg sync.WaitGroup
	var resA, resB llama.Result
	var errA, errB error
	wg.Add(2)
	go func() { defer wg.Done(); resA, errA = ctxA.Generate(prompt, params) }()
	go func() { defer wg.Done(); resB, errB = ctxB.Generate(prompt, params) }()
	wg.Wait()

	if errA != nil {
		t.Fatalf("instance A generate: %v", errA)
	}
	if errB != nil {
		t.Fatalf("instance B generate: %v", errB)
	}
	if resA.Text != resB.Text {
		t.Fatalf("independent instances diverged on the same greedy prompt:\n A: %q\n B: %q", resA.Text, resB.Text)
	}
	if resA.Text == "" {
		t.Fatal("empty generation")
	}
}

// A second model can be loaded into the SAME instance (this is what
// speculative decoding needs — draft and target handles in one engine).
func TestMultipleModelsInOneInstance(t *testing.T) {
	dir, err := filepath.Abs(filepath.Dir(modelPath()))
	if err != nil {
		t.Fatal(err)
	}
	inst, err := llama.New(llama.WithPreopenDir(dir))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer inst.Close()

	name := filepath.Base(modelPath())
	m1, err := inst.LoadModel(name)
	if err != nil {
		t.Fatalf("first LoadModel: %v", err)
	}
	m2, err := inst.LoadModel(name)
	if err != nil {
		t.Fatalf("second LoadModel into the same instance: %v", err)
	}
	for _, m := range []*llama.Model{m1, m2} {
		if _, err := m.Info(); err != nil {
			t.Fatalf("model info: %v", err)
		}
	}
}
