package llama_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// TestGenerateSurvivesGCPressure pins the engine's stop-the-world
// behavior: a multi-threaded generate must make progress while the Go
// GC constantly stops the world. Engine bundles before llamawasm2go
// v0.2.3 livelocked here — ggml's barrier spin-waits were captured
// into assembly the runtime cannot async-preempt, so a worker spinning
// at a barrier blocked every stop-the-world while the worker it waited
// for stayed parked, and the process burned every core forever. The
// engine's transpiler now plants a bounded preemption point in those
// spins, so the run below completes in well under a second of work.
//
// The scenario runs in a CHILD process with the deadline held by the
// parent, because the failure mode freezes the child's entire runtime:
// no goroutine runs during the wedged stop-the-world, so in-process
// timers — including go test's own -test.timeout — never fire. Only an
// external observer can turn the hang into a failure. The deadline is
// deliberately generous (the GOAMD64=v1 scalar fallback is slow); a
// healthy engine finishes orders of magnitude sooner, and a wedged one
// never finishes at all.
func TestGenerateSurvivesGCPressure(t *testing.T) {
	if os.Getenv("GO_LLAMA_GC_STRESS_CHILD") == "1" {
		gcStressChild(t)
		return
	}

	if _, err := os.Stat(modelPath()); err != nil {
		t.Skipf("test model missing (%v); run `make testdata` or set GO_LLAMA_TEST_MODEL", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0],
		"-test.run", "^TestGenerateSurvivesGCPressure$", "-test.v")
	cmd.Env = append(os.Environ(), "GO_LLAMA_GC_STRESS_CHILD=1")
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		t.Fatalf("stop-the-world livelock: the GC-pressure generate did not finish "+
			"within 90s and was killed from outside — the engine's spin-waits are "+
			"not preemptible\nchild output:\n%s", out)
	}
	if err != nil {
		t.Fatalf("GC-pressure child failed: %v\n%s", err, out)
	}
}

// gcStressChild is the in-child half: hammer runtime.GC from one
// goroutine — every cycle is a stop-the-world request racing the
// engine's barrier spins — while a threaded context generates.
func gcStressChild(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256, NThreads: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()

	res, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 32, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.NDecoded == 0 {
		t.Fatal("generate under GC pressure decoded no tokens")
	}
	fmt.Printf("generated %d tokens under GC pressure\n", res.NDecoded)
}
