package llama_test

import (
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// TestThreadScaling pins that a context created with NThreads > 1 really
// computes on several cores: ggml's graph plan must keep the requested
// thread count on the wasip1-threads build (upstream llama.cpp clamps
// every __wasi__ target to one thread; patches/wasm-threads-graph-plan.patch
// in llama-wasm lifts that for the threads target). The check is CPU
// time against wall time over a long prompt evaluation: workers that
// never run leave the ratio at about one whatever NThreads says.
func TestThreadScaling(t *testing.T) {
	if runtime.NumCPU() < 4 {
		t.Skip("needs at least 4 CPUs")
	}
	m := load(t)
	prompt := strings.Repeat("Once upon a time there was a little girl who liked to play outside. ", 24)

	cpuRatio := func(threads uint32) float64 {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 1024, NBatch: 1024, NUBatch: 512, NThreads: threads})
		if err != nil {
			t.Fatalf("NewContext(threads=%d): %v", threads, err)
		}
		defer ctx.Close()
		if _, err := ctx.Eval(prompt, true, false); err != nil { // warm-up
			t.Fatal(err)
		}
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
		var before, after syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &before); err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		const reps = 3
		for i := 0; i < reps; i++ {
			if _, err := ctx.Eval(prompt, true, false); err != nil {
				t.Fatal(err)
			}
			if err := ctx.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		wall := time.Since(t0)
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &after); err != nil {
			t.Fatal(err)
		}
		cpu := time.Duration(after.Utime.Nano()-before.Utime.Nano()) + time.Duration(after.Stime.Nano()-before.Stime.Nano())
		ratio := cpu.Seconds() / wall.Seconds()
		t.Logf("threads=%d: wall %v cpu %v ratio %.2f", threads, wall/reps, cpu/reps, ratio)
		return ratio
	}
	one := cpuRatio(1)
	four := cpuRatio(4)
	if four < 1.8 {
		t.Errorf("NThreads=4 used %.2f CPUs (NThreads=1: %.2f): the graph is computed on one thread", four, one)
	}
}
