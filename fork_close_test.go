package llama_test

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// goroutines returns the goroutine count once it has settled at or below
// want, or the count it stuck at after a few seconds: a joined worker's
// goroutine exits a moment after the join returns.
func goroutines(want int) int {
	deadline := time.Now().Add(5 * time.Second)
	for {
		runtime.GC()
		n := runtime.NumGoroutine()
		if n <= want || time.Now().After(deadline) {
			return n
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// TestForkCloseStopsWorkers pins the teardown order of a fork and of the
// snapshot build: the ggml threadpool workers a context computes with are
// stopped and joined BEFORE the memory they run in goes away — a fork's on
// Instance.Close, the builder's before the capture. When they were not,
// every closed fork leaked its pool workers as parked goroutines (and the
// builder its own, once per snapshot), and a worker caught between a
// graph's last barrier and the futex park trapped its next atomic access
// against the zeroed memory bound and killed the whole process ("wasm:
// atomic access out of bounds") — near-deterministic under concurrent forks
// contending for cores, which is exactly this test's shape.
//
// A pool's workers are the only goroutines a fork or a build creates, and
// they are joined synchronously, so the count must return to the baseline
// exactly: one fork's pool is poolThreads-1 goroutines, and a tolerance of
// that size would pass a test that leaked a fork.
func TestForkCloseStopsWorkers(t *testing.T) {
	const prefix = "Once upon a time"
	runtime.GC()
	time.Sleep(100 * time.Millisecond)
	base := runtime.NumGoroutine()

	// The builder decodes the prefix on two threads: its one worker must be
	// joined by the build, not left parked in the image's memory for good.
	snap := buildTestSnapshot(t, prefix)
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked by the snapshot build: %d at baseline, %d after", base, n)
	}

	const workers, iters, poolThreads = 2, 8, 4
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				f, err := snap.Fork(poolThreads)
				if err != nil {
					t.Error(err)
					return
				}
				if _, err := f.Context("main").Generate(prefix+" there was a dragon in the hills", llama.Params{NPredict: 8, CachePrompt: true}); err != nil {
					t.Error(err)
					_ = f.Close()
					return
				}
				if err := f.Close(); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()

	// Unfixed, the 16 forks above leave 16 x (poolThreads-1) parked
	// goroutines behind.
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked across fork closes: %d at baseline, %d after %d forks", base, n, workers*iters)
	}
}

// TestForkCloseWithLiveSlots closes a fork while a Slots is still
// scheduling a task on its kept context: the scheduler must be stopped
// before the context is freed (a step on a freed context is a guest
// use-after-free, and one on a closed engine a nil-memory panic on the
// scheduler's goroutine, which nothing recovers), and the task must end
// with ErrSlotsClosed.
func TestForkCloseWithLiveSlots(t *testing.T) {
	const prefix = "Once upon a time"
	snap := buildTestSnapshot(t, prefix)
	runtime.GC()
	base := runtime.NumGoroutine()

	f, err := snap.Fork(2)
	if err != nil {
		t.Fatal(err)
	}
	c := f.Context("main")
	s, err := c.Slots()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Slots(); !errors.Is(err, llama.ErrSlotsRunning) {
		t.Fatalf("second Slots on the context: err = %v, want ErrSlotsRunning", err)
	}
	task, err := s.Post(context.Background(), llama.Task{Prompt: prefix + " there was", Params: llama.Params{NPredict: 200}})
	if err != nil {
		t.Fatal(err)
	}
	// Let the scheduler get into its stride before the close lands.
	for range task.Outputs() {
		break
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close with a live Slots: %v", err)
	}
	if _, err := task.Wait(); !errors.Is(err, llama.ErrSlotsClosed) {
		t.Fatalf("task after the fork closed: err = %v, want ErrSlotsClosed", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Slots.Close after the fork closed: %v", err)
	}
	if _, err := c.Generate(prefix, llama.Params{NPredict: 1}); err == nil {
		t.Fatal("Generate on the closed fork's context succeeded")
	}
	// The scheduler goroutine and the pool worker are gone with the fork.
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked: %d at baseline, %d after", base, n)
	}
}

// TestRegisterRejectsAliasedContext: a fork frees each kept context on
// Close, so one context under two names would be freed twice.
func TestRegisterRejectsAliasedContext(t *testing.T) {
	load(t)
	rejected := errors.New("the build rejected the alias")
	_, err := llama.NewSnapshot(func(b *llama.SnapshotBuilder) error {
		m, err := b.LoadModel(modelPath())
		if err != nil {
			return err
		}
		c, err := m.NewContext(llama.ContextParams{NCtx: 64})
		if err != nil {
			return err
		}
		if err := b.Register("main", c); err != nil {
			return err
		}
		if err := b.Register("alias", c); err == nil {
			return errors.New("registering the same context under a second name succeeded")
		}
		return rejected
	})
	if !errors.Is(err, rejected) {
		t.Fatalf("NewSnapshot: err = %v, want the build's own rejection", err)
	}
}

// TestForkCloseAfterTrap: a generation that traps on the main thread
// mid-graph (a failed assertion inside a decode does this) leaves the
// pool's workers waiting at a barrier for a pass that never comes. Closing
// the fork must still join them and return — with the engine's barrier
// deaf to the pool being stopped, this hung for ever under the engine
// lock — and the memory is unmapped only once they are gone.
func TestForkCloseAfterTrap(t *testing.T) {
	const prefix = "Once upon a time"
	snap := buildTestSnapshot(t, prefix)
	runtime.GC()
	base := runtime.NumGoroutine()

	f, err := snap.Fork(4)
	if err != nil {
		t.Fatal(err)
	}
	c := f.Context("main")
	if err := llama.TrapNextGraph(c); err != nil {
		t.Fatal(err)
	}
	_, err = c.Generate(prefix+" there was", llama.Params{NPredict: 4, CachePrompt: true})
	if !llama.IsTrap(err) {
		t.Fatalf("Generate after arming the trap: err = %v, want an engine trap", err)
	}
	// An engine that trapped is only good for tearing down: the frames the
	// trap unwound ran no destructors, so whatever they held is as it was.
	if _, err := c.Generate(prefix, llama.Params{NPredict: 1}); !llama.IsTrap(err) {
		t.Fatalf("Generate after the trap: err = %v, want a refusal naming the trap", err)
	}
	closed := make(chan error, 1)
	go func() { closed <- f.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close after the trap: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Close after the trap did not return: the pool's workers were not released")
	}
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked after the trapped fork closed: %d at baseline, %d after", base, n)
	}
}

// TestForkWithoutRoomForWorkersFailsCleanly caps a fork's memory at the
// image itself and asks for more worker stacks than any free heap of the
// image can hold, so a spawn fails partway through creating the pool:
// Fork must fail with the engine's own error (not a trap: the engine
// joins and frees the workers it did spawn), leave no worker behind, and
// release the fork's memory.
func TestForkWithoutRoomForWorkersFailsCleanly(t *testing.T) {
	const prefix = "Once upon a time"
	probe := buildTestSnapshot(t, prefix)
	size := llama.SnapshotImageSize(probe)
	snap := buildTestSnapshotWith(t, prefix, llama.WithMaxMemory(size))
	runtime.GC()
	base := runtime.NumGoroutine()

	// 63 worker stacks are megabytes; the tiny image's free heap is not.
	f, err := snap.Fork(64)
	if err == nil {
		_ = f.Close()
		t.Fatal("Fork found room for 63 worker stacks in a memory capped at the image: the cap did not bite")
	}
	if !llama.IsGuestError(err) {
		t.Fatalf("Fork with no room for its workers: err = %v, want the engine's own error (the pool refused), not %T", err, err)
	}
	if errors.Is(err, llama.ErrMemoryLeaked) {
		t.Fatalf("Fork with no room for its workers leaked the fork: %v", err)
	}
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked by the failed fork: %d at baseline, %d after", base, n)
	}
}

// TestSnapshotBuildPanicJoinsWorkers: a build callback that panics after
// creating a threaded context still has that context's workers joined
// before the image's memory goes away; NewSnapshot reports the panic as
// an error and nothing leaks.
func TestSnapshotBuildPanicJoinsWorkers(t *testing.T) {
	load(t)
	runtime.GC()
	base := runtime.NumGoroutine()
	_, err := llama.NewSnapshot(func(b *llama.SnapshotBuilder) error {
		m, err := b.LoadModel(modelPath())
		if err != nil {
			return err
		}
		if _, err := m.NewContext(llama.ContextParams{NCtx: 64, NThreads: 4}); err != nil {
			return err
		}
		panic("the build panics")
	})
	if err == nil {
		t.Fatal("NewSnapshot with a panicking build succeeded")
	}
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked by the panicking build: %d at baseline, %d after", base, n)
	}
}

// TestSnapshotRejectsKeptContextClosedByBuild: a kept context the build
// closed (here through its model's Close) must not be sealed into the
// image as a freed handle.
func TestSnapshotRejectsKeptContextClosedByBuild(t *testing.T) {
	load(t)
	_, err := llama.NewSnapshot(func(b *llama.SnapshotBuilder) error {
		m, err := b.LoadModel(modelPath())
		if err != nil {
			return err
		}
		c, err := m.NewContext(llama.ContextParams{NCtx: 64})
		if err != nil {
			return err
		}
		if err := b.Register("main", c); err != nil {
			return err
		}
		return m.Close()
	})
	if err == nil {
		t.Fatal("NewSnapshot sealed a kept context the build had closed")
	}
}
