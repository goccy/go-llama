package llama_test

import (
	"context"
	"errors"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// newTestInstance brings up a private instance over the test model with
// one context of nThreads threads.
func newTestInstance(t *testing.T, nThreads uint32, p llama.ContextParams) (*llama.Llama, *llama.Context) {
	t.Helper()
	load(t) // model presence
	dir, err := filepath.Abs(filepath.Dir(modelPath()))
	if err != nil {
		t.Fatal(err)
	}
	inst, err := llama.New(llama.WithPreopenDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	m, err := inst.LoadModel(filepath.Base(modelPath()))
	if err != nil {
		t.Fatal(err)
	}
	p.NThreads = nThreads
	c, err := m.NewContext(p)
	if err != nil {
		t.Fatal(err)
	}
	return inst, c
}

// TestCloseFreesOpenContexts: an instance closed with a context still open
// frees that context first — joining its pool workers — rather than
// unmapping the memory under them, and a Slots scheduling on it ends its
// tasks with ErrSlotsClosed.
func TestCloseFreesOpenContexts(t *testing.T) {
	runtime.GC()
	base := runtime.NumGoroutine()
	inst, c := newTestInstance(t, 4, llama.ContextParams{NCtx: 512, NSeqMax: 2})
	s, err := c.Slots()
	if err != nil {
		t.Fatal(err)
	}
	task, err := s.Post(context.Background(), llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 200}})
	if err != nil {
		t.Fatal(err)
	}
	for range task.Outputs() {
		break
	}
	if err := inst.Close(); err != nil {
		t.Fatalf("Close with an open context and a live Slots: %v", err)
	}
	if _, err := task.Wait(); !errors.Is(err, llama.ErrSlotsClosed) {
		t.Fatalf("task after Close: err = %v, want ErrSlotsClosed", err)
	}
	if _, err := c.Generate("x", llama.Params{NPredict: 1}); err == nil {
		t.Fatal("Generate on a context of a closed instance succeeded")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Context.Close after the instance closed: %v", err)
	}
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked: %d at baseline, %d after", base, n)
	}
}

// TestCloseWaitsForCallInFlight: closing a context (or its instance) while
// a generation runs on it waits for the generation to return instead of
// freeing the context under it. The generation is held inside its piece
// callback until the test has seen Close not return.
func TestCloseWaitsForCallInFlight(t *testing.T) {
	inst, c := newTestInstance(t, 2, llama.ContextParams{NCtx: 512})
	defer inst.Close()
	started := make(chan struct{})
	resume := make(chan struct{})
	var once sync.Once
	type outcome struct {
		res llama.Result
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		res, err := c.Stream("Once upon a time", llama.Params{NPredict: 32}, func(string) {
			once.Do(func() {
				close(started)
				<-resume
			})
		})
		done <- outcome{res, err}
	}()
	<-started
	closed := make(chan error, 1)
	go func() { closed <- c.Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned while the generation was running")
	case <-time.After(200 * time.Millisecond):
	}
	close(resume)
	out := <-done
	if out.err != nil {
		t.Fatalf("generation: %v", out.err)
	}
	if out.res.NDecoded == 0 {
		t.Fatal("generation produced nothing")
	}
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := c.Generate("x", llama.Params{NPredict: 1}); err == nil {
		t.Fatal("Generate after Close succeeded")
	}
}

// TestConcurrentCloseWaitsForTeardown verifies that a second instance or
// model Close has the same completion semantics as the first: it does not
// return while the first closer is still waiting for an in-flight call.
func TestConcurrentCloseWaitsForTeardown(t *testing.T) {
	for _, target := range []string{"instance", "model"} {
		t.Run(target, func(t *testing.T) {
			inst, c := newTestInstance(t, 2, llama.ContextParams{NCtx: 512})
			closer := inst.Close
			if target == "model" {
				defer inst.Close()
				closer = c.Model().Close
			}
			started := make(chan struct{})
			resume := make(chan struct{})
			resumeOpen := true
			defer func() {
				if resumeOpen {
					close(resume)
				}
			}()
			var once sync.Once
			generated := make(chan error, 1)
			go func() {
				_, err := c.Stream("Once upon a time", llama.Params{NPredict: 32}, func(string) {
					once.Do(func() {
						close(started)
						<-resume
					})
				})
				generated <- err
			}()
			<-started

			closed1 := make(chan error, 1)
			closed2 := make(chan error, 1)
			go func() { closed1 <- closer() }()
			go func() { closed2 <- closer() }()
			for i, ch := range []<-chan error{closed1, closed2} {
				select {
				case <-ch:
					t.Fatalf("Close %d returned while teardown was waiting for generation", i+1)
				case <-time.After(100 * time.Millisecond):
				}
			}
			close(resume)
			resumeOpen = false
			if err := <-generated; err != nil {
				t.Fatalf("generation: %v", err)
			}
			if err := <-closed1; err != nil {
				t.Fatalf("first Close: %v", err)
			}
			if err := <-closed2; err != nil {
				t.Fatalf("second Close: %v", err)
			}
		})
	}
}

// TestModelCloseFreesOpenContexts: closing a model closes its open
// contexts first (a context must not outlive its model), joining their
// workers; calls on them fail afterwards, and the instance's Close finds
// nothing left to do.
func TestModelCloseFreesOpenContexts(t *testing.T) {
	runtime.GC()
	base := runtime.NumGoroutine()
	inst, c := newTestInstance(t, 4, llama.ContextParams{NCtx: 256})
	m := c.Model()
	if err := m.Close(); err != nil {
		t.Fatalf("Model.Close with an open context: %v", err)
	}
	if _, err := c.Generate("x", llama.Params{NPredict: 1}); err == nil {
		t.Fatal("Generate on a context of a closed model succeeded")
	}
	if n := goroutines(base); n > base {
		t.Fatalf("goroutines leaked: %d at baseline, %d after", base, n)
	}
	if err := inst.Close(); err != nil {
		t.Fatalf("Close after Model.Close: %v", err)
	}
	if _, err := c.Generate("x", llama.Params{NPredict: 1}); !errors.Is(err, llama.ErrInstanceClosed) {
		t.Fatalf("Generate after the instance closed: err = %v, want ErrInstanceClosed", err)
	}
}

// TestDraftCloseInterruptsSpeculativeGeneration: closing the draft context
// of a running GenerateWithDraft ends that generation (it polls both
// contexts' interrupt flags) instead of waiting it out, and the target
// stays usable.
func TestDraftCloseInterruptsSpeculativeGeneration(t *testing.T) {
	inst, target := newTestInstance(t, 1, llama.ContextParams{NCtx: 1024})
	defer inst.Close()
	draft, err := target.Model().NewContext(llama.ContextParams{NCtx: 1024})
	if err != nil {
		t.Fatal(err)
	}
	type outcome struct {
		res llama.Result
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		res, err := target.GenerateWithDraft(draft, "Once upon a time", llama.Params{NPredict: 800}, 4)
		done <- outcome{res, err}
	}()
	time.Sleep(20 * time.Millisecond)
	if err := draft.Close(); err != nil {
		t.Fatalf("draft Close during speculative generation: %v", err)
	}
	out := <-done
	if out.err != nil {
		t.Fatalf("speculative generation: %v", out.err)
	}
	if out.res.Reason != llama.StopInterrupted || out.res.NDecoded >= 800 {
		t.Fatalf("speculative generation stopped for %v after %d tokens, want StopInterrupted before 800", out.res.Reason, out.res.NDecoded)
	}
	if _, err := target.Generate("x", llama.Params{NPredict: 1}); err != nil {
		t.Fatalf("target after the draft closed: %v", err)
	}
}

// TestInterruptStopsSpeculativePrefillWithoutToken pins interruption between
// prompt chunks: once prefill observes the flag, speculative generation must
// not sample a token from the partial prompt.
func TestInterruptStopsSpeculativePrefillWithoutToken(t *testing.T) {
	inst, target := newTestInstance(t, 1, llama.ContextParams{NCtx: 1024, NBatch: 64})
	defer inst.Close()
	draft, err := target.Model().NewContext(llama.ContextParams{NCtx: 1024, NBatch: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Close()

	type outcome struct {
		res llama.Result
		err error
	}
	done := make(chan outcome, 1)
	prompt := strings.Repeat("Once upon a time in a land far away, ", 40)
	go func() {
		res, err := target.GenerateWithDraft(draft, prompt, llama.Params{NPredict: 8}, 4)
		done <- outcome{res: res, err: err}
	}()

	ticker := time.NewTicker(100 * time.Microsecond)
	defer ticker.Stop()
	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	for {
		select {
		case out := <-done:
			if out.err != nil {
				t.Fatalf("speculative generation: %v", out.err)
			}
			if out.res.Reason != llama.StopInterrupted || !out.res.Interrupted {
				t.Fatalf("generation stopped for %v (interrupted=%v), want StopInterrupted", out.res.Reason, out.res.Interrupted)
			}
			if out.res.NDecoded != 0 || len(out.res.Tokens) != 0 || out.res.Text != "" {
				t.Fatalf("interrupted prefill produced %d tokens and text %q", out.res.NDecoded, out.res.Text)
			}
			return
		case <-ticker.C:
			if err := target.Interrupt(); err != nil {
				t.Fatalf("interrupt: %v", err)
			}
		case <-deadline.C:
			t.Fatal("speculative prefill did not stop")
		}
	}
}
