package llama_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

var slotPrompts = []string{
	"Once upon a time",
	"The capital of France is",
	"def fibonacci(n):",
	"In the beginning",
	"A list of three fruits:",
	"The quick brown fox",
}

func newSlotsContext(t *testing.T, m *llama.Model, nSlots uint32) (*llama.Context, *llama.Slots) {
	t.Helper()
	// NCtx is split across the sequences (no KVUnified), so give each slot
	// room for the long tasks below.
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 2048, NSeqMax: nSlots})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	t.Cleanup(func() { ctx.Close() })
	slots, err := ctx.Slots()
	if err != nil {
		t.Fatalf("Slots: %v", err)
	}
	t.Cleanup(func() { slots.Close() })
	return ctx, slots
}

// Six greedy tasks on four slots — more tasks than slots, so the queue is
// exercised — must each produce exactly what Generate produces for the same
// prompt on a fresh context: the same text, tokens, counts and stop reason.
func TestSlotsMatchGenerate(t *testing.T) {
	m := load(t)
	params := llama.Params{NPredict: 12, Temperature: 0}

	alone := make([]llama.Result, len(slotPrompts))
	for i, p := range slotPrompts {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
		if err != nil {
			t.Fatalf("NewContext: %v", err)
		}
		res, err := ctx.Generate(p, params)
		ctx.Close()
		if err != nil {
			t.Fatalf("Generate(%q): %v", p, err)
		}
		alone[i] = res
	}

	for _, unified := range []bool{false, true} {
		t.Run(map[bool]string{false: "streams", true: "unified"}[unified], func(t *testing.T) {
			slotsMatchGenerate(t, m, unified, params, alone)
		})
	}
}

func slotsMatchGenerate(t *testing.T, m *llama.Model, unified bool, params llama.Params, alone []llama.Result) {
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 2048, NSeqMax: 4, KVUnified: unified})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()
	slots, err := ctx.Slots()
	if err != nil {
		t.Fatalf("Slots: %v", err)
	}
	defer slots.Close()
	cmpls := make([]*llama.TaskCompletion, len(slotPrompts))
	for i, p := range slotPrompts {
		c, err := slots.Post(context.Background(), llama.Task{Prompt: p, Params: params})
		if err != nil {
			t.Fatalf("Post(%q): %v", p, err)
		}
		if c.ID() != i+1 {
			t.Errorf("task %d posted with id %d", i+1, c.ID())
		}
		cmpls[i] = c
	}
	for i, c := range cmpls {
		res, err := c.Wait()
		if err != nil {
			t.Fatalf("task %d: Wait: %v", c.ID(), err)
		}
		want := alone[i]
		if res.Text != want.Text {
			t.Errorf("task %d (%q): text %q, Generate alone %q", c.ID(), slotPrompts[i], res.Text, want.Text)
		}
		if len(res.Tokens) != len(want.Tokens) {
			t.Errorf("task %d: %d tokens, Generate alone %d", c.ID(), len(res.Tokens), len(want.Tokens))
		} else {
			for j := range res.Tokens {
				if res.Tokens[j] != want.Tokens[j] {
					t.Errorf("task %d: token %d = %d, Generate alone %d", c.ID(), j, res.Tokens[j], want.Tokens[j])
					break
				}
			}
		}
		if res.Reason != want.Reason || res.NDecoded != want.NDecoded || res.NPrompt != want.NPrompt {
			t.Errorf("task %d: reason %q decoded %d prompt %d; Generate alone %q %d %d",
				c.ID(), res.Reason, res.NDecoded, res.NPrompt, want.Reason, want.NDecoded, want.NPrompt)
		}
		// The outputs are kept for late readers and concatenate to the text.
		var sb strings.Builder
		n := 0
		for out := range c.Outputs() {
			sb.WriteString(out.Text)
			n++
		}
		if sb.String() != res.Text || n != res.NDecoded {
			t.Errorf("task %d: %d outputs concatenate to %q, result text %q (%d tokens)", c.ID(), n, sb.String(), res.Text, res.NDecoded)
		}
	}
	st, err := slots.Status()
	if err != nil {
		t.Fatal(err)
	}
	if st.NSlots != 4 || st.Active != 0 || st.Queued != 0 || st.Used != 0 {
		t.Errorf("status after all tasks finished: %+v", st)
	}
}

// Outputs streams while the task runs; Result refuses until it has
// finished and then agrees with Wait.
func TestSlotsOutputsThenResult(t *testing.T) {
	m := load(t)
	_, slots := newSlotsContext(t, m, 2)
	c, err := slots.Post(context.Background(), llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 8, Temperature: 0}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Result(); !errors.Is(err, llama.ErrTaskRunning) {
		t.Errorf("Result before the task finished: err = %v, want ErrTaskRunning", err)
	}
	var texts []string
	for out := range c.Outputs() {
		texts = append(texts, out.Text)
	}
	res, err := c.Result()
	if err != nil {
		t.Fatalf("Result after Outputs ended: %v", err)
	}
	wres, werr := c.Wait()
	if werr != nil || wres.Text != res.Text {
		t.Errorf("Wait after Result: %q, %v; Result %q", wres.Text, werr, res.Text)
	}
	if len(texts) != 8 || strings.Join(texts, "") != res.Text {
		t.Errorf("%d outputs %q vs result %q", len(texts), strings.Join(texts, ""), res.Text)
	}
}

// Cancelling the Post's ctx drops the task: its slot is released, and the
// completion carries the text so far, StopInterrupted and ctx.Err().
func TestSlotsCancelViaContext(t *testing.T) {
	m := load(t)
	_, slots := newSlotsContext(t, m, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, err := slots.Post(ctx, llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 200, Temperature: 0}})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for out := range c.Outputs() {
		n++
		if out.Text == "" && out.Token == 0 {
			t.Errorf("empty output")
		}
		if n == 3 {
			cancel()
		}
	}
	res, err := c.Wait()
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Wait after cancel: err = %v, want context.Canceled", err)
	}
	if res.Reason != llama.StopInterrupted || !res.Interrupted || res.NDecoded < 3 || res.NDecoded > 200 {
		t.Errorf("cancelled result: %+v", res)
	}
	st, err := slots.Status()
	if err != nil {
		t.Fatal(err)
	}
	if st.Active != 0 || st.Used != 0 {
		t.Errorf("status after cancel: %+v", st)
	}

	// A deadline works the same way, on a task that never got a slot too.
	dctx, dcancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer dcancel()
	long := llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 400, Temperature: 0}}
	first, err := slots.Post(dctx, long)
	if err != nil {
		t.Fatal(err)
	}
	second, err := slots.Post(dctx, long)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []*llama.TaskCompletion{first, second} {
		if _, err := c.Wait(); !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("task %d: err = %v, want DeadlineExceeded", c.ID(), err)
		}
	}
}

// Close cancels every task and refuses further posts; the context stays
// usable for single-sequence work.
func TestSlotsClose(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 2048, NSeqMax: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	slots, err := ctx.Slots()
	if err != nil {
		t.Fatal(err)
	}
	long := llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 400, Temperature: 0}}
	var cmpls []*llama.TaskCompletion
	for i := 0; i < 3; i++ {
		c, err := slots.Post(context.Background(), long)
		if err != nil {
			t.Fatal(err)
		}
		cmpls = append(cmpls, c)
	}
	if err := slots.Close(); err != nil {
		t.Fatal(err)
	}
	for _, c := range cmpls {
		if _, err := c.Wait(); !errors.Is(err, llama.ErrSlotsClosed) {
			t.Errorf("task %d after Close: err = %v, want ErrSlotsClosed", c.ID(), err)
		}
	}
	if _, err := slots.Post(context.Background(), long); !errors.Is(err, llama.ErrSlotsClosed) {
		t.Errorf("Post after Close: err = %v, want ErrSlotsClosed", err)
	}
	if err := slots.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if _, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 4, Temperature: 0}); err != nil {
		t.Errorf("Generate after Slots.Close: %v", err)
	}
}

// While a task is held, the context's single-sequence methods refuse; once
// the slots are idle they work again.
func TestSlotsRefuseSingleSequenceCalls(t *testing.T) {
	m := load(t)
	ctx, slots := newSlotsContext(t, m, 2)
	pctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c, err := slots.Post(pctx, llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 400, Temperature: 0}})
	if err != nil {
		t.Fatal(err)
	}
	for range c.Outputs() {
		break // one output: the task holds a slot
	}
	if _, err := ctx.Generate("The", llama.Params{NPredict: 2, Temperature: 0}); err == nil {
		t.Error("Generate ran while a task held a slot")
	}
	if _, err := ctx.Score("The end"); err == nil {
		t.Error("Score ran while a task held a slot")
	}
	cancel()
	if _, err := c.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait: %v", err)
	}
	if _, err := ctx.Generate("The", llama.Params{NPredict: 2, Temperature: 0}); err != nil {
		t.Errorf("Generate after the task was cancelled: %v", err)
	}
}

// A task that can never fit the context is refused at Post.
func TestSlotsRefuseOversizedTask(t *testing.T) {
	m := load(t)
	_, slots := newSlotsContext(t, m, 2)
	_, err := slots.Post(context.Background(), llama.Task{Prompt: "Once upon a time", Params: llama.Params{NPredict: 100000, Temperature: 0}})
	if err == nil {
		t.Fatal("Post accepted a task larger than the context")
	}
	if _, err := slots.Post(context.Background(), llama.Task{Prompt: "x", Params: llama.Params{CachePrompt: true}}); err == nil {
		t.Fatal("Post accepted CachePrompt")
	}
}

// A system prompt is decoded once and shared: tasks whose prompt starts with
// it reuse its cells (NCached) and still produce exactly what Generate
// produces for the full prompt, in both KV layouts.
func TestSlotsSystemPrompt(t *testing.T) {
	m := load(t)
	const system = "You are a router. Answer with one word: the user's intent.\nUser:"
	tails := []string{" book a table for two", " cancel my order", " what is the weather"}
	params := llama.Params{NPredict: 8, Temperature: 0}
	alone := make([]llama.Result, len(tails))
	for i, tail := range tails {
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 512})
		if err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate(system+tail, params)
		ctx.Close()
		if err != nil {
			t.Fatal(err)
		}
		alone[i] = res
	}
	for _, unified := range []bool{false, true} {
		t.Run(map[bool]string{false: "streams", true: "unified"}[unified], func(t *testing.T) {
			ctx, err := m.NewContext(llama.ContextParams{NCtx: 2048, NSeqMax: 4, KVUnified: unified})
			if err != nil {
				t.Fatal(err)
			}
			defer ctx.Close()
			slots, err := ctx.Slots()
			if err != nil {
				t.Fatal(err)
			}
			defer slots.Close()
			if err := slots.SetSystemPrompt(system); err != nil {
				t.Fatalf("SetSystemPrompt: %v", err)
			}
			st, err := slots.Status()
			if err != nil {
				t.Fatal(err)
			}
			if st.NSlots != 3 {
				t.Errorf("system prompt should occupy one sequence: %+v", st)
			}
			var cmpls []*llama.TaskCompletion
			for _, tail := range tails {
				c, err := slots.Post(context.Background(), llama.Task{Prompt: system + tail, Params: params})
				if err != nil {
					t.Fatal(err)
				}
				cmpls = append(cmpls, c)
			}
			for i, c := range cmpls {
				res, err := c.Wait()
				if err != nil {
					t.Fatalf("task %d: %v", c.ID(), err)
				}
				if res.Text != alone[i].Text || res.NDecoded != alone[i].NDecoded {
					t.Errorf("task %d: %q (%d), Generate alone %q (%d)", c.ID(), res.Text, res.NDecoded, alone[i].Text, alone[i].NDecoded)
				}
				if res.NCached < 8 || res.NCached >= res.NPrompt {
					t.Errorf("task %d: NCached %d of %d prompt tokens: the system prompt was not reused", c.ID(), res.NCached, res.NPrompt)
				}
			}
			// A prompt that does not start with the system prompt decodes in full.
			c, err := slots.Post(context.Background(), llama.Task{Prompt: "Once upon a time", Params: params})
			if err != nil {
				t.Fatal(err)
			}
			res, err := c.Wait()
			if err != nil {
				t.Fatal(err)
			}
			if res.NCached != 0 {
				t.Errorf("unrelated prompt reused %d tokens", res.NCached)
			}
			if err := slots.SetSystemPrompt(""); err != nil {
				t.Fatalf("clear: %v", err)
			}
			if st, _ := slots.Status(); st.NSlots != 4 {
				t.Errorf("clearing the system prompt should release its sequence: %+v", st)
			}
		})
	}
}
