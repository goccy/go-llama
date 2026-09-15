package llama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"

	bridge "github.com/goccy/go-llama/internal"
)

// Task is what a Slots runs: a prompt with its sampling parameters. It is a
// plain value — Post does not modify it, and the same Task can be posted any
// number of times. Params.CachePrompt is not available to a task: every task
// decodes its own sequence.
type Task struct {
	Prompt string
	Params Params
}

// TaskOutput is one produced token: its id and the text it renders to. The
// texts of a task's outputs concatenate to the Text of its Result, except
// that a matched stop string is delivered as it is decoded and only
// afterwards trimmed from the result.
type TaskOutput struct {
	Token int32
	Text  string
}

// SlotsStatus is a snapshot of a Slots: how many slots there are and how many
// hold a task, how many tasks wait for one, and how much of the context's KV
// cache (NCtx cells) the running tasks hold.
type SlotsStatus struct {
	NSlots int `json:"n_slots"`
	Active int `json:"active"`
	Queued int `json:"queued"`
	NCtx   int `json:"n_ctx"`
	Used   int `json:"used"`
}

// ErrTaskRunning is returned by TaskCompletion.Result while the task has not
// finished.
var ErrTaskRunning = errors.New("llama: task is still running")

// ErrSlotsClosed is the error of every task a Slots dropped when it was
// closed, and of a Post after Close.
var ErrSlotsClosed = errors.New("llama: slots are closed")

// Slots runs tasks concurrently on one Context — continuous batching, after
// llama.cpp's server. The context holds as many slots as its
// ContextParams.NSeqMax, each running one task on its own KV sequence; one
// scheduling step decodes a single batch drawn from every busy slot, so the
// slots share the per-step cost instead of each paying it.
//
// A posted task waits for a free slot in FIFO order, and for enough free
// cells in the shared cache: its prompt plus its NPredict must fit next to
// what the running tasks may still write (an NPredict of zero means "until
// the context ends", which reserves the whole remainder).
//
// While a Slots holds tasks, the single-sequence methods of its Context
// (Generate, Stream, Score, ScoreChoices, Eval, Embed, SaveState, LoadState)
// refuse to run; Reset drops every task. Close cancels every task and stops
// the scheduler; the Context stays usable.
type Slots struct {
	c *Context

	// mu serializes every engine call this Slots makes on the context, and
	// guards tasks and closed.
	mu     sync.Mutex
	tasks  map[int]*TaskCompletion
	closed bool

	wake chan struct{} // nudged by Post: there is something to schedule
	stop chan struct{} // closed by Close
	// done is closed when the scheduler has returned, which it does only
	// once the Slots is finished with the context: every task ended, the
	// context's record of it cleared.
	done chan struct{}
}

// ErrSlotsRunning is returned by Context.Slots while an earlier Slots of
// the context is still open.
var ErrSlotsRunning = errors.New("llama: slots: the context already has a scheduler; close it first")

// Slots starts a scheduler over the context's slots. The context must have
// been created with NSeqMax set to the number of slots wanted (1 still
// works: tasks then run one at a time, each on its own sequence). A
// context runs one Slots at a time: a second call while the first is open
// returns ErrSlotsRunning. Closing the context closes its Slots.
func (c *Context) Slots() (*Slots, error) {
	release, err := c.enter("slots")
	if err != nil {
		return nil, err
	}
	defer release()
	c.st.slotsMu.Lock()
	defer c.st.slotsMu.Unlock()
	if c.st.slots != nil {
		return nil, ErrSlotsRunning
	}
	s := &Slots{
		c:     c,
		tasks: map[int]*TaskCompletion{},
		wake:  make(chan struct{}, 1),
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
	}
	c.st.slots = s
	go s.loop()
	return s, nil
}

// Post queues task and returns its completion. The task is cancelled when
// ctx is done: its slot is released, and its completion finishes with the
// text produced so far, StopInterrupted, and ctx.Err() as the error.
func (s *Slots) Post(ctx context.Context, task Task) (*TaskCompletion, error) {
	release, err := s.c.enter("slots post")
	if err != nil {
		return nil, err
	}
	defer release()
	if task.Params.CachePrompt {
		return nil, errors.New("llama: slots post: CachePrompt is not available to a task")
	}
	raw, err := json.Marshal(taskRequest{Prompt: task.Prompt, genRequest: task.Params.wire()})
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, ErrSlotsClosed
	}
	js, err := s.c.model.inst.e().LlamaCtxSlotsPost(s.c.h, string(raw), uint32(len(raw)))
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	var out struct {
		envelope
		ID int `json:"id"`
	}
	if err := decode("slots post", js, &out); err != nil {
		s.mu.Unlock()
		return nil, err
	}
	t := &TaskCompletion{id: out.ID, done: make(chan struct{})}
	t.cond = sync.NewCond(&t.mu)
	s.tasks[out.ID] = t
	s.mu.Unlock()
	s.nudge()
	if ctx.Done() != nil {
		go func() {
			select {
			case <-ctx.Done():
				s.cancel(t, ctx.Err())
			case <-t.done:
			}
		}()
	}
	return t, nil
}

// SetSystemPrompt decodes text once and shares its KV cells with every task
// whose prompt starts with it, so such a task decodes only the rest (its
// Result.NCached counts the reused tokens). This is the system prompt of
// llama.cpp's server of old: it occupies one of the context's sequences,
// leaving NSeqMax-1 slots for tasks, and with KVUnified the sharing is free
// (a copy is metadata); without it each task copies the cells, still far
// cheaper than decoding them. The text is tokenized on its own, so a task's
// prompt should be exactly this text followed by the rest; a prompt that
// tokenizes differently at the boundary simply decodes in full. Refused
// while tasks are held; an empty text clears it.
func (s *Slots) SetSystemPrompt(text string) error {
	release, err := s.c.enter("slots system prompt")
	if err != nil {
		return err
	}
	defer release()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrSlotsClosed
	}
	js, err := s.c.model.inst.e().LlamaCtxSlotsSystemPrompt(s.c.h, text, uint32(len(text)))
	if err != nil {
		return err
	}
	var out struct {
		envelope
		NTokens int `json:"n_tokens"`
	}
	return decode("slots system prompt", js, &out)
}

// Status reports the slots, the tasks and the cache cells in use.
func (s *Slots) Status() (SlotsStatus, error) {
	release, err := s.c.enter("slots status")
	if err != nil {
		return SlotsStatus{}, err
	}
	defer release()
	s.mu.Lock()
	defer s.mu.Unlock()
	js, err := s.c.model.inst.e().LlamaCtxSlotsStatus(s.c.h)
	if err != nil {
		return SlotsStatus{}, err
	}
	var out struct {
		envelope
		SlotsStatus
	}
	if err := decode("slots status", js, &out); err != nil {
		return SlotsStatus{}, err
	}
	return out.SlotsStatus, nil
}

// Close cancels every task (their completions finish with ErrSlotsClosed)
// and stops the scheduler. It returns once the scheduler has returned,
// which is after every task has ended; a concurrent second Close returns
// then too.
func (s *Slots) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		<-s.done
		return nil
	}
	s.closed = true
	s.mu.Unlock()
	close(s.stop)
	<-s.done
	return nil
}

// detach forgets the Slots on its context, so a new one can be started.
func (s *Slots) detach() {
	s.c.st.slotsMu.Lock()
	if s.c.st.slots == s {
		s.c.st.slots = nil
	}
	s.c.st.slotsMu.Unlock()
}

func (s *Slots) nudge() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// loop is the scheduler: one update per iteration while tasks are held,
// asleep otherwise. It owns the Slots' end, whichever way it comes: Close
// (the tasks are cancelled, with ErrSlotsClosed), or the engine gone
// under it — closed, or trapped, after which the guest's slots are in no
// state to step again (the tasks are failed with that error; there is
// nothing left to schedule on, and a Slots parked for a Close that may
// never come would hold the instance). Either way it clears the
// context's record of the Slots and then closes done.
func (s *Slots) loop() {
	defer close(s.done)
	defer s.detach()
	for {
		select {
		case <-s.stop:
			s.cancelAll(ErrSlotsClosed)
			return
		case <-s.wake:
		}
		for {
			select {
			case <-s.stop:
				s.cancelAll(ErrSlotsClosed)
				return
			default:
			}
			busy, err := s.update()
			if err != nil {
				if engineGone(err) {
					// Closed first, so no Post lands between the tasks
					// being failed and the Slots being over.
					s.mu.Lock()
					s.closed = true
					s.mu.Unlock()
					s.fail(err)
					return
				}
				s.fail(err)
				break
			}
			if !busy {
				break
			}
		}
	}
}

// cancelAll cancels every held task with err.
func (s *Slots) cancelAll(err error) {
	s.mu.Lock()
	pending := make([]*TaskCompletion, 0, len(s.tasks))
	for _, t := range s.tasks {
		pending = append(pending, t)
	}
	s.mu.Unlock()
	for _, t := range pending {
		s.cancel(t, err)
	}
}

// slotEvent is one entry of an update's events: a produced token (Token
// set), a finished task (Final set) or a failed one (Error set).
type slotEvent struct {
	ID    int    `json:"id"`
	Token *int32 `json:"token"`
	textFields
	Final *slotFinal `json:"final"`
	Error string     `json:"error"`
}

// slotFinal is the result object of a finished task: generate's result with
// its b64 copy of the text.
type slotFinal struct {
	Result
	B64 string `json:"b64"`
}

// update runs one scheduling step and delivers its events. busy reports
// whether any task is still held, so the caller knows to step again.
func (s *Slots) update() (busy bool, err error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return false, nil
	}
	js, err := s.c.model.inst.e().LlamaCtxSlotsUpdate(s.c.h)
	s.mu.Unlock()
	if err != nil {
		return false, err
	}
	var out struct {
		envelope
		Active int         `json:"active"`
		Queued int         `json:"queued"`
		Events []slotEvent `json:"events"`
	}
	if err := decode("slots update", js, &out); err != nil {
		return false, err
	}
	for _, ev := range out.Events {
		s.mu.Lock()
		t := s.tasks[ev.ID]
		s.mu.Unlock()
		if t == nil {
			continue // cancelled meanwhile
		}
		switch {
		case ev.Final != nil:
			res, err := finalResult(ev.Final)
			s.finish(t, res, err)
		case ev.Error != "":
			s.finish(t, Result{}, fmt.Errorf("llama: slots: task %d: %s", ev.ID, ev.Error))
		case ev.Token != nil:
			text, err := ev.textFields.bytes("slots update")
			if err != nil {
				s.finish(t, Result{}, err)
				continue
			}
			t.push(TaskOutput{Token: *ev.Token, Text: text})
		}
	}
	return out.Active > 0 || out.Queued > 0, nil
}

// finalResult turns a final event into a Result with its text taken from the
// lossless b64 copy.
func finalResult(f *slotFinal) (Result, error) {
	res := f.Result
	text, err := textFields{Text: f.Text, B64: f.B64}.bytes("slots update")
	if err != nil {
		return Result{}, err
	}
	res.Text = text
	return res, nil
}

// cancel drops a task in the engine and finishes its completion with err.
func (s *Slots) cancel(t *TaskCompletion, err error) {
	s.mu.Lock()
	if _, held := s.tasks[t.id]; !held {
		s.mu.Unlock()
		return
	}
	js, cerr := s.c.model.inst.e().LlamaCtxSlotsCancel(s.c.h, int32(t.id))
	s.mu.Unlock()
	if cerr != nil {
		if !engineGone(cerr) {
			// The engine declined: the task is no longer known to it,
			// having finished in an update racing this cancel, and that
			// update's final wins.
			return
		}
		// The engine itself is gone (closed, or trapped): nothing will
		// finish the task now but this. The caller's reason stays the
		// error, with the engine's attached.
		s.finish(t, Result{}, errors.Join(err, cerr))
		return
	}
	var out struct {
		envelope
		Queued bool       `json:"queued"`
		Final  *slotFinal `json:"final"`
	}
	if derr := decode("slots cancel", js, &out); derr != nil {
		s.finish(t, Result{}, derr)
		return
	}
	res := Result{Reason: StopInterrupted, Interrupted: true}
	if out.Final != nil {
		if r, ferr := finalResult(out.Final); ferr == nil {
			res = r
		}
	}
	s.finish(t, res, err)
}

// engineGone reports whether err says the engine can take no further
// call on the context: it is closed, or it trapped. Anything else — the
// guest's own error object, a malformed reply — leaves it running.
func engineGone(err error) bool {
	var trap *bridge.TrapError
	return errors.Is(err, ErrInstanceClosed) || errors.As(err, &trap)
}

// fail ends every held task with err when an update itself failed.
func (s *Slots) fail(err error) {
	s.mu.Lock()
	pending := make([]*TaskCompletion, 0, len(s.tasks))
	for _, t := range s.tasks {
		pending = append(pending, t)
	}
	s.mu.Unlock()
	for _, t := range pending {
		s.finish(t, Result{}, err)
	}
}

// finish completes t and forgets it.
func (s *Slots) finish(t *TaskCompletion, res Result, err error) {
	s.mu.Lock()
	delete(s.tasks, t.id)
	s.mu.Unlock()
	t.finish(res, err)
}

// TaskCompletion is a posted task: its outputs as they are produced, and its
// result once it has finished.
type TaskCompletion struct {
	id int

	mu      sync.Mutex
	cond    *sync.Cond
	outputs []TaskOutput
	ended   bool
	res     Result
	err     error
	done    chan struct{}
}

// ID is the task's id in the engine; ids start at 1 for each Context.
func (t *TaskCompletion) ID() int { return t.id }

// Outputs yields the task's outputs as they are produced and returns when
// the task has finished. It never blocks the scheduler: outputs are kept
// until read, so a slow reader only delays itself. Leaving the loop early
// stops nothing; cancel the Post's ctx to stop the task.
func (t *TaskCompletion) Outputs() iter.Seq[TaskOutput] {
	return func(yield func(TaskOutput) bool) {
		for i := 0; ; i++ {
			t.mu.Lock()
			for i >= len(t.outputs) && !t.ended {
				t.cond.Wait()
			}
			if i >= len(t.outputs) {
				t.mu.Unlock()
				return
			}
			out := t.outputs[i]
			t.mu.Unlock()
			if !yield(out) {
				return
			}
		}
	}
}

// Wait blocks until the task has finished and returns its result: the same
// Result a Generate of the task would return. A cancelled task returns the
// text produced so far with StopInterrupted and the cancellation error.
func (t *TaskCompletion) Wait() (Result, error) {
	<-t.done
	return t.res, t.err
}

// Result returns the finished task's result without waiting; ErrTaskRunning
// while it has not finished.
func (t *TaskCompletion) Result() (Result, error) {
	select {
	case <-t.done:
		return t.res, t.err
	default:
		return Result{}, ErrTaskRunning
	}
}

func (t *TaskCompletion) push(out TaskOutput) {
	t.mu.Lock()
	t.outputs = append(t.outputs, out)
	t.mu.Unlock()
	t.cond.Broadcast()
}

func (t *TaskCompletion) finish(res Result, err error) {
	t.mu.Lock()
	if t.ended {
		t.mu.Unlock()
		return
	}
	t.ended = true
	t.res, t.err = res, err
	t.mu.Unlock()
	t.cond.Broadcast()
	close(t.done)
}
