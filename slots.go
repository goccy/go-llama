package llama

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"
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
	done chan struct{} // closed when the scheduler has returned
}

// Slots starts a scheduler over the context's slots. The context must have
// been created with NSeqMax set to the number of slots wanted (1 still
// works: tasks then run one at a time, each on its own sequence).
func (c *Context) Slots() (*Slots, error) {
	if err := c.use("slots"); err != nil {
		return nil, err
	}
	s := &Slots{
		c:     c,
		tasks: map[int]*TaskCompletion{},
		wake:  make(chan struct{}, 1),
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
	}
	go s.loop()
	return s, nil
}

// Post queues task and returns its completion. The task is cancelled when
// ctx is done: its slot is released, and its completion finishes with the
// text produced so far, StopInterrupted, and ctx.Err() as the error.
func (s *Slots) Post(ctx context.Context, task Task) (*TaskCompletion, error) {
	if err := s.c.use("slots post"); err != nil {
		return nil, err
	}
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

// Status reports the slots, the tasks and the cache cells in use.
func (s *Slots) Status() (SlotsStatus, error) {
	if err := s.c.use("slots status"); err != nil {
		return SlotsStatus{}, err
	}
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
// and stops the scheduler. It returns once the scheduler has returned.
func (s *Slots) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		<-s.done
		return nil
	}
	s.closed = true
	pending := make([]*TaskCompletion, 0, len(s.tasks))
	for _, t := range s.tasks {
		pending = append(pending, t)
	}
	s.mu.Unlock()
	close(s.stop)
	<-s.done
	for _, t := range pending {
		s.cancel(t, ErrSlotsClosed)
	}
	return nil
}

func (s *Slots) nudge() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// loop is the scheduler: one update per iteration while tasks are held,
// asleep otherwise.
func (s *Slots) loop() {
	defer close(s.done)
	for {
		select {
		case <-s.stop:
			return
		case <-s.wake:
		}
		for {
			select {
			case <-s.stop:
				return
			default:
			}
			busy, err := s.update()
			if err != nil {
				s.fail(err)
				break
			}
			if !busy {
				break
			}
		}
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
		// The task is no longer known to the engine: it finished in an
		// update racing this cancel, and that update's final wins.
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
