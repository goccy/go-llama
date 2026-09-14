package llama

// Streaming and interruption — the two things that have to reach a generation
// that is ALREADY running.
//
// They reach it from opposite directions, because the engine is one translated
// module with one C stack: the goroutine inside Generate is the only one that
// can be there, so nothing else may call in.
//
//   - Streaming goes outward. The engine calls the host once per decoded
//     token, on the generating goroutine, so the text is handed over with no
//     concurrency at all.
//   - Interruption has to go inward, and a call is impossible, so it writes
//     one aligned word in linear memory through base.AccessMemory — the same
//     idiom the sibling wasm2go embeddings use for their interrupt flags.

import (
	"fmt"
	"runtime"

	bridge "github.com/goccy/go-llama/internal"
)

// Stream generates from prompt like Generate and calls onPiece with each piece
// of text as it is decoded, then returns the same complete Result.
//
// onPiece runs inside the generation, on the goroutine that called Stream, so
// it must not call back into this context or into any other part of the
// engine. Keep it short: the generation is stopped while it runs.
//
// The pieces concatenate to Result.Text, with one exception: a stop string is
// delivered as it is decoded and only then trimmed from Result.Text, so with
// Params.Stop set the stream can run a few characters past the returned text.
//
// A nil onPiece makes this exactly Generate.
func (c *Context) Stream(prompt string, p Params, onPiece func(string)) (Result, error) {
	if onPiece == nil {
		return c.Generate(prompt, p)
	}
	// generate holds the context open for the generation; this is only the
	// early answer for a closed context, before a sink is installed.
	if err := c.use("stream"); err != nil {
		return Result{}, err
	}
	ps := &pieceSink{onPiece: onPiece}
	sink, err := c.model.inst.e().NewTokenSink(ps)
	if err != nil {
		return Result{}, fmt.Errorf("llama: stream: install sink: %w", err)
	}
	res, genErr := c.generate(prompt, p.wire(), sink)
	// The sink's C++ trampoline is released by its finalizer; keep it alive
	// until the generation that dispatches into it has returned.
	runtime.KeepAlive(sink)
	// Whatever is still held back is all there will ever be, so hand it over
	// even if it is a truncated sequence: dropping it would lose bytes that
	// Result.Text still has.
	ps.flush()
	return res, genErr
}

// pieceSink adapts a Go func to the generated callback interface. The engine
// calls it once per decoded token, from inside the generation, so no locking
// is needed around held.
type pieceSink struct {
	bridge.Token_SinkCallbackDefaults
	onPiece func(string)
	// held is a trailing multi-byte sequence the last token cut in half; it
	// is emitted once the token that completes it arrives. See utf8.go.
	held []byte
}

func (s *pieceSink) OnPiece(piece string) error {
	s.held = append(s.held, piece...)
	n := completeUTF8Prefix(s.held)
	if n > 0 {
		s.onPiece(string(s.held[:n]))
		s.held = append(s.held[:0], s.held[n:]...)
	}
	return nil
}

func (s *pieceSink) flush() {
	if len(s.held) > 0 {
		s.onPiece(string(s.held))
		s.held = s.held[:0]
	}
}

// Interrupt stops a running generation at its next token WITHOUT executing any
// engine code: it writes the interrupt flag straight into linear memory.
// Calling in instead would need the lock the generation holds, and would share
// the C stack with it. Generate then returns what it has, with
// Reason == StopInterrupted.
//
// Safe to call from any goroutine, including while a generation runs, and
// while a Close is waiting for one (Close interrupts it itself). A call
// when nothing is running is a no-op: the flag is cleared when generation
// starts.
func (c *Context) Interrupt() error {
	if c.model.inst.closed.Load() {
		return fmt.Errorf("llama: interrupt: %w", ErrInstanceClosed)
	}
	return c.model.inst.interrupt(c.st)
}
