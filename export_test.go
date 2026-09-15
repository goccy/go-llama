package llama

// Test-only windows into the instance internals.

import (
	"errors"

	bridge "github.com/goccy/go-llama/internal"
)

// EngineImageBacked reports whether the instance's engine memory is a
// copy-on-write map of a shared image rather than a private allocation.
func EngineImageBacked(l *Llama) bool { return l.e().ImageBacked() }

// TrapNextGraph arms the engine's test hook on c: the next graph computed
// on it traps on the main thread mid-graph, leaving the threadpool's
// workers waiting at a barrier — what a failed assertion inside a decode
// leaves behind.
func TrapNextGraph(c *Context) error {
	js, err := c.model.inst.e().LlamaCtxDbgTrapNextGraph(c.h)
	if err != nil {
		return err
	}
	var out struct{ envelope }
	return decode("trap next graph", js, &out)
}

// IsTrap reports whether err is (or wraps) an engine trap.
func IsTrap(err error) bool {
	var trap *bridge.TrapError
	return errors.As(err, &trap)
}

// SnapshotImageSize is the size in bytes of the snapshot's memory image —
// the memory a fork starts with, and the least it can be capped at.
func SnapshotImageSize(s *Snapshot) uint64 { return s.snap.ImageSize() }

// IsGuestError reports whether err is (or wraps) a failure the guest
// reported normally, as opposed to a trap or a closed engine.
func IsGuestError(err error) bool {
	var guest *bridge.GuestError
	return errors.As(err, &guest)
}
