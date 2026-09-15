package llama

import (
	"errors"
	"fmt"
	"runtime"
	"sort"

	bridge "github.com/goccy/go-llama/internal"
)

// Instance snapshots: prepare once, fork many times.
//
// An engine instance serialises every call, so the unit of parallelism is
// the instance — but preparing one the ordinary way (create a context,
// restore a prompt-prefix state) writes tens of MB of private pages per
// instance: the KV cache is allocated and zeroed anew, and the prefix state
// is copied into it. A Snapshot removes that: NewSnapshot prepares an
// engine ONCE — model loaded, contexts created, prefixes decoded — and
// captures it as a copy-on-write image. Fork then brings up an engine that
// starts exactly there: the prepared KV caches are shared pages, and a
// fork pays only for what it writes (its suffix's KV entries and the
// compute scratch), typically an order of magnitude less.
//
// Worker threads do not survive a snapshot — they are host constructs, not
// memory. NewSnapshot therefore frees every kept context's threadpool
// before capturing (a fork that skips reattaching merely runs
// single-threaded), and Fork attaches a fresh pool with the requested
// thread count, which the fork's Close joins again.

// SnapshotBuilder is the instance handed to a NewSnapshot callback: a normal
// *Llama plus Register, which records the contexts forks will use.
type SnapshotBuilder struct {
	*Llama
	kept map[string]keptContext
}

type keptContext struct {
	modelH uint64
	ctxH   uint64
	st     *ctxState // the builder's own record of the context
}

// Register records a prepared context under name, making it available from
// every fork of the snapshot via Instance.Context(name). A context is
// registered once: every fork frees each of its kept contexts on Close, so
// the same context under two names would be freed twice.
func (b *SnapshotBuilder) Register(name string, c *Context) error {
	if c == nil || c.st.closed.Load() {
		return errors.New("llama: register: context is nil or closed")
	}
	if c.model.inst != b.Llama {
		return errors.New("llama: register: context does not belong to this builder")
	}
	if _, dup := b.kept[name]; dup {
		return fmt.Errorf("llama: register: %q already registered", name)
	}
	for other, k := range b.kept {
		if k.st == c.st {
			return fmt.Errorf("llama: register: context already registered as %q", other)
		}
	}
	b.kept[name] = keptContext{modelH: c.model.h, ctxH: c.h, st: c.st}
	return nil
}

// Snapshot is the copy-on-write image of a prepared instance. It stays valid
// for the life of the process; forks reference it and share its pages.
type Snapshot struct {
	snap *bridge.InstanceSnapshot
	cfg  config
	kept map[string]keptContext
}

// NewSnapshot boots an engine on a snapshot image and hands it to build to
// prepare: load the model, create contexts, decode their prompt prefixes
// (Params.CachePrompt), and Register the ones forks will use. When build
// returns, every kept context's threadpool is freed — its workers joined,
// since threads are host constructs and cannot be captured — and the image
// is sealed.
//
// The builder instance itself is consumed by the build; do not retain the
// *Llama, models or contexts it produced beyond the callback.
func NewSnapshot(build func(*SnapshotBuilder) error, opts ...Option) (*Snapshot, error) {
	var cfg config
	for _, o := range opts {
		o(&cfg)
	}
	kept := map[string]keptContext{}
	var wedged error
	snap, err := bridge.BuildInstanceSnapshot(cfg.bridgeOptions(true), func(m *bridge.Module) error {
		l := &Llama{cfg: cfg}
		l.cfg.noSharedModel = true // load into THIS image, not the process-wide model snapshot
		l.eng.Store(m)
		b := &SnapshotBuilder{Llama: l, kept: kept}
		// A build that panics is a build that failed: recovered here, so
		// that its contexts are retired like a failed build's, before the
		// image builder sees an error and the memory goes away.
		buildErr := func() (err error) {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("llama: snapshot build panicked: %v", r)
				}
			}()
			return build(b)
		}()
		if buildErr == nil && len(kept) == 0 {
			buildErr = errors.New("llama: snapshot build kept no context")
		}
		// Threads are not memory: before the image is sealed — or, on a
		// failed build, before its memory goes away — every pool the build
		// created is stopped and its workers, live goroutines of this
		// process, joined rather than left parked for good. A kept context
		// keeps everything but its pool (a fork that skips
		// AttachThreadpool then runs single-threaded, safely); a context
		// the build created and did not keep is freed outright.
		if err := l.retireContexts(kept, buildErr != nil); err != nil {
			buildErr = errors.Join(buildErr, err)
		}
		if l.wedged.Load() {
			// A join trapped: workers may still run in this memory, and an
			// error returned here would have the image builder unmap it
			// under them. Seal the image instead: its mapping then stays
			// as long as those workers hold the engine, which is for good.
			// Nothing forks it.
			m.PreserveMemoryOnAbandon()
			wedged = errors.Join(buildErr, ErrMemoryLeaked)
			return nil
		}
		return buildErr
	})
	if err != nil {
		return nil, err
	}
	if wedged != nil {
		return nil, wedged
	}
	return &Snapshot{snap: snap, cfg: cfg, kept: kept}, nil
}

// retireContexts prepares the builder's contexts for the capture: the
// kept ones lose their pool (and any Slots on them, and the calls the
// build left in flight are waited for — the image must not carry a call
// in progress), the others are freed. With all set, the kept ones are
// freed as well — the build failed, and nothing will fork them. A kept
// context that the build closed itself is an error: the image would seal
// a freed handle for every fork to use.
func (l *Llama) retireContexts(kept map[string]keptContext, all bool) error {
	// By the Go-side record, never by guest handle: a handle is a heap
	// address the guest reuses once the context is freed, so a context
	// created after a kept one was closed could pass for it.
	keptSt := map[*ctxState]string{}
	for name, k := range kept {
		keptSt[k.st] = name
	}
	var errs []error
	for _, st := range l.takeContexts() {
		name, isKept := keptSt[st]
		if isKept && !all {
			if err := l.retireKept(st); err != nil {
				errs = append(errs, fmt.Errorf("llama: retire %s: %w", name, err))
			}
			continue
		}
		if err := l.freeContext(st); err != nil {
			errs = append(errs, err)
		}
	}
	if !all {
		for st, name := range keptSt {
			if st.freed.Load() {
				errs = append(errs, fmt.Errorf("llama: kept context %s was closed by the build", name))
			}
		}
	}
	return errors.Join(errs...)
}

// retireKept frees a kept context's threadpool, with the context held
// exclusively as a free would hold it, its Slots closed first. The
// builder's Context is closed to calls from then on: the image is about
// to be sealed, and a call the build left running (its own bug) must not
// enter it meanwhile. A call in flight is waited for, not interrupted —
// it is preparing what the image keeps.
func (l *Llama) retireKept(st *ctxState) error {
	st.closed.Store(true)
	st.calls.Lock()
	defer st.calls.Unlock()
	st.slotsMu.Lock()
	s := st.slots
	st.slotsMu.Unlock()
	if s != nil {
		_ = s.Close()
	}
	return l.noteTrap(freeThreadpool(l.e(), st.h))
}

// Fork brings up an instance from the snapshot: the prepared contexts are
// live immediately, with fresh threadpools of nThreads workers each
// (nThreads <= 1 keeps them single-threaded). Close the fork when done
// with it — its private pages are returned then.
func (s *Snapshot) Fork(nThreads uint32) (*Instance, error) {
	eng, err := bridge.NewEngineFromInstanceSnapshot(s.snap, s.cfg.bridgeOptions(false))
	if err != nil {
		return nil, fmt.Errorf("llama: fork: %w", err)
	}
	l := &Llama{cfg: s.cfg}
	l.eng.Store(eng)
	f := &Instance{l: l, ctxs: map[string]*Context{}}
	models := map[uint64]*Model{}
	names := make([]string, 0, len(s.kept))
	for name := range s.kept {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		k := s.kept[name]
		m, ok := models[k.modelH]
		if !ok {
			m = &Model{inst: l, h: k.modelH}
			models[k.modelH] = m
			l.models++
		}
		c := &Context{model: m, h: k.ctxH, st: &ctxState{h: k.ctxH, modelH: k.modelH}}
		f.ctxs[name] = c
		l.ctxMu.Lock()
		l.track(c.st)
		l.ctxMu.Unlock()
	}
	// The instance frees its contexts on Close, which joins the pool
	// workers a fork attached, so a fork the GC finds unclosed is torn down
	// in the same order as a closed one. The finalizer cannot run while a
	// Context or Model of the fork is reachable (they point at l), and the
	// Contexts themselves are reachable from nowhere in l.
	runtime.SetFinalizer(l, func(l *Llama) { go l.Close() })
	for _, name := range names {
		c := f.ctxs[name]
		if c.st.interruptAddr, err = eng.LlamaCtxInterruptAddr(c.h); err != nil || c.st.interruptAddr == 0 {
			return nil, errors.Join(fmt.Errorf("llama: fork: interrupt address for %s: %v", name, err), f.Close())
		}
		if nThreads > 1 {
			// A trap here leaves the pool in an unknown state (workers may
			// have started before it); l.noteTrap makes Close keep the
			// memory mapped rather than unmap it under them, and Close's
			// own error (ErrMemoryLeaked then) is part of the answer.
			if err := l.noteTrap(attachThreadpool(eng, c.h, nThreads)); err != nil {
				return nil, errors.Join(fmt.Errorf("llama: fork: attach threadpool for %s: %w", name, err), f.Close())
			}
		}
	}
	return f, nil
}

// Instance is one engine instance forked from a Snapshot.
type Instance struct {
	l    *Llama
	ctxs map[string]*Context
}

// Context returns the kept context by name, or nil.
func (f *Instance) Context(name string) *Context { return f.ctxs[name] }

// Close releases the fork: its kept contexts are freed, which joins the
// threadpool workers Fork attached, and then its copy-on-write mapping is
// unmapped, returning every private page. The kept contexts need no
// Close of their own; one that was closed already is simply skipped.
//
// If freeing a context trapped, whether its workers were joined is
// unknown, and the mapping is kept rather than unmapped under threads
// that may still run in it: Close then returns ErrMemoryLeaked along with
// the context's error. Any error of a kept context's Close is returned.
func (f *Instance) Close() error { return f.l.Close() }

// attachThreadpool rebuilds and attaches a context's ggml threadpool via the
// bridge; n <= 1 detaches instead. The bridge reports the thread counts the
// context will compute with afterwards; they must equal the pool size, since
// a pool the context does not use leaves its workers idling.
func attachThreadpool(m *bridge.Module, ctxH uint64, n uint32) error {
	js, err := m.LlamaCtxAttachThreadpool(ctxH, n)
	if err != nil {
		return err
	}
	return checkThreadpool("attach threadpool", js, n)
}

// freeThreadpool stops and joins the threadpool of a context this engine
// created, leaving the context single-threaded. For the snapshot builder
// only: a fork's contexts carry no pool of their own until attachThreadpool
// gives them one, and Close frees that one with the context.
func freeThreadpool(m *bridge.Module, ctxH uint64) error {
	js, err := m.LlamaCtxFreeThreadpool(ctxH)
	if err != nil {
		return err
	}
	return checkThreadpool("free threadpool", js, 1)
}

// checkThreadpool decodes a threadpool RPC's reply and checks the thread
// counts the context computes with against the pool it now has.
func checkThreadpool(what, js string, n uint32) error {
	var out struct {
		envelope
		NThreads      uint32 `json:"n_threads"`
		NThreadsBatch uint32 `json:"n_threads_batch"`
	}
	if err := decode(what, js, &out); err != nil {
		return err
	}
	want := max(n, 1)
	if out.NThreads != want || out.NThreadsBatch != want {
		return fmt.Errorf("%s: context computes with %d/%d threads, want %d", what, out.NThreads, out.NThreadsBatch, want)
	}
	return nil
}
