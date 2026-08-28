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
// memory. NewSnapshot therefore detaches every kept context's threadpool
// before capturing (a fork that skips reattaching merely runs
// single-threaded), and Fork attaches a fresh pool with the requested
// thread count.

// SnapshotBuilder is the instance handed to a NewSnapshot callback: a normal
// *Llama plus Register, which records the contexts forks will use.
type SnapshotBuilder struct {
	*Llama
	kept map[string]keptContext
}

type keptContext struct {
	modelH uint64
	ctxH   uint64
}

// Register records a prepared context under name, making it available from
// every fork of the snapshot via Instance.Context(name).
func (b *SnapshotBuilder) Register(name string, c *Context) error {
	if c == nil || c.closed.Load() {
		return errors.New("llama: register: context is nil or closed")
	}
	if c.model.inst != b.Llama {
		return errors.New("llama: register: context does not belong to this builder")
	}
	if _, dup := b.kept[name]; dup {
		return fmt.Errorf("llama: register: %q already registered", name)
	}
	b.kept[name] = keptContext{modelH: c.model.h, ctxH: c.h}
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
// returns, every kept context's threadpool is detached (threads are host
// constructs and cannot be captured) and the image is sealed.
//
// The builder instance itself is consumed by the build; do not retain the
// *Llama, models or contexts it produced beyond the callback.
func NewSnapshot(build func(*SnapshotBuilder) error, opts ...Option) (*Snapshot, error) {
	var cfg config
	for _, o := range opts {
		o(&cfg)
	}
	kept := map[string]keptContext{}
	snap, err := bridge.BuildInstanceSnapshot(cfg.bridgeOptions(true), func(m *bridge.Module) error {
		l := &Llama{cfg: cfg}
		l.cfg.noSharedModel = true // load into THIS image, not the process-wide model snapshot
		l.eng.Store(m)
		b := &SnapshotBuilder{Llama: l, kept: kept}
		if err := build(b); err != nil {
			return err
		}
		if len(kept) == 0 {
			return errors.New("llama: snapshot build kept no context")
		}
		// Threads are not memory: detach every kept context's pool so the
		// captured state never references the builder's dead workers. A fork
		// that skips AttachThreadpool then runs single-threaded, safely.
		names := make([]string, 0, len(kept))
		for name := range kept {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			if err := attachThreadpool(m, kept[name].ctxH, 1); err != nil {
				return fmt.Errorf("llama: detach %s: %w", name, err)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return &Snapshot{snap: snap, cfg: cfg, kept: kept}, nil
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
	runtime.SetFinalizer(l, func(l *Llama) { l.Close() })
	f := &Instance{l: l, ctxs: map[string]*Context{}}
	models := map[uint64]*Model{}
	for name, k := range s.kept {
		m, ok := models[k.modelH]
		if !ok {
			m = &Model{inst: l, h: k.modelH}
			models[k.modelH] = m
			l.models++
		}
		c := &Context{model: m, h: k.ctxH}
		if c.interruptAddr, err = eng.LlamaCtxInterruptAddr(k.ctxH); err != nil || c.interruptAddr == 0 {
			_ = l.Close()
			return nil, fmt.Errorf("llama: fork: interrupt address for %s: %v", name, err)
		}
		if nThreads > 1 {
			if err := attachThreadpool(eng, k.ctxH, nThreads); err != nil {
				_ = l.Close()
				return nil, fmt.Errorf("llama: fork: attach threadpool for %s: %w", name, err)
			}
		}
		f.ctxs[name] = c
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

// Close releases the fork: its copy-on-write mapping is unmapped, returning
// every private page. The kept contexts need no individual Close — their
// backing memory is the mapping itself.
func (f *Instance) Close() error { return f.l.Close() }

// attachThreadpool rebuilds and attaches a context's ggml threadpool via the
// bridge; n <= 1 detaches instead.
func attachThreadpool(m *bridge.Module, ctxH uint64, n uint32) error {
	js, err := m.LlamaCtxAttachThreadpool(ctxH, n)
	if err != nil {
		return err
	}
	var out struct {
		envelope
	}
	return decode("attach threadpool", js, &out)
}
