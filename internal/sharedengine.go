package internal

// Copy-on-write engine memory.
//
// Every engine instance starts from the same ~9 MB of data segments and, when
// several instances load the same model, the same gigabytes of weights. Paying
// for a private copy of all that per instance is pure waste: build the common
// state ONCE per process and map it copy-on-write into each instance, so
// read-only pages stay physically shared and only the pages an instance
// actually writes become private. wasm2go owns the machinery
// (base.NewSharedImage / base.NewSharedSnapshot); this file says what to image.
//
// Two tiers, mirroring the shared-image design used by other wasm2go embedders:
//
//   - The DATA-SEGMENT image backs every engine (NewEngine): a copy-on-write
//     map of the module's installed data segments. The instance re-runs its own
//     initialization over it — memory.init compares before it writes, so the
//     segment pages stay shared — with its own WASI, so nothing about the
//     builder's configuration leaks into instances.
//
//   - A MODEL snapshot backs engines that load a model an earlier engine
//     already loaded (NewEngineFromModelSnapshot): a copy-on-write map of a
//     fully started engine with the model in memory, keyed by everything the
//     baked state depends on. Instances re-run nothing and inherit the model
//     handle; the weights stay physically shared until written, which for
//     inference is never.
//
// Where mmap is unavailable, or when GO_LLAMA_NO_SHARED_IMAGE is set, every
// tier reports an error and callers fall back to private allocations — correct,
// only larger.

import (
	"fmt"
	"os"
	"sync"

	wasm2go "github.com/goccy/llamawasm2go"
	"github.com/goccy/llamawasm2go/base"
)

const (
	wasmPageSize = 65536

	// defaultMemoryCeiling sizes the copy-on-write mapping an instance grows
	// into when Options.MaxMemoryBytes is 0. It is ADDRESS SPACE, not memory:
	// untouched pages never page in, so generous headroom is nearly free, and
	// a model bigger than this needs an explicit WithMaxMemory anyway.
	defaultMemoryCeiling = 64 << 30
)

func sharedImagesDisabled() bool { return os.Getenv("GO_LLAMA_NO_SHARED_IMAGE") != "" }

// memoryCeiling resolves the size of an instance's linear-memory mapping. The
// mapping length is also the growth cap (the module fails memory.grow past it
// rather than reallocating off the shared mapping), so an explicit MaxMemory
// wins, and the reserve is folded in for callers that ask for more than the
// default.
func memoryCeiling(opts Options) int {
	c := uint64(defaultMemoryCeiling)
	if opts.MaxMemoryBytes > 0 {
		c = opts.MaxMemoryBytes
	}
	if r := uint64(opts.MemoryReserveBytes); r > c {
		c = r
	}
	return int(c / wasmPageSize * wasmPageSize)
}

// --- copy-on-write data-segment image (every engine) ------------------------

// sharedEngineImage returns the process-wide data-segment image, built on
// first use. The builder runs only the start section (Initialize): instances
// run WasmInit themselves, with their own WASI.
func sharedEngineImage() *base.SharedImage {
	return base.NewSharedImage(func() (g *base.Module, err error) {
		if sharedImagesDisabled() {
			return nil, fmt.Errorf("disabled by GO_LLAMA_NO_SHARED_IMAGE")
		}
		defer func() {
			if r := recover(); r != nil {
				g, err = nil, fmt.Errorf("the engine's start section panicked: %v", r)
			}
		}()
		m := &Module{}
		m.g = wasm2go.NewWithWASI(base.DefaultWASI(), envStubs{m: m}, wasmifyStubs{m: m})
		wasm2go.Initialize(m.g)
		return m.g, nil
	})
}

// --- copy-on-write model snapshot (engines sharing a loaded model) ----------

// ModelSnapshot is a copy-on-write image of a started engine with one model
// loaded, plus the model handle baked into that memory.
type ModelSnapshot struct {
	img    *base.SharedImage
	handle uint64
	err    error
}

// Err reports why the snapshot is unavailable, or nil.
func (s *ModelSnapshot) Err() error { return s.err }

// Handle is the loaded model's bridge handle inside the snapshotted memory —
// valid in every instance built from the snapshot.
func (s *ModelSnapshot) Handle() uint64 { return s.handle }

var (
	modelSnapMu sync.Mutex
	modelSnaps  = map[string]*ModelSnapshot{}
)

// SharedModelSnapshot returns the process-wide snapshot for key, building it
// on first use. key must cover everything the baked memory depends on: the
// guest environment, the filesystem configuration, and the model path (the
// caller owns the key format). A failed build is remembered per key — callers
// fall back to loading the model privately.
func SharedModelSnapshot(key string, opts Options, load func(*Module) (uint64, error)) *ModelSnapshot {
	modelSnapMu.Lock()
	defer modelSnapMu.Unlock()
	if s, ok := modelSnaps[key]; ok {
		return s
	}
	s := buildModelSnapshot(opts, load)
	modelSnaps[key] = s
	return s
}

func buildModelSnapshot(opts Options, load func(*Module) (uint64, error)) *ModelSnapshot {
	if sharedImagesDisabled() {
		return &ModelSnapshot{err: fmt.Errorf("disabled by GO_LLAMA_NO_SHARED_IMAGE")}
	}
	var handle uint64
	img := base.NewSharedSnapshot(func() (g *base.Module, err error) {
		defer func() {
			if r := recover(); r != nil {
				g, err = nil, fmt.Errorf("starting the engine to snapshot panicked: %v", r)
			}
		}()
		// The builder engine is private on purpose: buildSharedImage copies its
		// memory and discards the instance, and a private allocation is the Go
		// heap's to reclaim — an image-backed builder would leak its mapping.
		m, err := NewPrivateEngine(opts)
		if err != nil {
			return nil, err
		}
		h, err := load(m)
		if err != nil {
			return nil, err
		}
		if h == 0 {
			return nil, fmt.Errorf("model load returned handle 0")
		}
		handle = h
		return m.g, nil
	})
	if err := img.Err(); err != nil {
		return &ModelSnapshot{err: err}
	}
	return &ModelSnapshot{img: img, handle: handle}
}

// NewEngineFromModelSnapshot brings up an engine as a copy-on-write map of
// snap: fully started, model loaded, nothing re-run. opts supplies THIS
// instance's WASI (its stdio and filesystem are per-instance; the baked state
// depends only on what the snapshot key covers).
func NewEngineFromModelSnapshot(snap *ModelSnapshot, opts Options) (*Module, error) {
	if snap.err != nil {
		return nil, snap.err
	}
	mem, err := snap.img.Memory(memoryCeiling(opts))
	if err != nil {
		return nil, err
	}
	m := &Module{}
	m.g = wasm2go.NewFromSnapshot(engineWASI(opts), envStubs{m: m}, wasmifyStubs{m: m},
		mem, snap.img.Size(), snap.img.Globals())
	engineMmaps.Store(m, mem)
	return m, nil
}
