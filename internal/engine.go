package internal

// Hand-written per-instance surface over the generated bridge.
//
// The generated functions in llama.go (LlamaModelLoad, LlamaCtxGenerate, ...)
// route through module()/globalModule — a single process-wide instance. This
// file instead runs many engines concurrently and in isolation, each on its
// own wasm module (its own linear memory, so llama.cpp's C++ globals live in
// per-module memory). Every method below drives THIS engine's module via
// m.invoke, so there is no global module.
//
// The method IDs and their generated entry points are hand-maintained (see the
// mid* block and the invokers table). They do NOT auto-follow a proto change
// the way the generated functions do — the service numbers its RPCs
// alphabetically, so ONE added method renumbers every later one. On every
// bridge regeneration, re-check the invokers table against llama.go's
// Inv_0_* entry points. Everything funnels through the one invokeMethod
// helper, so callers only ever name a mid; they never touch invoke, a raw
// service/method number, or an entry point.

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	wasm2go "github.com/goccy/llamawasm2go"
	"github.com/goccy/llamawasm2go/base"
)

// Method IDs for service 0 (the data plane), in the alphabetical order the
// bridge numbers them. Their ordinal value IS the generated method number, so
// the iota order must match llama.go's Inv_0_0, Inv_0_1, ... exactly.
const (
	midChatApplyTemplate int32 = iota
	midCtxAttachThreadpool
	midCtxDbgTrapNextGraph
	midCtxEmbed
	midCtxEmbedTokens
	midCtxEval
	midCtxFree
	midCtxFreeThreadpool
	midCtxGenerate
	midCtxGenerateSpeculative
	midCtxInterruptAddr
	midCtxLoraSet
	midCtxNew
	midCtxReset
	midCtxScore
	midCtxScoreChoices
	midCtxSlotsCancel
	midCtxSlotsPost
	midCtxSlotsStatus
	midCtxSlotsSystemPrompt
	midCtxSlotsUpdate
	midCtxStateLoad
	midCtxStateSave
	midDetokenize
	midLoraFree
	midLoraLoad
	midModelFree
	midModelInfo
	midModelLoad
	midModelLoadProgressAddr
	midModelTensors
	midTokenToPiece
	midTokenize
	midWasmBuildInfo
	midWasmFree
	midWasmInit
	midWasmLastError
	midCount
)

// invokers maps a service-0 method ID to the generated per-export entry point
// (wasm2go.Inv_0_<n>). This is the single table to reconcile with llama.go
// after a regeneration; nothing else in this file names an Inv_0_* directly.
var invokers = [midCount]func(*base.Module, wptr, wptr) (int64, error){
	midChatApplyTemplate:      wasm2go.Inv_0_0,
	midCtxAttachThreadpool:    wasm2go.Inv_0_1,
	midCtxDbgTrapNextGraph:    wasm2go.Inv_0_2,
	midCtxEmbed:               wasm2go.Inv_0_3,
	midCtxEmbedTokens:         wasm2go.Inv_0_4,
	midCtxEval:                wasm2go.Inv_0_5,
	midCtxFree:                wasm2go.Inv_0_6,
	midCtxFreeThreadpool:      wasm2go.Inv_0_7,
	midCtxGenerate:            wasm2go.Inv_0_8,
	midCtxGenerateSpeculative: wasm2go.Inv_0_9,
	midCtxInterruptAddr:       wasm2go.Inv_0_10,
	midCtxLoraSet:             wasm2go.Inv_0_11,
	midCtxNew:                 wasm2go.Inv_0_12,
	midCtxReset:               wasm2go.Inv_0_13,
	midCtxScore:               wasm2go.Inv_0_14,
	midCtxScoreChoices:        wasm2go.Inv_0_15,
	midCtxSlotsCancel:         wasm2go.Inv_0_16,
	midCtxSlotsPost:           wasm2go.Inv_0_17,
	midCtxSlotsStatus:         wasm2go.Inv_0_18,
	midCtxSlotsSystemPrompt:   wasm2go.Inv_0_19,
	midCtxSlotsUpdate:         wasm2go.Inv_0_20,
	midCtxStateLoad:           wasm2go.Inv_0_21,
	midCtxStateSave:           wasm2go.Inv_0_22,
	midDetokenize:             wasm2go.Inv_0_23,
	midLoraFree:               wasm2go.Inv_0_24,
	midLoraLoad:               wasm2go.Inv_0_25,
	midModelFree:              wasm2go.Inv_0_26,
	midModelInfo:              wasm2go.Inv_0_27,
	midModelLoad:              wasm2go.Inv_0_28,
	midModelLoadProgressAddr:  wasm2go.Inv_0_29,
	midModelTensors:           wasm2go.Inv_0_30,
	midTokenToPiece:           wasm2go.Inv_0_31,
	midTokenize:               wasm2go.Inv_0_32,
	midWasmBuildInfo:          wasm2go.Inv_0_33,
	midWasmFree:               wasm2go.Inv_0_34,
	midWasmInit:               wasm2go.Inv_0_35,
	midWasmLastError:          wasm2go.Inv_0_36,
}

// NewEngine brings up an independent engine instance: its own wasm module
// (its own view of linear memory / C heap), configured by opts. The memory is
// a copy-on-write map of the process-wide data-segment image when one is
// available (see sharedengine.go), a private allocation otherwise; either way
// the instance runs its own initialization with its own WASI.
func NewEngine(opts Options) (m *Module, err error) {
	img := sharedEngineImage()
	mem, imgErr := mapSharedMemory(img, opts)
	if imgErr != nil {
		return newPrivateEngine(opts)
	}
	m = newModule()
	m.adopt(wasm2go.NewWithMemory(engineWASI(opts), envStubs{m: m}, wasmifyStubs{m: m},
		mem, img.Size()))
	engineMmaps.Store(m, mem)
	if err := initEngine(m); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

// newPrivateEngine is NewEngine without the shared image: the instance's
// memory is its own allocation. Never a caller-facing choice — copy-on-write
// always applies when the platform can map it (GO_LLAMA_NO_SHARED_IMAGE is
// the debugging escape hatch): this is the automatic fallback when mapping
// is unavailable, and what the snapshot builder uses on purpose, because a
// builder instance is discarded after its memory is copied into the image
// and an mmap-backed one would leak its mapping.
func newPrivateEngine(opts Options) (m *Module, err error) {
	m = newModule()
	env := envStubs{m: m}
	wm := wasmifyStubs{m: m}
	wasi := engineWASI(opts)
	if opts.MemoryReserveBytes > 0 {
		m.adopt(wasm2go.NewWithWASIReserve(wasi, env, wm, opts.MemoryReserveBytes))
	} else {
		m.adopt(wasm2go.NewWithWASI(wasi, env, wm))
	}
	if opts.MaxMemoryBytes > 0 {
		wasm2go.SetMaxMemory(m.g, opts.MaxMemoryBytes)
	}
	if err := initEngine(m); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

func engineWASI(opts Options) base.Wasi_snapshot_preview1Imports {
	if opts.WASI != nil {
		return opts.WASI
	}
	return base.DefaultWASI()
}

// initEngine runs the start section (installs the data segments — over a
// shared image, memory.init finds them in place and leaves the pages shared)
// and _initialize (the C++ static constructors) under a recover, so a trap in
// a static initializer surfaces as an error rather than a panic.
func initEngine(m *Module) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("engine init panicked: %v", r)
		}
	}()
	wasm2go.Initialize(m.g)
	_ = wasm2go.WasmInit(m.g)
	return nil
}

// engineMmaps tracks the copy-on-write mapping backing an engine's memory.
// This state cannot live on Module itself — the struct is defined in the
// generated (and attestation-verified) llama.go, which hand-written code must
// not modify — so it rides in a side table keyed by the module.
var engineMmaps sync.Map // *Module -> []byte

// engineGates holds each engine's gate (engineGate): the lock that orders
// an engine's calls against its Close, what it knows of its threads, and
// whether it trapped. A call holds the lock shared for its duration and
// checks Closed under it; Close takes it exclusively. Without it a call
// that passed its closed-check just before Close detached the memory
// would run the transpiled entry over a nil memory — a panic, not an
// error, and on a goroutine of the caller's (a Slots scheduler, say) with
// nothing to recover it. The lock nests outside m.mu on both paths.
//
// The entry is stored when the engine is created (newModule) and deleted
// when it is released; a missing entry means closed, and nothing is ever
// inserted afterwards, so a dead engine is not pinned by its own gate. A
// side table for the same reason as engineMmaps.
var engineGates sync.Map // *Module -> *engineGate

type engineGate struct {
	mu sync.RWMutex
	// threads counts the engine's wasi threads from the start of each to
	// its goroutine's end. A guest's pthread_join returns as soon as the
	// thread has published its exit, while the thread still has an
	// atomic notify to make in the memory; the unmap waits for the
	// goroutines themselves.
	threads atomic.Int64
	// trapped is the first trap the engine took, if any. C++ frames a
	// trap unwound ran no destructors: whatever they held — locks, heap
	// blocks, half-built state — stays as it was, so the engine is only
	// good for tearing down afterwards.
	trapped atomic.Pointer[TrapError]
	// preserveMemory suppresses an in-place image builder's finalizer when
	// workers may still reference that mapping.
	preserveMemory atomic.Bool
}

// newModule is the constructor every engine goes through: it registers
// the engine's gate. adopt then hands it the transpiled module.
func newModule() *Module {
	m := &Module{}
	engineGates.Store(m, &engineGate{})
	return m
}

// adopt installs the transpiled module and hooks its thread entry so the
// gate counts the engine's threads.
func (m *Module) adopt(g *base.Module) {
	m.g = g
	gate, _ := engineGates.Load(m)
	start := g.ThreadStart64
	if start == nil {
		return
	}
	g.ThreadStart64 = func(child *base.Module, tid int32, arg int64) {
		gate.(*engineGate).threads.Add(1)
		defer gate.(*engineGate).threads.Add(-1)
		start(child, tid, arg)
	}
}

// threadExitWait is how long Close waits for the engine's threads to be
// gone after every context of it was freed (which joined them): they need
// microseconds, and a wait this long means one never will.
const threadExitWait = 10 * time.Second

// ErrThreadsAlive is returned by Close when the engine's threads did not
// exit in time; the memory was kept mapped rather than unmapped under
// them.
var ErrThreadsAlive = errors.New("threads of the engine are still running; memory kept mapped")

// ErrEngineClosed is returned by every call on a closed engine. Its text
// is the user-facing one: the public package exposes it as its own
// instance-closed sentinel.
var ErrEngineClosed = errors.New("instance is closed")

// GuestError is a failure the guest reported normally: the bridge's own
// error object in the reply, with the guest's state as the call left it.
// It is the third kind of call error next to ErrEngineClosed and
// TrapError, and the only one after which the guest is known to be
// consistent.
type GuestError struct {
	Err error
}

func (e *GuestError) Error() string { return e.Err.Error() }
func (e *GuestError) Unwrap() error { return e.Err }

// TrapError is an engine call that trapped: the guest hit an unreachable
// (a failed assertion, an out-of-bounds access) and unwound to the call
// boundary. The call's own frames are gone, but whatever guest state it
// was changing is left as it was at the trap — in particular a pool of
// worker threads the call was about to create, join or hand over may be
// in any state. Callers that release memory decide on it with errors.As.
type TrapError struct {
	Err error
	// Refused is set when the call never entered the guest: the engine
	// had trapped earlier (Err names that trap) and takes only teardown
	// calls since. The guest's state is whatever that earlier trap left,
	// not something this call changed.
	Refused bool
}

func (e *TrapError) Error() string { return e.Err.Error() }
func (e *TrapError) Unwrap() error { return e.Err }

// Closed reports whether Close (or Abandon) has detached the engine from
// its memory.
func (m *Module) Closed() bool { return m.g == nil || m.g.Memory == nil }

// Close releases the engine's memory. An engine backed by a copy-on-write
// mapping owns that mapping — it is not Go heap, so it must be unmapped here
// rather than left to the GC; a private allocation is simply detached for the
// GC to reclaim. Either way the module is left memoryless, and the
// data-plane calls (invokeMethod) refuse to run afterwards, so a late call
// gets ErrEngineClosed instead of touching freed pages. Idempotent.
//
// The caller guarantees no guest thread of this engine is still running:
// a wasi thread shares the memory but not this Module's view of it, so it
// would not see the detach and would trap against the unmapped pages, on
// its own goroutine, with nothing to recover it. When that cannot be
// guaranteed, Abandon instead.
//
// Close returns ErrThreadsAlive when a thread of the engine is still
// running after threadExitWait: the memory is then kept, as by Abandon.
func (m *Module) Close() error { return m.release(true) }

// Abandon is Close without the unmap: the engine is detached and refuses
// calls like a closed one, but its memory stays mapped for the life of the
// process. For an engine whose guest threads may still be alive — a call
// that should have joined them trapped — this leaks the mapping instead of
// pulling it out from under them. It is also how a snapshot builder's
// engine is retired: its memory IS the image, owned by the image from
// then on.
func (m *Module) Abandon() { _ = m.release(false) }

// PreserveMemoryOnAbandon keeps an in-place image builder's mapping alive
// after Abandon. NewSharedSnapshotInPlace installs a finalizer that normally
// unmaps the builder memory after capture; a builder with possibly live
// threads must suppress it.
func (m *Module) PreserveMemoryOnAbandon() {
	if g, ok := engineGates.Load(m); ok {
		g.(*engineGate).preserveMemory.Store(true)
	}
}

func (m *Module) release(unmap bool) error {
	g, ok := engineGates.Load(m)
	if !ok {
		return nil // released already
	}
	gate := g.(*engineGate)
	gate.mu.Lock()
	defer gate.mu.Unlock()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.Closed() {
		return nil
	}
	var err error
	if unmap && !threadsGone(&gate.threads, threadExitWait) {
		unmap = false
		err = ErrThreadsAlive
	}
	if !unmap && gate.preserveMemory.Load() {
		runtime.SetFinalizer(m.g, nil)
	}
	// Detach the module from its memory before releasing it, so a stray late
	// call fails a closed-check instead of touching unmapped pages. Memory
	// and M are this Module's own view; MemSize is shared by pointer with
	// every wasi thread's copy of the module (ThreadLaunch), so it is the
	// one thing a live thread would see: zero it only when the memory does
	// go away, so that a thread still running in an abandoned memory keeps
	// its bound and runs on rather than trapping against zero.
	// Under MemMu as well: base.AccessMemory (Interrupt's write of the
	// interrupt word) holds it while it touches the memory, so it either
	// completes before the detach or finds no memory afterwards, never a
	// page that was unmapped under it.
	m.g.MemMu.Lock()
	defer m.g.MemMu.Unlock()
	m.g.Memory = nil
	m.g.M = nil
	mem, mapped := engineMmaps.LoadAndDelete(m)
	if unmap {
		m.g.MemSize.Store(0)
		if mapped {
			base.UnmapMemory(mem.([]byte))
		}
	}
	engineGates.Delete(m)
	return err
}

// threadsGone reports whether the thread count reaches zero within d.
// Polled rather than waited on, so nothing outlives the deadline.
func threadsGone(n *atomic.Int64, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for n.Load() > 0 {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(100 * time.Microsecond)
	}
	return true
}

// ImageBacked reports whether this engine's memory is a copy-on-write map of
// a shared image (data-segment or snapshot) rather than a private allocation.
func (m *Module) ImageBacked() bool {
	_, ok := engineMmaps.Load(m)
	return ok
}

// Base returns the engine's transpiled module, for base.AccessMemory (the only
// safe way to touch linear memory from another goroutine — an interrupt flag
// or a polled progress word).
func (m *Module) Base() *base.Module { return m.g }

// invokeMethod runs one service-0 RPC on THIS engine's module and folds in the
// standard bridge error check. The single seam every data-plane method funnels
// through; it reuses the generated (m *Module) invoke, the invokers entry-point
// table and the protobuf helpers, so a method body is only its marshalling.
//
// Three kinds of error come back, told apart by type, never by message:
// ErrEngineClosed for a call after Close; a *TrapError when the guest
// trapped; and a *GuestError when the guest reported a failure normally.
func (m *Module) invokeMethod(mid int32, req []byte) ([]byte, error) {
	resp, err := m.invokeGated(0, mid, req, invokers[mid], teardownMethods[mid])
	if err != nil {
		return nil, err
	}
	if e := pbExtractError(resp); e != nil {
		return nil, &GuestError{Err: e}
	}
	return resp, nil
}

// teardownMethods are the calls an engine that trapped still takes: the
// frees, which are what makes the engine safe to release (a context
// free joins its threadpool workers). Everything else is refused after a
// trap; see engineGate.trapped.
var teardownMethods = map[int32]bool{
	midCtxFree:           true,
	midCtxFreeThreadpool: true,
	midModelFree:         true,
	midLoraFree:          true,
	midWasmFree:          true,
}

// invokeGated is invoke under the engine's gate: ErrEngineClosed after
// Close, a *TrapError when the guest trapped (the only error the generated
// invoke produces — its entry points recover a trap into an error and
// nothing else) or when it trapped earlier and this is not a teardown
// call, the raw reply otherwise. Never called from inside a callback the
// guest makes during a gated call: the gate is a read-write lock, and a
// second shared hold on the same goroutine deadlocks against a Close
// waiting to take it exclusively.
func (m *Module) invokeGated(serviceID, methodID int32, req []byte, call func(*base.Module, wptr, wptr) (int64, error), teardown bool) ([]byte, error) {
	g, ok := engineGates.Load(m)
	if !ok {
		return nil, ErrEngineClosed
	}
	gate := g.(*engineGate)
	gate.mu.RLock()
	defer gate.mu.RUnlock()
	if m.Closed() {
		return nil, ErrEngineClosed
	}
	if t := gate.trapped.Load(); t != nil && !teardown {
		return nil, &TrapError{Err: fmt.Errorf("engine trapped earlier: %w", t.Err), Refused: true}
	}
	resp, err := m.invoke(serviceID, methodID, req, call)
	if err != nil {
		trap := &TrapError{Err: err}
		gate.trapped.CompareAndSwap(nil, trap)
		return nil, trap
	}
	return resp, nil
}

// tokenSink is a Token_SinkNode bound to a specific engine: OnPiece and
// teardown route to m, not the global module. The callback service (service 1)
// has only these few RPCs, so they name their Inv_1_* entry points directly
// rather than through the data-plane invokers table.
type tokenSink struct {
	ptr uint64
	m   *Module
}

func (s *tokenSink) rawPtr() uint64 { return s.ptr }
func (s *tokenSink) isToken_Sink()  {}
func (s *tokenSink) OnPiece(piece string) error {
	buf := pbAppendHandle(pbNewBuf(), 1, s.ptr)
	buf = pbAppendString(buf, 2, piece)
	// Ungated on purpose: OnPiece runs inside the generation that owns the
	// sink, which already holds the engine's gate (see invokeGated).
	resp, err := s.m.invoke(1, 0, buf, wasm2go.Inv_1_0)
	if err != nil {
		return err
	}
	return pbExtractError(resp)
}

// NewTokenSink installs a token-sink callback on this engine and returns a
// handle to pass to LlamaCtxGenerate. The finalizer frees the guest object and
// unregisters the callback on the same engine.
func (m *Module) NewTokenSink(impl Token_SinkCallback) (Token_SinkNode, error) {
	adapter := &token_SinkCallbackAdapter{impl: impl}
	id := m.registerCB(adapter)
	buf := pbAppendInt32(pbNewBuf(), 1, id)
	resp, err := m.invokeGated(1, 1, buf, wasm2go.Inv_1_1, false)
	if err == nil {
		err = pbExtractError(resp)
	}
	if err != nil {
		m.unregisterCB(id)
		return nil, err
	}
	s := &tokenSink{ptr: readScalarAtField(resp, 1, (*pbReader).readUint64), m: m}
	runtime.SetFinalizer(s, func(s *tokenSink) {
		// A leaked sink can outlive its engine, and a panic in a finalizer
		// goroutine is fatal — never let the guest-side free escalate.
		defer func() { _ = recover() }()
		if s.ptr != 0 {
			b := pbAppendHandle(pbNewBuf(), 1, s.ptr)
			_, _ = s.m.invokeGated(1, 2, b, wasm2go.Inv_1_2, true)
		}
		s.ptr = 0
		s.m.unregisterCB(id)
	})
	return s, nil
}

func (m *Module) registerCB(handler CallbackHandler) int32 {
	m.cbMu.Lock()
	defer m.cbMu.Unlock()
	if m.callbacks == nil {
		m.callbacks = make(map[int32]CallbackHandler)
	}
	m.nextCBID++
	id := m.nextCBID
	m.callbacks[id] = handler
	return id
}

func (m *Module) unregisterCB(id int32) {
	m.cbMu.Lock()
	delete(m.callbacks, id)
	m.cbMu.Unlock()
}

func (m *Module) LlamaChatApplyTemplate(model uint64, messagesJson string, messagesJsonLen uint32, templateOverride string, templateOverrideLen uint32, addAssistant int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendString(buf, 2, messagesJson)
	buf = pbAppendUint64(buf, 3, uint64(messagesJsonLen))
	buf = pbAppendString(buf, 4, templateOverride)
	buf = pbAppendUint64(buf, 5, uint64(templateOverrideLen))
	buf = pbAppendInt32(buf, 6, addAssistant)
	resp, err := m.invokeMethod(midChatApplyTemplate, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxEmbed(ctx uint64, text string, textLen uint32, normalize int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, text)
	buf = pbAppendUint64(buf, 3, uint64(textLen))
	buf = pbAppendInt32(buf, 4, normalize)
	resp, err := m.invokeMethod(midCtxEmbed, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxEmbedTokens(ctx uint64, tokensJson string, tokensJsonLen uint32, normalize int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, tokensJson)
	buf = pbAppendUint64(buf, 3, uint64(tokensJsonLen))
	buf = pbAppendInt32(buf, 4, normalize)
	resp, err := m.invokeMethod(midCtxEmbedTokens, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxEval(ctx uint64, text string, textLen uint32, addSpecial int32, parseSpecial int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, text)
	buf = pbAppendUint64(buf, 3, uint64(textLen))
	buf = pbAppendInt32(buf, 4, addSpecial)
	buf = pbAppendInt32(buf, 5, parseSpecial)
	resp, err := m.invokeMethod(midCtxEval, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxFree(ctx uint64) error {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	_, err := m.invokeMethod(midCtxFree, buf)
	return err
}

// LlamaCtxDbgTrapNextGraph arms the bridge's test hook: the next graph
// computed on ctx traps on the main thread, mid-graph, with the pool's
// workers waiting at a barrier. For tests of the teardown after a trap.
func (m *Module) LlamaCtxDbgTrapNextGraph(ctx uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxDbgTrapNextGraph, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxFreeThreadpool(ctx uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxFreeThreadpool, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxGenerate(ctx uint64, prompt string, promptLen uint32, paramsJson string, paramsJsonLen uint32, sink Token_SinkNode) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, prompt)
	buf = pbAppendUint64(buf, 3, uint64(promptLen))
	buf = pbAppendString(buf, 4, paramsJson)
	buf = pbAppendUint64(buf, 5, uint64(paramsJsonLen))
	buf = pbAppendHandlePtr(buf, 6, sink)
	resp, err := m.invokeMethod(midCtxGenerate, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxGenerateSpeculative(ctx uint64, draftCtx uint64, prompt string, promptLen uint32, paramsJson string, paramsJsonLen uint32, nDraft int32, sink Token_SinkNode) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendUint64(buf, 2, draftCtx)
	buf = pbAppendString(buf, 3, prompt)
	buf = pbAppendUint64(buf, 4, uint64(promptLen))
	buf = pbAppendString(buf, 5, paramsJson)
	buf = pbAppendUint64(buf, 6, uint64(paramsJsonLen))
	buf = pbAppendInt32(buf, 7, nDraft)
	buf = pbAppendHandlePtr(buf, 8, sink)
	resp, err := m.invokeMethod(midCtxGenerateSpeculative, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxAttachThreadpool(ctx uint64, nThreads uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendUint64(buf, 2, uint64(nThreads))
	resp, err := m.invokeMethod(midCtxAttachThreadpool, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxInterruptAddr(ctx uint64) (uint64, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxInterruptAddr, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint64), nil
}

func (m *Module) LlamaCtxLoraSet(ctx uint64, adaptersJson string, adaptersJsonLen uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, adaptersJson)
	buf = pbAppendUint64(buf, 3, uint64(adaptersJsonLen))
	resp, err := m.invokeMethod(midCtxLoraSet, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxNew(model uint64, paramsJson string, paramsJsonLen uint32) (uint64, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendString(buf, 2, paramsJson)
	buf = pbAppendUint64(buf, 3, uint64(paramsJsonLen))
	resp, err := m.invokeMethod(midCtxNew, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint64), nil
}

func (m *Module) LlamaCtxReset(ctx uint64) error {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	_, err := m.invokeMethod(midCtxReset, buf)
	return err
}

func (m *Module) LlamaCtxScore(ctx uint64, text string, textLen uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, text)
	buf = pbAppendUint64(buf, 3, uint64(textLen))
	resp, err := m.invokeMethod(midCtxScore, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxScoreChoices(ctx uint64, choices string, choicesLen uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, choices)
	buf = pbAppendUint64(buf, 3, uint64(choicesLen))
	resp, err := m.invokeMethod(midCtxScoreChoices, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxSlotsPost(ctx uint64, task string, taskLen uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, task)
	buf = pbAppendUint64(buf, 3, uint64(taskLen))
	resp, err := m.invokeMethod(midCtxSlotsPost, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxSlotsUpdate(ctx uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxSlotsUpdate, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxSlotsCancel(ctx uint64, id int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendInt32(buf, 2, id)
	resp, err := m.invokeMethod(midCtxSlotsCancel, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxSlotsSystemPrompt(ctx uint64, text string, textLen uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, text)
	buf = pbAppendUint64(buf, 3, uint64(textLen))
	resp, err := m.invokeMethod(midCtxSlotsSystemPrompt, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxSlotsStatus(ctx uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxSlotsStatus, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxStateLoad(ctx uint64, data string, size uint32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	buf = pbAppendString(buf, 2, data)
	buf = pbAppendUint64(buf, 3, uint64(size))
	resp, err := m.invokeMethod(midCtxStateLoad, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaCtxStateSave(ctx uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxStateSave, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaDetokenize(model uint64, tokensJson string, tokensJsonLen uint32, renderSpecial int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendString(buf, 2, tokensJson)
	buf = pbAppendUint64(buf, 3, uint64(tokensJsonLen))
	buf = pbAppendInt32(buf, 4, renderSpecial)
	resp, err := m.invokeMethod(midDetokenize, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaLoraFree(adapter uint64) error {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, adapter)
	_, err := m.invokeMethod(midLoraFree, buf)
	return err
}

func (m *Module) LlamaLoraLoad(model uint64, path string, pathLen uint32) (uint64, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendString(buf, 2, path)
	buf = pbAppendUint64(buf, 3, uint64(pathLen))
	resp, err := m.invokeMethod(midLoraLoad, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint64), nil
}

func (m *Module) LlamaModelFree(model uint64) error {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	_, err := m.invokeMethod(midModelFree, buf)
	return err
}

func (m *Module) LlamaModelInfo(model uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	resp, err := m.invokeMethod(midModelInfo, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaModelTensors(model uint64) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	resp, err := m.invokeMethod(midModelTensors, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaModelLoad(path string, pathLen uint32, nGpuLayers int32, useMmap int32) (uint64, error) {
	buf := pbNewBuf()
	buf = pbAppendString(buf, 1, path)
	buf = pbAppendUint64(buf, 2, uint64(pathLen))
	buf = pbAppendInt32(buf, 3, nGpuLayers)
	buf = pbAppendInt32(buf, 4, useMmap)
	resp, err := m.invokeMethod(midModelLoad, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint64), nil
}

func (m *Module) LlamaModelLoadProgressAddr() (uint64, error) {
	buf := pbNewBuf()
	resp, err := m.invokeMethod(midModelLoadProgressAddr, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint64), nil
}

func (m *Module) LlamaTokenToPiece(model uint64, token int32, renderSpecial int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendInt32(buf, 2, token)
	buf = pbAppendInt32(buf, 3, renderSpecial)
	resp, err := m.invokeMethod(midTokenToPiece, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaTokenize(model uint64, text string, textLen uint32, addSpecial int32, parseSpecial int32) (string, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, model)
	buf = pbAppendString(buf, 2, text)
	buf = pbAppendUint64(buf, 3, uint64(textLen))
	buf = pbAppendInt32(buf, 4, addSpecial)
	buf = pbAppendInt32(buf, 5, parseSpecial)
	resp, err := m.invokeMethod(midTokenize, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaWasmBuildInfo() (string, error) {
	buf := pbNewBuf()
	resp, err := m.invokeMethod(midWasmBuildInfo, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}

func (m *Module) LlamaWasmFree() error {
	buf := pbNewBuf()
	_, err := m.invokeMethod(midWasmFree, buf)
	return err
}

func (m *Module) LlamaWasmInit() error {
	buf := pbNewBuf()
	_, err := m.invokeMethod(midWasmInit, buf)
	return err
}

func (m *Module) LlamaWasmLastError() (string, error) {
	buf := pbNewBuf()
	resp, err := m.invokeMethod(midWasmLastError, buf)
	if err != nil {
		return "", err
	}
	return readScalarAtField(resp, 1, (*pbReader).readString), nil
}
