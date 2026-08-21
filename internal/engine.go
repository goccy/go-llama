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
	"fmt"
	"runtime"

	wasm2go "github.com/goccy/llamawasm2go"
	"github.com/goccy/llamawasm2go/base"
)

// Method IDs for service 0 (the data plane), in the alphabetical order the
// bridge numbers them. Their ordinal value IS the generated method number, so
// the iota order must match llama.go's Inv_0_0, Inv_0_1, ... exactly.
const (
	midChatApplyTemplate int32 = iota
	midCtxEmbed
	midCtxEmbedTokens
	midCtxEval
	midCtxFree
	midCtxGenerate
	midCtxGenerateSpeculative
	midCtxInterruptAddr
	midCtxLoraSet
	midCtxNew
	midCtxReset
	midCtxScore
	midCtxStateLoad
	midCtxStateSave
	midDetokenize
	midLoraFree
	midLoraLoad
	midModelFree
	midModelInfo
	midModelLoad
	midModelLoadProgressAddr
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
	midCtxEmbed:               wasm2go.Inv_0_1,
	midCtxEmbedTokens:         wasm2go.Inv_0_2,
	midCtxEval:                wasm2go.Inv_0_3,
	midCtxFree:                wasm2go.Inv_0_4,
	midCtxGenerate:            wasm2go.Inv_0_5,
	midCtxGenerateSpeculative: wasm2go.Inv_0_6,
	midCtxInterruptAddr:       wasm2go.Inv_0_7,
	midCtxLoraSet:             wasm2go.Inv_0_8,
	midCtxNew:                 wasm2go.Inv_0_9,
	midCtxReset:               wasm2go.Inv_0_10,
	midCtxScore:               wasm2go.Inv_0_11,
	midCtxStateLoad:           wasm2go.Inv_0_12,
	midCtxStateSave:           wasm2go.Inv_0_13,
	midDetokenize:             wasm2go.Inv_0_14,
	midLoraFree:               wasm2go.Inv_0_15,
	midLoraLoad:               wasm2go.Inv_0_16,
	midModelFree:              wasm2go.Inv_0_17,
	midModelInfo:              wasm2go.Inv_0_18,
	midModelLoad:              wasm2go.Inv_0_19,
	midModelLoadProgressAddr:  wasm2go.Inv_0_20,
	midTokenToPiece:           wasm2go.Inv_0_21,
	midTokenize:               wasm2go.Inv_0_22,
	midWasmBuildInfo:          wasm2go.Inv_0_23,
	midWasmFree:               wasm2go.Inv_0_24,
	midWasmInit:               wasm2go.Inv_0_25,
	midWasmLastError:          wasm2go.Inv_0_26,
}

// NewEngine brings up an independent engine instance: its own wasm module
// (its own view of linear memory / C heap), configured by opts. The memory is
// a copy-on-write map of the process-wide data-segment image when one is
// available (see sharedengine.go), a private allocation otherwise; either way
// the instance runs its own initialization with its own WASI.
func NewEngine(opts Options) (m *Module, err error) {
	img := sharedEngineImage()
	mem, imgErr := img.Memory(memoryCeiling(opts))
	if imgErr != nil {
		return NewPrivateEngine(opts)
	}
	m = &Module{}
	m.g = wasm2go.NewWithMemory(engineWASI(opts), envStubs{m: m}, wasmifyStubs{m: m},
		mem, img.Size())
	m.mmapped = mem
	if err := initEngine(m); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

// NewPrivateEngine is NewEngine without the shared image: the instance's
// memory is its own allocation. The fallback when copy-on-write mapping is
// unavailable, and what a snapshot builder uses on purpose.
func NewPrivateEngine(opts Options) (m *Module, err error) {
	m = &Module{}
	env := envStubs{m: m}
	wm := wasmifyStubs{m: m}
	wasi := engineWASI(opts)
	if opts.MemoryReserveBytes > 0 {
		m.g = wasm2go.NewWithWASIReserve(wasi, env, wm, opts.MemoryReserveBytes)
	} else {
		m.g = wasm2go.NewWithWASI(wasi, env, wm)
	}
	if opts.MaxMemoryBytes > 0 {
		wasm2go.SetMaxMemory(m.g, opts.MaxMemoryBytes)
	}
	if err := initEngine(m); err != nil {
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

// Close releases the engine's memory. An engine backed by a copy-on-write
// mapping owns that mapping — it is not Go heap, so it must be unmapped here
// rather than left to the GC; a private allocation is simply detached for the
// GC to reclaim. Either way the module is left memoryless, so a late call (a
// leaked handle's finalizer, a use-after-close) errors in invoke instead of
// touching freed pages. Idempotent.
func (m *Module) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.g == nil || m.g.Memory == nil {
		return
	}
	mem := m.mmapped
	m.mmapped = nil
	// Detach the module from its memory before releasing it, so a stray late
	// call fails the invoke closed-check instead of touching unmapped pages.
	m.g.Memory = nil
	m.g.M = nil
	m.g.MemSize.Store(0)
	if mem != nil {
		base.UnmapMemory(mem)
	}
}

// ImageBacked reports whether this engine's memory is a copy-on-write map of
// a shared image (data-segment or snapshot) rather than a private allocation.
func (m *Module) ImageBacked() bool { return m.mmapped != nil }

// Base returns the engine's transpiled module, for base.AccessMemory (the only
// safe way to touch linear memory from another goroutine — an interrupt flag
// or a polled progress word).
func (m *Module) Base() *base.Module { return m.g }

// invokeMethod runs one service-0 RPC on THIS engine's module and folds in the
// standard bridge error check. The single seam every data-plane method funnels
// through; it reuses the generated (m *Module) invoke, the invokers entry-point
// table and the protobuf helpers, so a method body is only its marshalling.
func (m *Module) invokeMethod(mid int32, req []byte) ([]byte, error) {
	resp, err := m.invoke(0, mid, req, invokers[mid])
	if err != nil {
		return nil, err
	}
	if e := pbExtractError(resp); e != nil {
		return nil, e
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
	resp, err := m.invoke(1, 1, buf, wasm2go.Inv_1_1)
	if err == nil {
		err = pbExtractError(resp)
	}
	if err != nil {
		m.unregisterCB(id)
		return nil, err
	}
	s := &tokenSink{ptr: readScalarAtField(resp, 1, (*pbReader).readUint64), m: m}
	runtime.SetFinalizer(s, func(s *tokenSink) {
		if s.ptr != 0 {
			b := pbAppendHandle(pbNewBuf(), 1, s.ptr)
			_, _ = s.m.invoke(1, 2, b, wasm2go.Inv_1_2)
			s.ptr = 0
		}
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

func (m *Module) LlamaCtxInterruptAddr(ctx uint64) (uint32, error) {
	buf := pbNewBuf()
	buf = pbAppendUint64(buf, 1, ctx)
	resp, err := m.invokeMethod(midCtxInterruptAddr, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint32), nil
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

func (m *Module) LlamaModelLoadProgressAddr() (uint32, error) {
	buf := pbNewBuf()
	resp, err := m.invokeMethod(midModelLoadProgressAddr, buf)
	if err != nil {
		return 0, err
	}
	return readScalarAtField(resp, 1, (*pbReader).readUint32), nil
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
