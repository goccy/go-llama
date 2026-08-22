package llama

// The API layered on top of the generated binding in internal.
//
// A Llama is one engine instance: llama.cpp compiled to wasm and translated to
// Go, with its own linear memory and C heap. Create one with New, load models
// into it, and open contexts over those models. Several Llama instances are
// fully independent — each owns its own memory — so they run concurrently on
// different goroutines. Within one instance the generated bridge serialises
// every call (one C stack), and several models / contexts share it, which is
// how llama.cpp is meant to be used: the weights are loaded once and each
// context keeps its own KV cache.

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"

	bridge "github.com/goccy/go-llama/internal"
	"github.com/goccy/llamawasm2go/base"
)

// FS is the filesystem the guest sees when Config.FS is set. base.NewMemFS
// returns a ready-made in-memory implementation, so a model can be served out
// of memory rather than off disk.
type FS = base.FS

// Llama is one engine instance: llama.cpp compiled to wasm, with its own
// view of linear memory and its own C heap. Load models into it and open
// contexts over them. Instances are independent — several run concurrently —
// while within one instance calls are serialised.
//
// Memory an instance never writes is physically shared with the other
// instances in the process: engines are built over copy-on-write maps of a
// process-wide image, and instances that load a model an earlier instance
// already loaded (same options, same path) share the weights the same way
// instead of loading them again. Set GO_LLAMA_NO_SHARED_IMAGE to force every
// instance onto a private allocation.
type Llama struct {
	// eng is swapped exactly once — at the first LoadModel, when a shared
	// model snapshot replaces the freshly started engine — so readers go
	// through e() and the swap is atomic.
	eng    atomic.Pointer[bridge.Module]
	cfg    config
	mu     sync.Mutex // guards the engine swap and models during LoadModel
	models int        // models loaded into the current engine
	closed atomic.Bool
}

// e is the engine accessor every method funnels through.
func (l *Llama) e() *bridge.Module { return l.eng.Load() }

// config holds an instance's resolved options.
type config struct {
	preopenDir         string
	fs                 FS
	env                []string
	stdout, stderr     io.Writer
	memoryReserveBytes int
	maxMemoryBytes     uint64
}

// An Option configures a Llama at New time.
type Option func(*config)

// WithPreopenDir scopes the guest filesystem to a host directory. Guest paths
// then resolve inside it, so a model is named relative to it. Without it the
// guest sees the whole host filesystem, where absolute paths work as written.
func WithPreopenDir(dir string) Option { return func(c *config) { c.preopenDir = dir } }

// WithFS replaces the filesystem backend entirely — feed a model from memory
// or a virtual tree. It takes precedence over WithPreopenDir. base.NewMemFS
// returns a ready-made in-memory implementation.
func WithFS(fs FS) Option { return func(c *config) { c.fs = fs } }

// WithEnv sets the environment the guest sees. Without it the guest gets an
// empty environment: the host process environment is never leaked.
func WithEnv(env []string) Option { return func(c *config) { c.env = env } }

// WithStdout and WithStderr capture what the guest writes to fd 1 and fd 2.
// Unset discards, the default, because llama.cpp's own logging is silenced
// inside the bridge anyway.
func WithStdout(w io.Writer) Option { return func(c *config) { c.stdout = w } }
func WithStderr(w io.Writer) Option { return func(c *config) { c.stderr = w } }

// WithMemoryReserve pre-reserves linear memory. Loading a multi-gigabyte model
// otherwise grows memory in steps, copying it forward each time; reserving up
// front avoids that. Zero uses the engine's default.
func WithMemoryReserve(bytes int) Option { return func(c *config) { c.memoryReserveBytes = bytes } }

// WithMaxMemory caps linear-memory growth, so a model bigger than expected
// fails inside the guest instead of growing the host process. Zero means the
// engine's default ceiling: for an instance backed by a copy-on-write mapping
// that is the 64 GiB of address space the mapping reserves (untouched pages
// cost nothing), and for a private allocation there is no cap.
func WithMaxMemory(bytes uint64) Option { return func(c *config) { c.maxMemoryBytes = bytes } }

// New brings up an engine instance. The zero-option instance gives the guest
// the host filesystem, an empty environment and discarded stdio — the
// filesystem access a Go program reading a model file already has.
func New(opts ...Option) (*Llama, error) {
	var cfg config
	for _, o := range opts {
		o(&cfg)
	}
	eng, err := bridge.NewEngine(cfg.bridgeOptions(false))
	if err != nil {
		return nil, fmt.Errorf("llama: start engine: %w", err)
	}
	l := &Llama{cfg: cfg}
	l.eng.Store(eng)
	// The engine may own a copy-on-write mapping, which is not Go heap: if the
	// instance is dropped without Close, unmap it when the GC finds it.
	runtime.SetFinalizer(l, func(l *Llama) { l.Close() })
	return l, nil
}

// wasi builds the instance's WASI from its resolved options. Each engine gets
// a fresh one — WASI carries per-engine filesystem state, so an engine swapped
// in at LoadModel must not inherit the previous engine's.
func (c *config) wasi(discardIO bool) *base.WasiStubs {
	wasi := base.DefaultWASI()
	wasi.SetEnv(c.env)
	switch {
	case c.fs != nil:
		wasi.SetFS(c.fs)
	case c.preopenDir != "":
		wasi.SetPreopenDir(c.preopenDir)
	}
	if c.stdout != nil && !discardIO {
		wasi.SetStdout(c.stdout)
	} else {
		wasi.SetStdout(io.Discard)
	}
	if c.stderr != nil && !discardIO {
		wasi.SetStderr(c.stderr)
	} else {
		wasi.SetStderr(io.Discard)
	}
	return wasi
}

// bridgeOptions is the internal engine configuration for these options.
// discardIO drops the caller's stdio writers — what a shared snapshot builder
// wants, since its output belongs to no particular instance.
func (c *config) bridgeOptions(discardIO bool) bridge.Options {
	return bridge.Options{
		WASI:               c.wasi(discardIO),
		MemoryReserveBytes: c.memoryReserveBytes,
		MaxMemoryBytes:     c.maxMemoryBytes,
	}
}

// Close releases the instance, including the copy-on-write memory mapping when
// the engine is backed by one. Models and contexts opened from it must be
// closed first; using any of them afterward fails — the engine's memory is
// gone.
func (l *Llama) Close() error {
	if l.closed.Swap(true) {
		return nil
	}
	runtime.SetFinalizer(l, nil)
	if eng := l.e(); eng != nil {
		eng.Close()
	}
	return nil
}

func (l *Llama) lastError() string {
	msg, err := l.e().LlamaWasmLastError()
	if err != nil {
		return err.Error()
	}
	if msg == "" {
		return "unknown error"
	}
	return msg
}

// ContextParams configures a Context. The zero value asks for the model's own
// training context length and llama.cpp's batch defaults.
type ContextParams struct {
	// NCtx is the context window in tokens. Zero means the model's training
	// context length. The KV cache scales with it.
	NCtx uint32
	// NBatch and NUBatch are the logical and physical batch sizes. Zero means
	// llama.cpp's defaults.
	NBatch  uint32
	NUBatch uint32
	// NThreads is how many threads ggml may use. It has an effect only in a
	// threads-enabled build (BuildInfo.Threads); the single-threaded wasm
	// clamps to 1.
	NThreads uint32
	// Embeddings puts the context in embedding mode, which Context.Embed
	// requires and which disables generation.
	Embeddings bool
	// RopeFreqBase and RopeFreqScale override the model's RoPE
	// configuration, the knobs behind context-length extension schemes.
	// Zero keeps the model's own values.
	RopeFreqBase  float32
	RopeFreqScale float32
}

// ctxRequest is ContextParams' wire form; like genRequest, absent
// means the engine's default.
type ctxRequest struct {
	NCtx          uint32  `json:"n_ctx,omitempty"`
	NBatch        uint32  `json:"n_batch,omitempty"`
	NUBatch       uint32  `json:"n_ubatch,omitempty"`
	NThreads      uint32  `json:"n_threads,omitempty"`
	Embeddings    int     `json:"embeddings,omitempty"`
	RopeFreqBase  float32 `json:"rope_freq_base,omitempty"`
	RopeFreqScale float32 `json:"rope_freq_scale,omitempty"`
}

// Model is a loaded GGUF model. Several contexts can share one, and several
// models can be loaded into one Llama instance at once.
type Model struct {
	inst   *Llama
	h      uint64
	closed atomic.Bool
}

// Context is one inference context over a Model: its own KV cache and its own
// sampling state. Independent conversations get independent contexts over the
// same weights.
type Context struct {
	model *Model
	h     uint64

	// interruptAddr is the linear-memory address of this context's interrupt
	// flag, resolved once at creation so Interrupt never has to call into the
	// engine — which it could not do while a generation is inside it.
	interruptAddr uint32

	closed atomic.Bool
}

// LoadModel loads a GGUF model from path into this instance.
//
// path is a guest path. Without WithPreopenDir / WithFS the guest sees the
// host filesystem, so a relative path is resolved against the working
// directory first; with a scoped filesystem it is used as given, relative to
// that root.
func (l *Llama) LoadModel(path string) (*Model, error) {
	if l.closed.Load() {
		return nil, errors.New("llama: instance is closed")
	}
	guestPath := path
	if l.cfg.fs == nil && l.cfg.preopenDir == "" {
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("llama: resolve model path: %w", err)
		}
		guestPath = abs
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	// Shared fast path: when this is the instance's first model and the
	// filesystem configuration is keyable (a host directory, not an arbitrary
	// FS), the process keeps one copy-on-write snapshot of a started engine
	// with this model in memory. Every instance after the first maps it
	// instead of loading the weights again; the first pays one extra copy of
	// the loaded memory into the image. Any failure here falls back to the
	// private load below.
	if l.models == 0 && l.cfg.fs == nil {
		key := fmt.Sprintf("dir=%q|env=%q|model=%q", l.cfg.preopenDir, l.cfg.env, guestPath)
		snap := bridge.SharedModelSnapshot(key, l.cfg.bridgeOptions(true),
			func(m *bridge.Module) (uint64, error) {
				return m.LlamaModelLoad(guestPath, uint32(len(guestPath)), 0, 0)
			})
		if snap.Err() == nil {
			if eng, err := bridge.NewEngineFromModelSnapshot(snap, l.cfg.bridgeOptions(false)); err == nil {
				old := l.eng.Swap(eng)
				old.Close()
				l.models++
				return &Model{inst: l, h: snap.Handle()}, nil
			}
		}
	}

	h, err := l.e().LlamaModelLoad(guestPath, uint32(len(guestPath)), 0, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: load %s: %w", path, err)
	}
	if h == 0 {
		return nil, fmt.Errorf("llama: load %s: %s", path, l.lastError())
	}
	l.models++
	return &Model{inst: l, h: h}, nil
}

// BuildInfo reports how the embedded engine was compiled.
func (l *Llama) BuildInfo() (Build, error) {
	if l.closed.Load() {
		return Build{}, errors.New("llama: instance is closed")
	}
	js, err := l.e().LlamaWasmBuildInfo()
	if err != nil {
		return Build{}, err
	}
	var out struct {
		envelope
		Build
	}
	if err := decode("build info", js, &out); err != nil {
		return Build{}, err
	}
	return out.Build, nil
}

// use reports whether the model is still usable. The handle is a raw C++
// pointer, so calling on through it after Close would be a use-after-free
// inside the engine rather than a Go error.
func (m *Model) use(what string) error {
	if m.closed.Load() {
		return fmt.Errorf("llama: %s: model is closed", what)
	}
	if m.inst.closed.Load() {
		return fmt.Errorf("llama: %s: instance is closed", what)
	}
	return nil
}

// Close frees the model. Contexts created from it must be closed first.
func (m *Model) Close() error {
	if m.closed.Swap(true) {
		return nil
	}
	return m.inst.e().LlamaModelFree(m.h)
}

// Info returns the model's metadata.
func (m *Model) Info() (ModelInfo, error) {
	if err := m.use("model info"); err != nil {
		return ModelInfo{}, err
	}
	js, err := m.inst.e().LlamaModelInfo(m.h)
	if err != nil {
		return ModelInfo{}, err
	}
	var out struct {
		envelope
		ModelInfo
	}
	if err := decode("model info", js, &out); err != nil {
		return ModelInfo{}, err
	}
	return out.ModelInfo, nil
}

// Tokenize splits text into tokens. addSpecial adds the model's BOS/EOS
// convention; parseSpecial lets special-token text ("<|im_start|>") tokenize
// as that token rather than as its characters.
func (m *Model) Tokenize(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	if err := m.use("tokenize"); err != nil {
		return nil, err
	}
	js, err := m.inst.e().LlamaTokenize(m.h, text, uint32(len(text)), b2i(addSpecial), b2i(parseSpecial))
	if err != nil {
		return nil, err
	}
	var out struct {
		envelope
		Tokens []int32 `json:"tokens"`
	}
	if err := decode("tokenize", js, &out); err != nil {
		return nil, err
	}
	return out.Tokens, nil
}

// Detokenize renders tokens back to text.
func (m *Model) Detokenize(tokens []int32, renderSpecial bool) (string, error) {
	if err := m.use("detokenize"); err != nil {
		return "", err
	}
	raw, err := json.Marshal(tokens)
	if err != nil {
		return "", err
	}
	js, err := m.inst.e().LlamaDetokenize(m.h, string(raw), uint32(len(raw)), b2i(renderSpecial))
	if err != nil {
		return "", err
	}
	return decodeText("detokenize", js)
}

// TokenToPiece renders one token. A byte-level token can render to invalid
// UTF-8 on its own; accumulate pieces before treating the result as text.
func (m *Model) TokenToPiece(token int32, renderSpecial bool) (string, error) {
	if err := m.use("token_to_piece"); err != nil {
		return "", err
	}
	js, err := m.inst.e().LlamaTokenToPiece(m.h, token, b2i(renderSpecial))
	if err != nil {
		return "", err
	}
	var out struct {
		envelope
		Text string `json:"text"`
		B64  string `json:"b64"`
	}
	if err := decode("token_to_piece", js, &out); err != nil {
		return "", err
	}
	// A byte-fallback token holds a partial UTF-8 sequence, which the JSON
	// text field cannot carry losslessly; the base64 field carries the raw
	// bytes exactly as llama.cpp's llama_token_to_piece returns them.
	if out.B64 != "" {
		raw, err := base64.StdEncoding.DecodeString(out.B64)
		if err != nil {
			return "", fmt.Errorf("token_to_piece: bad b64 payload: %w", err)
		}
		return string(raw), nil
	}
	return out.Text, nil
}

// ApplyChatTemplate renders messages into a prompt with the model's chat
// template. templateOverride replaces it, and is required when the GGUF
// carries none; addAssistant appends the generation prefix.
func (m *Model) ApplyChatTemplate(messages []Message, templateOverride string, addAssistant bool) (string, error) {
	if err := m.use("chat template"); err != nil {
		return "", err
	}
	raw, err := json.Marshal(messages)
	if err != nil {
		return "", err
	}
	js, err := m.inst.e().LlamaChatApplyTemplate(m.h, string(raw), uint32(len(raw)),
		templateOverride, uint32(len(templateOverride)), b2i(addAssistant))
	if err != nil {
		return "", err
	}
	var out struct {
		envelope
		Prompt string `json:"prompt"`
	}
	if err := decode("chat template", js, &out); err != nil {
		return "", err
	}
	return out.Prompt, nil
}

// LoRA is a loaded LoRA adapter. It adapts contexts over the model that
// loaded it (Context.SetLoRA) and must not outlive that model.
type LoRA struct {
	model  *Model
	h      uint64
	closed atomic.Bool
}

// LoadLoRA loads a LoRA adapter GGUF for this model. path resolves like
// LoadModel's.
func (m *Model) LoadLoRA(path string) (*LoRA, error) {
	if err := m.use("load lora"); err != nil {
		return nil, err
	}
	guestPath := path
	if m.inst.cfg.fs == nil && m.inst.cfg.preopenDir == "" {
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("llama: resolve lora path: %w", err)
		}
		guestPath = abs
	}
	h, err := m.inst.e().LlamaLoraLoad(m.h, guestPath, uint32(len(guestPath)))
	if err != nil {
		return nil, err
	}
	if h == 0 {
		return nil, fmt.Errorf("llama: load lora %s: %s", path, m.inst.lastError())
	}
	return &LoRA{model: m, h: h}, nil
}

// Close frees the adapter. Contexts still using it must clear it first.
func (l *LoRA) Close() error {
	if l.closed.Swap(true) {
		return nil
	}
	return l.model.inst.e().LlamaLoraFree(l.h)
}

// LoRAWeight pairs an adapter with the scale to apply it at.
type LoRAWeight struct {
	Adapter *LoRA
	Scale   float32
}

// SetLoRA replaces the context's ENTIRE adapter configuration — the set
// semantics of llama.cpp's own API. An empty (or nil) slice removes every
// adapter, so SetLoRA(nil) is the spelling of "clear".
func (c *Context) SetLoRA(adapters []LoRAWeight) error {
	if err := c.use("set lora"); err != nil {
		return err
	}
	pairs := make([][2]float64, len(adapters))
	for i, a := range adapters {
		if a.Adapter == nil || a.Adapter.closed.Load() {
			return errors.New("llama: set lora: nil or closed adapter")
		}
		pairs[i] = [2]float64{float64(a.Adapter.h), float64(a.Scale)}
	}
	raw, err := json.Marshal(pairs)
	if err != nil {
		return err
	}
	js, err := c.model.inst.e().LlamaCtxLoraSet(c.h, string(raw), uint32(len(raw)))
	if err != nil {
		return err
	}
	var out struct{ envelope }
	return decode("set lora", js, &out)
}

// NewContext creates an inference context over the model.
func (m *Model) NewContext(p ContextParams) (*Context, error) {
	if err := m.use("new context"); err != nil {
		return nil, err
	}
	raw, err := json.Marshal(ctxRequest{
		NCtx:          p.NCtx,
		NBatch:        p.NBatch,
		NUBatch:       p.NUBatch,
		NThreads:      p.NThreads,
		Embeddings:    int(b2i(p.Embeddings)),
		RopeFreqBase:  p.RopeFreqBase,
		RopeFreqScale: p.RopeFreqScale,
	})
	if err != nil {
		return nil, err
	}
	h, err := m.inst.e().LlamaCtxNew(m.h, string(raw), uint32(len(raw)))
	if err != nil {
		return nil, err
	}
	if h == 0 {
		return nil, fmt.Errorf("llama: new context: %s", m.inst.lastError())
	}
	c := &Context{model: m, h: h}
	if c.interruptAddr, err = m.inst.e().LlamaCtxInterruptAddr(h); err != nil {
		return nil, err
	}
	if c.interruptAddr == 0 {
		return nil, errors.New("llama: new context: engine reported no interrupt address")
	}
	return c, nil
}

// use reports whether the context is still usable, and that its model is
// too: a context outliving its model holds a dangling llama_context.
func (c *Context) use(what string) error {
	if c.closed.Load() {
		return fmt.Errorf("llama: %s: context is closed", what)
	}
	return c.model.use(what)
}

// Close frees the context. The Model outlives it.
func (c *Context) Close() error {
	if c.closed.Swap(true) {
		return nil
	}
	return c.model.inst.e().LlamaCtxFree(c.h)
}

// Generate runs generation from prompt and returns the whole result.
func (c *Context) Generate(prompt string, p Params) (Result, error) {
	return c.generate(prompt, p.wire(), nil)
}

// GenerateWithDraft is Generate accelerated by speculative decoding: a
// draft context over a smaller model with the SAME vocabulary proposes up
// to nDraft tokens per round (0 picks a default) and this context verifies
// them in one batch. Every emitted token is sampled by THIS context's
// sampler chain, so the output distribution is exactly Generate's — the
// draft only trades its cheap decodes for larger verification batches.
//
// Both contexts' KV caches restart from the prompt. Result.NDrafted and
// Result.NAccepted report how well the draft anticipated the target.
func (c *Context) GenerateWithDraft(draft *Context, prompt string, p Params, nDraft int) (Result, error) {
	if err := c.use("generate"); err != nil {
		return Result{}, err
	}
	if draft == nil {
		return Result{}, errors.New("llama: generate with draft: nil draft context")
	}
	if err := draft.use("generate (draft)"); err != nil {
		return Result{}, err
	}
	raw, err := json.Marshal(p.wire())
	if err != nil {
		return Result{}, err
	}
	js, err := c.model.inst.e().LlamaCtxGenerateSpeculative(c.h, draft.h, prompt, uint32(len(prompt)),
		string(raw), uint32(len(raw)), int32(nDraft), nil)
	if err != nil {
		return Result{}, err
	}
	var out struct {
		envelope
		Result
	}
	if err := decode("generate", js, &out); err != nil {
		return Result{}, err
	}
	return out.Result, nil
}

// Reset drops the KV cache so the next generation starts from nothing.
func (c *Context) Reset() error {
	if err := c.use("reset"); err != nil {
		return err
	}
	return c.model.inst.e().LlamaCtxReset(c.h)
}

// Embed returns the embedding of text. The context must have been created with
// ContextParams.Embeddings set. normalize applies L2 normalisation.
func (c *Context) Embed(text string, normalize bool) ([]float32, error) {
	if err := c.use("embed"); err != nil {
		return nil, err
	}
	js, err := c.model.inst.e().LlamaCtxEmbed(c.h, text, uint32(len(text)), b2i(normalize))
	if err != nil {
		return nil, err
	}
	var out struct {
		envelope
		Embedding []float32 `json:"embedding"`
	}
	if err := decode("embed", js, &out); err != nil {
		return nil, err
	}
	return out.Embedding, nil
}

// EvalResult reports what Eval put into the KV cache: NTokens from this
// call, NPast the total sequence length now cached.
type EvalResult struct {
	NTokens int32 `json:"n_tokens"`
	NPast   int32 `json:"n_past"`
}

// Eval decodes text into the context's KV cache without sampling —
// prompt prefill for a later Generate on the same context, or plain
// evaluation. Positions continue from the cache's current end; Reset
// starts over. addSpecial and parseSpecial mirror Tokenize.
func (c *Context) Eval(text string, addSpecial, parseSpecial bool) (EvalResult, error) {
	if err := c.use("eval"); err != nil {
		return EvalResult{}, err
	}
	js, err := c.model.inst.e().LlamaCtxEval(c.h, text, uint32(len(text)), b2i(addSpecial), b2i(parseSpecial))
	if err != nil {
		return EvalResult{}, err
	}
	var out struct {
		envelope
		EvalResult
	}
	if err := decode("eval", js, &out); err != nil {
		return EvalResult{}, err
	}
	return out.EvalResult, nil
}

// EmbedTokens is Embed from token ids instead of text.
func (c *Context) EmbedTokens(tokens []int32, normalize bool) ([]float32, error) {
	if err := c.use("embed tokens"); err != nil {
		return nil, err
	}
	raw, err := json.Marshal(tokens)
	if err != nil {
		return nil, err
	}
	js, err := c.model.inst.e().LlamaCtxEmbedTokens(c.h, string(raw), uint32(len(raw)), b2i(normalize))
	if err != nil {
		return nil, err
	}
	var out struct {
		envelope
		Embedding []float32 `json:"embedding"`
	}
	if err := decode("embed tokens", js, &out); err != nil {
		return nil, err
	}
	return out.Embedding, nil
}

// SaveState serializes the context's state — KV cache and sampling RNG —
// into a byte slice that LoadState can restore, also in a later process.
// Save a long system prompt's state once and every future context skips
// re-decoding it.
func (c *Context) SaveState() ([]byte, error) {
	if err := c.use("save state"); err != nil {
		return nil, err
	}
	js, err := c.model.inst.e().LlamaCtxStateSave(c.h)
	if err != nil {
		return nil, err
	}
	var out struct {
		envelope
		Addr uint32 `json:"addr"`
		Size uint32 `json:"size"`
	}
	if err := decode("save state", js, &out); err != nil {
		return nil, err
	}
	m := c.model.inst.e().Base()
	if m == nil {
		return nil, errors.New("llama: save state: engine is not running")
	}
	state := make([]byte, out.Size)
	var cpErr error
	// AccessMemory holds the lock memory.grow takes, so the guest buffer
	// can neither move nor be resliced during the copy.
	base.AccessMemory(m, func(mem []byte) {
		end := uint64(out.Addr) + uint64(out.Size)
		if end > uint64(len(mem)) {
			cpErr = fmt.Errorf("llama: save state: %d bytes at %d are outside linear memory", out.Size, out.Addr)
			return
		}
		copy(state, mem[out.Addr:end])
	})
	if cpErr != nil {
		return nil, cpErr
	}
	return state, nil
}

// LoadState restores a state produced by SaveState. The context must be
// over the same model with compatible parameters, or the engine rejects
// the payload.
func (c *Context) LoadState(state []byte) error {
	if err := c.use("load state"); err != nil {
		return err
	}
	js, err := c.model.inst.e().LlamaCtxStateLoad(c.h, string(state), uint32(len(state)))
	if err != nil {
		return err
	}
	var out struct{ envelope }
	return decode("load state", js, &out)
}

// ScoreResult reports the teacher-forced negative log-likelihood of a text:
// the model decodes the tokenized text once and NLL sums
// -log softmax(logits_i)[token_{i+1}] over the NScored predicting positions.
// Perplexity is math.Exp(NLL / NScored).
type ScoreResult struct {
	NTokens int32   `json:"n_tokens"`
	NScored int32   `json:"n_scored"`
	NLL     float64 `json:"nll"`
}

// Score computes the teacher-forced negative log-likelihood of text under the
// model, the quantity llama.cpp's perplexity tool averages. It decodes text
// into the context's KV cache; call Reset before generating afterwards.
func (c *Context) Score(text string) (ScoreResult, error) {
	if err := c.use("score"); err != nil {
		return ScoreResult{}, err
	}
	js, err := c.model.inst.e().LlamaCtxScore(c.h, text, uint32(len(text)))
	if err != nil {
		return ScoreResult{}, err
	}
	var out struct {
		envelope
		ScoreResult
	}
	if err := decode("score", js, &out); err != nil {
		return ScoreResult{}, err
	}
	return out.ScoreResult, nil
}

// generate is the one path into the engine's generation loop. sink is nil for
// a plain Generate and the caller's piece sink for Stream.
func (c *Context) generate(prompt string, req genRequest, sink bridge.Token_SinkNode) (Result, error) {
	if err := c.use("generate"); err != nil {
		return Result{}, err
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return Result{}, err
	}
	js, err := c.model.inst.e().LlamaCtxGenerate(c.h, prompt, uint32(len(prompt)), string(raw), uint32(len(raw)), sink)
	if err != nil {
		return Result{}, err
	}
	var out struct {
		envelope
		Result
	}
	if err := decode("generate", js, &out); err != nil {
		return Result{}, err
	}
	return out.Result, nil
}

// b2i converts a Go bool to the int32 flag the C bridge takes.
func b2i(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// lastError returns the engine's message for the most recent failed call —
// the reason behind a zero handle, which is all a handle-returning export can
// signal on its own.

// decode unmarshals a bridge result and turns a not-ok envelope into an error.
// what names the operation; the raw document is quoted on a parse failure so a
// malformed result is diagnosable.
func decode(what, js string, v enveloped) error {
	if err := json.Unmarshal([]byte(js), v); err != nil {
		return fmt.Errorf("llama: %s: decode result %q: %w", what, js, err)
	}
	return v.err(what)
}

// decodeText is the common shape of the calls that return just a string.
func decodeText(what, js string) (string, error) {
	var out struct {
		envelope
		Text string `json:"text"`
	}
	if err := decode(what, js, &out); err != nil {
		return "", err
	}
	return out.Text, nil
}
