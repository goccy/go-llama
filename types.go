package llama

import (
	"fmt"
	"math"
	"slices"
)

// The decoded forms of the JSON documents the wasm bridge returns. The bridge
// bundles each call's outputs into one JSON object so a call is one round trip
// and one atomic result; these types are the Go side of that contract, and
// their field tags must match llama_api.h.

// ModelInfo describes a loaded model. Everything here comes from the GGUF
// metadata, so a field is zero when the file does not carry it.
type ModelInfo struct {
	// Desc is llama.cpp's own one-line description, e.g. "llama 7B Q4_K - Medium".
	Desc string `json:"desc"`
	// NParams is the parameter count; SizeBytes the on-disk tensor size.
	NParams   uint64 `json:"n_params"`
	SizeBytes uint64 `json:"size_bytes"`
	// NCtxTrain is the context length the model was trained with — the
	// largest context worth asking for.
	NCtxTrain int `json:"n_ctx_train"`
	NEmbd     int `json:"n_embd"`
	NLayer    int `json:"n_layer"`
	NHead     int `json:"n_head"`
	NHeadKV   int `json:"n_head_kv"`
	NVocab    int `json:"n_vocab"`
	// HasEncoder / HasDecoder describe the architecture; a plain causal LM
	// has only a decoder.
	HasEncoder bool `json:"has_encoder"`
	HasDecoder bool `json:"has_decoder"`
	// BOSToken / EOSToken are the vocabulary's sentence delimiters, -1
	// when the model defines none. AddBOS is whether the tokenizer
	// prepends BOSToken by convention (what Tokenize's addSpecial obeys).
	BOSToken int32 `json:"bos_token"`
	EOSToken int32 `json:"eos_token"`
	AddBOS   bool  `json:"add_bos"`
	// ChatTemplate is the Jinja-ish template string the GGUF carries, or ""
	// when it has none. Chat needs a template from somewhere: either this or
	// SamplingParams-independent override passed to ApplyChatTemplate.
	ChatTemplate string `json:"chat_template"`
}

// Build reports how the embedded engine was compiled. Diagnostics only; see
// BuildInfo.
type Build struct {
	SIMD       bool `json:"simd"`
	Threads    bool `json:"threads"`
	Exceptions bool `json:"exceptions"`
	MaxDevices int  `json:"max_devices"`
}

// StopReason says why generation ended.
type StopReason string

const (
	// StopEOS: the model emitted an end-of-generation token.
	StopEOS StopReason = "eos"
	// StopLength: the token budget ran out (Params.NPredict, or the context
	// filled up).
	StopLength StopReason = "length"
	// StopString: one of Params.Stop matched the tail of the output, which
	// was trimmed off.
	StopString StopReason = "stop"
	// StopInterrupted: Context.Interrupt was called from another goroutine.
	StopInterrupted StopReason = "interrupted"
)

// Timings is the engine's wall-clock split of one generation.
type Timings struct {
	PromptMS float64 `json:"prompt_ms"`
	DecodeMS float64 `json:"decode_ms"`
}

// Result is what Generate returns.
type Result struct {
	// Text is the generated text, with a matched stop string removed.
	Text string `json:"text"`
	// Tokens are the tokens behind Text.
	Tokens []int32 `json:"tokens"`
	// NPrompt is how many tokens the prompt occupied; NDecoded how many were
	// generated.
	NPrompt  int `json:"n_prompt"`
	NDecoded int `json:"n_decoded"`
	// NCached is how many leading prompt tokens a Params.CachePrompt
	// generation reused from the KV cache instead of re-decoding. Zero
	// without CachePrompt.
	NCached int `json:"n_cached"`
	// NDrafted / NAccepted are GenerateWithDraft's speculation counters —
	// how many tokens the draft proposed and how many the target accepted.
	// Zero on a plain Generate.
	NDrafted  int `json:"n_drafted"`
	NAccepted int `json:"n_accepted"`
	// Reason says why generation stopped; Interrupted is a shorthand for
	// Reason == StopInterrupted.
	Reason      StopReason `json:"stop_reason"`
	Interrupted bool       `json:"interrupted"`
	// Timings splits the generation's wall time between the prompt pass
	// and the decode loop.
	Timings Timings `json:"timings"`
}

// Params configures one generation.
//
// The zero value is greedy decoding with no truncation samplers, no penalties
// and no token limit: the most predictable thing the model can do, and
// reproducible run to run. Every field is honoured exactly as written — a zero
// is a decision ("no top-k", "temperature 0 means greedy"), not "use some
// default" — so what a caller leaves out cannot be overridden by a default
// buried in the engine.
type Params struct {
	// NPredict caps the number of tokens generated. Zero or negative means
	// "as many as the context allows".
	NPredict int
	// Temperature <= 0 selects greedy decoding, which makes generation
	// deterministic and skips the truncation samplers below.
	Temperature float32
	// TopK, TopP, MinP and TypicalP truncate the candidate set before
	// sampling. Zero disables each of them, as does 1.0 for the two
	// probability-mass ones.
	TopK     int
	TopP     float32
	MinP     float32
	TypicalP float32
	// RepeatPenalty, PresencePenalty and FrequencyPenalty discourage
	// repetition over the last RepeatLastN tokens. Zero disables all three;
	// so does RepeatPenalty 1.0, which is llama.cpp's own spelling of "off".
	RepeatPenalty    float32
	RepeatLastN      int
	PresencePenalty  float32
	FrequencyPenalty float32
	// Seed makes sampling reproducible. Zero is a fixed seed, so a
	// temperature above zero still replays identically; set a varying seed to
	// get varying output.
	Seed uint32
	// Mirostat selects the mirostat sampling algorithm: 0 off, 1 v1, 2 v2.
	// When on it replaces the TopK/TopP/MinP/TypicalP truncation samplers,
	// as in llama.cpp. MirostatTau is the target entropy (0 means llama.cpp's
	// 5.0) and MirostatEta the learning rate (0 means 0.1).
	Mirostat    int
	MirostatTau float32
	MirostatEta float32
	// IgnoreEOS keeps generating past end-of-generation tokens by excluding
	// them from sampling, like llama.cpp's --ignore-eos. Generation then runs
	// to NPredict, a stop string, or the context limit.
	IgnoreEOS bool
	// LogitBias adds a bias to specific tokens' logits before sampling.
	// math.Inf(-1) (or any very negative value) forbids a token outright.
	LogitBias map[int32]float32
	// Grammar is a GBNF grammar constraining the output. Empty means none.
	Grammar string
	// Stop ends generation when the output first contains one of these
	// strings — even mid-token — and the text is cut at the match.
	Stop []string
	// CachePrompt treats the prompt as the WHOLE intended context: the
	// longest prefix already in the context's KV cache is kept, whatever the
	// cache holds beyond it is dropped, and only the rest is decoded —
	// llama.cpp server's cache_prompt. Requests that share a long constant
	// preamble (a system prompt, a routing configuration) then pay only for
	// the part that changed; Result.NCached reports the reuse. Off, the
	// prompt appends at the cache's current end, which is what an
	// Eval-prefill continuation expects.
	CachePrompt bool
}

// genRequest is the wire form of a generation. It exists because the bridge's
// parameter object is "absent means the engine's default", and Params is
// "written means meant": every field is emitted, so a zero the caller chose
// cannot be mistaken for a field they never set.
type genRequest struct {
	NPredict         int     `json:"n_predict"`
	Temperature      float32 `json:"temperature"`
	TopK             int     `json:"top_k"`
	TopP             float32 `json:"top_p"`
	MinP             float32 `json:"min_p"`
	TypicalP         float32 `json:"typical_p"`
	RepeatPenalty    float32 `json:"repeat_penalty"`
	RepeatLastN      int     `json:"repeat_last_n"`
	PresencePenalty  float32 `json:"presence_penalty"`
	FrequencyPenalty float32 `json:"frequency_penalty"`
	Seed             uint32  `json:"seed"`
	Mirostat         int     `json:"mirostat"`
	MirostatTau      float32 `json:"mirostat_tau"`
	MirostatEta      float32 `json:"mirostat_eta"`
	IgnoreEOS        int     `json:"ignore_eos"`
	CachePrompt      int     `json:"cache_prompt,omitempty"`
	// LogitBias is wired as [[token,bias],...] — a JSON object would
	// stringify the token ids.
	LogitBias [][2]float64 `json:"logit_bias,omitempty"`
	Grammar   string       `json:"grammar,omitempty"`
	Stop      []string     `json:"stop,omitempty"`
}

// wire converts Params into the bridge's parameter object, translating the two
// places where Go's zero value and llama.cpp's "off" value disagree.
func (p Params) wire() genRequest {
	req := genRequest{
		NPredict:         p.NPredict,
		Temperature:      p.Temperature,
		TopK:             p.TopK,
		TopP:             p.TopP,
		MinP:             p.MinP,
		TypicalP:         p.TypicalP,
		RepeatPenalty:    p.RepeatPenalty,
		RepeatLastN:      p.RepeatLastN,
		PresencePenalty:  p.PresencePenalty,
		FrequencyPenalty: p.FrequencyPenalty,
		Seed:             p.Seed,
		Mirostat:         p.Mirostat,
		MirostatTau:      p.MirostatTau,
		MirostatEta:      p.MirostatEta,
		Grammar:          p.Grammar,
		Stop:             p.Stop,
	}
	if p.IgnoreEOS {
		req.IgnoreEOS = 1
	}
	if p.CachePrompt {
		req.CachePrompt = 1
	}
	// The engine reads a non-negative n_predict as an exact budget, so zero
	// would generate nothing; Params spells "no limit" as the zero value.
	if req.NPredict <= 0 {
		req.NPredict = -1
	}
	// llama.cpp disables the repetition penalty with 1.0, not 0.0, and a
	// literal 0.0 would zero the logits of every recent token instead.
	if req.RepeatPenalty == 0 {
		req.RepeatPenalty = 1
	}
	// Mirostat's tau/eta zeros mean "llama.cpp's defaults", not literal
	// zero — a zero learning rate would freeze the algorithm.
	if req.MirostatTau == 0 {
		req.MirostatTau = 5.0
	}
	if req.MirostatEta == 0 {
		req.MirostatEta = 0.1
	}
	// Sorted for a deterministic wire form; bias application itself is
	// order-independent. JSON cannot carry infinities, so ±Inf clamps to
	// ±1e9 — far beyond any real logit, so the effect is identical.
	if len(p.LogitBias) > 0 {
		toks := make([]int32, 0, len(p.LogitBias))
		for t := range p.LogitBias {
			toks = append(toks, t)
		}
		slices.Sort(toks)
		req.LogitBias = make([][2]float64, len(toks))
		for i, t := range toks {
			b := float64(p.LogitBias[t])
			b = math.Min(math.Max(b, -1e9), 1e9)
			req.LogitBias[i] = [2]float64{float64(t), b}
		}
	}
	return req
}

// Message is one turn of a chat, as ApplyChatTemplate consumes it.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// envelope is the common head of every JSON result the bridge returns: a call
// that failed reports it here rather than throwing across the wasm boundary.
type envelope struct {
	Ok    bool   `json:"ok"`
	Error string `json:"error"`
}

// err turns a not-ok envelope into an error naming the operation. A result
// that is neither ok nor carries a message still has to fail: silently
// returning the zero value would look like success.
func (e *envelope) err(what string) error {
	switch {
	case e.Ok:
		return nil
	case e.Error != "":
		return fmt.Errorf("llama: %s: %s", what, e.Error)
	default:
		return fmt.Errorf("llama: %s: failed without a message", what)
	}
}

// enveloped is what decode unmarshals into: the result types below all embed
// envelope, so the promoted err method reports the bridge's own failure.
type enveloped interface {
	err(what string) error
}
