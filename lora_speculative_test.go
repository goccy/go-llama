package llama_test

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"testing"

	llama "github.com/goccy/go-llama"
)

// modelArch reads general.architecture out of the test model's GGUF
// header by walking the key-value section structurally, so the adapter
// fixture matches whatever base model the suite runs against instead of
// assuming one architecture.
func modelArch(t *testing.T) string {
	t.Helper()
	f, err := os.Open(modelPath())
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	le := binary.LittleEndian
	r32 := func() uint32 {
		var v uint32
		if err := binary.Read(f, le, &v); err != nil {
			t.Fatalf("gguf read: %v", err)
		}
		return v
	}
	r64 := func() uint64 {
		var v uint64
		if err := binary.Read(f, le, &v); err != nil {
			t.Fatalf("gguf read: %v", err)
		}
		return v
	}
	rstr := func() string {
		b := make([]byte, r64())
		if _, err := io.ReadFull(f, b); err != nil {
			t.Fatalf("gguf read: %v", err)
		}
		return string(b)
	}
	skip := func(n uint64) {
		if _, err := f.Seek(int64(n), io.SeekCurrent); err != nil {
			t.Fatalf("gguf seek: %v", err)
		}
	}
	// Fixed sizes of the scalar gguf value types; string and array are
	// length-prefixed and handled separately.
	scalarSize := map[uint32]uint64{
		0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
	}
	const typStr, typArr = 8, 9
	var skipVal func(typ uint32)
	skipVal = func(typ uint32) {
		switch typ {
		case typStr:
			skip(r64())
		case typArr:
			elem := r32()
			n := r64()
			if sz, ok := scalarSize[elem]; ok {
				skip(n * sz)
				return
			}
			for i := uint64(0); i < n; i++ {
				skipVal(elem)
			}
		default:
			sz, ok := scalarSize[typ]
			if !ok {
				t.Fatalf("gguf: unknown value type %d", typ)
			}
			skip(sz)
		}
	}
	if magic := r32(); magic != 0x46554747 {
		t.Fatalf("not a gguf file: magic %#x", magic)
	}
	r32() // version
	r64() // tensor count
	nKV := r64()
	for i := uint64(0); i < nKV; i++ {
		key := rstr()
		typ := r32()
		if key == "general.architecture" && typ == typStr {
			return rstr()
		}
		skipVal(typ)
	}
	t.Fatal("gguf: general.architecture not found")
	return ""
}

// ggufAdapter builds a minimal GGUF v3 LoRA adapter in memory: one
// rank-1 A/B pair of ZEROS for tensorName, so applying it at any scale
// is a no-op — which is exactly what the test wants to assert about the
// plumbing without shipping a binary fixture.
func ggufAdapter(arch, tensorName string, nEmbd int) []byte {
	var buf bytes.Buffer
	le := binary.LittleEndian
	w32 := func(v uint32) { _ = binary.Write(&buf, le, v) }
	w64 := func(v uint64) { _ = binary.Write(&buf, le, v) }
	wf32 := func(v float32) { _ = binary.Write(&buf, le, v) }
	wstr := func(s string) { w64(uint64(len(s))); buf.WriteString(s) }

	const (
		typF32 = 6 // gguf metadata value type float32
		typStr = 8 // gguf metadata value type string
		tF32   = 0 // ggml tensor dtype F32
		align  = 32
	)

	w32(0x46554747) // "GGUF"
	w32(3)          // version
	w64(2)          // tensors: lora_a, lora_b
	w64(4)          // kv pairs

	wstr("general.type")
	w32(typStr)
	wstr("adapter")
	// The loader rejects an adapter whose architecture differs from the
	// base model's, so the caller passes the model's own.
	wstr("general.architecture")
	w32(typStr)
	wstr(arch)
	wstr("adapter.type")
	w32(typStr)
	wstr("lora")
	wstr("adapter.lora.alpha")
	w32(typF32)
	wf32(1.0)

	// llama.cpp expects, for a base tensor of ne [in,out]:
	// lora_a ne [in,r] and lora_b ne [r,out].
	rowBytes := uint64(4 * nEmbd)
	bOff := (rowBytes + align - 1) / align * align
	wstr(tensorName + ".lora_a")
	w32(2)
	w64(uint64(nEmbd))
	w64(1)
	w32(tF32)
	w64(0)
	wstr(tensorName + ".lora_b")
	w32(2)
	w64(1)
	w64(uint64(nEmbd))
	w32(tF32)
	w64(bOff)

	for buf.Len()%align != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, bOff+rowBytes)) // both tensors, all zeros
	return buf.Bytes()
}

// writeLoRAFixture materializes the adapter inside the engine's preopened
// directory (the model's directory — see TestMain) and removes it after.
func writeLoRAFixture(t *testing.T, nEmbd int) string {
	t.Helper()
	name := "zero-delta-adapter.gguf"
	host := filepath.Join(filepath.Dir(modelPath()), name)
	if err := os.WriteFile(host, ggufAdapter(modelArch(t), "blk.0.attn_q.weight", nEmbd), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(host) })
	return name
}

// TestLoRALoadSetClear drives the adapter plumbing end to end with a
// zero-delta adapter: loading, setting at a scale, and clearing must all
// succeed, and a zero adapter must leave greedy generation unchanged.
func TestLoRALoadSetClear(t *testing.T) {
	m := load(t)
	info, err := m.Info()
	if err != nil {
		t.Fatal(err)
	}
	fixture := writeLoRAFixture(t, info.NEmbd)

	l, err := m.LoadLoRA(fixture)
	if err != nil {
		t.Fatalf("LoadLoRA: %v", err)
	}
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	gen := func() string {
		t.Helper()
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
		res, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 12, Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		return res.Text
	}

	base := gen()
	if err := ctx.SetLoRA([]llama.LoRAWeight{{Adapter: l, Scale: 1.0}}); err != nil {
		t.Fatalf("SetLoRA: %v", err)
	}
	if got := gen(); got != base {
		t.Errorf("zero-delta adapter changed generation:\n  base: %q\n  lora: %q", base, got)
	}
	if err := ctx.SetLoRA(nil); err != nil {
		t.Fatalf("SetLoRA(nil): %v", err)
	}
	if got := gen(); got != base {
		t.Errorf("clearing adapters changed generation:\n  base: %q\n  after: %q", base, got)
	}
	if err := l.Close(); err != nil {
		t.Errorf("LoRA.Close: %v", err)
	}

	if _, err := m.LoadLoRA("no-such-adapter.gguf"); err == nil {
		t.Error("loading a missing adapter did not fail")
	}
}

// TestSpeculativeMatchesPlain pins speculative decoding's contract: the
// output IS the target's — greedy and seeded-sampling runs must equal the
// plain Generate byte for byte, and with the same model drafting, every
// greedy draft must be accepted.
func TestSpeculativeMatchesPlain(t *testing.T) {
	m := load(t)
	target, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	draft, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Close()

	const prompt = "Once upon a time"
	run := func(p llama.Params) (plain, spec llama.Result) {
		t.Helper()
		if err := target.Reset(); err != nil {
			t.Fatal(err)
		}
		plain, err := target.Generate(prompt, p)
		if err != nil {
			t.Fatal(err)
		}
		spec, err = target.GenerateWithDraft(draft, prompt, p, 4)
		if err != nil {
			t.Fatal(err)
		}
		return plain, spec
	}

	plain, spec := run(llama.Params{NPredict: 24, Temperature: 0})
	if plain.Text != spec.Text {
		t.Errorf("speculative greedy diverges from plain:\n  plain: %q\n  spec:  %q", plain.Text, spec.Text)
	}
	if spec.NDrafted == 0 {
		t.Error("draft proposed nothing")
	}
	// The draft IS the target here and both decode greedily, so every
	// proposal must be accepted.
	if spec.NAccepted != spec.NDrafted {
		t.Errorf("same-model greedy draft not fully accepted: %d/%d", spec.NAccepted, spec.NDrafted)
	}

	// Each emitted token is sampled by the target's chain in the same
	// order as a plain run, so a fixed seed replays identically too.
	plain, spec = run(llama.Params{NPredict: 24, Temperature: 0.9, TopP: 0.9, Seed: 7})
	if plain.Text != spec.Text {
		t.Errorf("speculative sampling diverges from plain at the same seed:\n  plain: %q\n  spec:  %q", plain.Text, spec.Text)
	}
}

// TestGenerateTimings: the engine reports where the time went.
func TestGenerateTimings(t *testing.T) {
	m := load(t)
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	res, err := ctx.Generate("Once upon a time", llama.Params{NPredict: 4, Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Timings.PromptMS <= 0 || res.Timings.DecodeMS <= 0 {
		t.Errorf("timings not reported: %+v", res.Timings)
	}
}

// TestModelInfoVocab: the vocab landmarks agree with what special-token
// parsing resolves through the tokenizer — the reported EOS id renders
// to a piece that tokenizes back to exactly that id. (Asserting against
// a literal like "</s>" would bake in one model family's EOS text.)
func TestModelInfoVocab(t *testing.T) {
	m := load(t)
	info, err := m.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.EOSToken < 0 {
		t.Fatalf("EOSToken not reported: %d", info.EOSToken)
	}
	piece, err := m.TokenToPiece(info.EOSToken, true)
	if err != nil {
		t.Fatalf("TokenToPiece(EOS): %v", err)
	}
	if piece == "" {
		t.Fatalf("EOSToken %d renders to an empty piece", info.EOSToken)
	}
	toks, err := m.Tokenize(piece, false, true)
	if err != nil {
		t.Fatalf("Tokenize(%q): %v", piece, err)
	}
	if len(toks) != 1 || toks[0] != info.EOSToken {
		t.Errorf("EOS piece %q does not round-trip: got tokens %v, want [%d]", piece, toks, info.EOSToken)
	}
}
