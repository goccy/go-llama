package llama_test

// Verification against native llama.cpp.
//
// These tests compare go-llama's engine — llama.cpp compiled to wasm and
// translated to Go — against a NATIVE build of the same llama.cpp
// commit. They need its tools on disk, so they are opt-in:
//
//	GO_LLAMA_NATIVE_REF=/path/to/llama.cpp/build/bin go test -run Native
//
// The directory must hold llama-tokenize, llama-completion and
// llama-bench. Without the variable the tests skip, keeping `make test`
// self-contained.
//
// What can and cannot be identical:
//
//   - Tokenization is integer code: token IDs must match EXACTLY, for
//     every script and edge case.
//   - Metadata is read straight out of the GGUF: must match.
//   - Greedy generation is float code. For an F32 model the wasm build
//     and the native build run the same arithmetic and must agree token
//     for token. For QUANTIZED models they run DIFFERENT kernels
//     (ggml's hand-written native SIMD vs the wasm build's generic
//     code), the logits differ in the last bits, and greedy decoding
//     legitimately diverges wherever two candidates are close — exactly
//     as it does between llama.cpp's own backends. Those tests assert
//     robustness and report the divergence point rather than demanding
//     the impossible.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	llama "github.com/goccy/go-llama"
)

// nativeRef returns the native tool directory or skips the test.
func nativeRef(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("GO_LLAMA_NATIVE_REF")
	if dir == "" {
		t.Skip("set GO_LLAMA_NATIVE_REF to a native llama.cpp build/bin directory to run native-reference verification")
	}
	return dir
}

// nativeTool runs one native tool and returns its stdout.
func nativeTool(t *testing.T, dir, tool string, args ...string) string {
	t.Helper()
	cmd := exec.Command(filepath.Join(dir, tool), args...)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s %v: %v", tool, args, err)
	}
	return string(out)
}

// tokenizerCorpus is the multilingual input set for exact token-ID
// comparison. The ASCII/whitespace/emoji cases mirror the hard cases in
// llama.cpp's own tokenizer tests; the rest covers the scripts this
// package is expected to serve.
var tokenizerCorpus = []string{
	"",
	" ",
	"  ",
	"\t",
	"\n\n\n",
	"Hello world",
	" Hello World!",
	"Hello, y'all! How are you 😁 ?",
	"ied 4 ½ months",
	"Äpfel Übergrößenträger",
	"нещо на Български",
	"こんにちは、世界。今日はいい天気ですね。",
	"日本の首都は東京です。",
	"漢字とひらがなとカタカナが混ざった文章です。",
	"안녕하세요 세계",
	"你好世界，这是一个测试。",
	"مرحبا بالعالم",
	"🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅",
	"3.14159 2,718 -42 1e10",
	"    Hello\n    Hello",
	"<|im_start|>user",
}

// TestNativeTokenizerIDs asserts exact token-ID equality with
// llama-tokenize across the corpus, for every test model present.
func TestNativeTokenizerIDs(t *testing.T) {
	dir := nativeRef(t)
	for _, model := range verifyModels(t) {
		m, err := testInst.LoadModel(filepath.Base(model))
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range tokenizerCorpus {
			if text == "" {
				continue // llama-tokenize rejects an empty prompt
			}
			out := nativeTool(t, dir, "llama-tokenize", "-m", model, "-p", text, "--ids")
			want := parseIDList(t, out)
			// parse_special=true matches llama-tokenize's default, so
			// special-token text ("<|im_start|>") compares as the
			// special token on both sides.
			got, err := m.Tokenize(text, true, true)
			if err != nil {
				t.Fatalf("Tokenize(%q): %v", text, err)
			}
			if !int32SlicesEqual(got, want) {
				t.Errorf("%s: token IDs diverge for %q:\n  go:     %v\n  native: %v",
					filepath.Base(model), text, got, want)
			}
		}
		m.Close()
	}
}

// TestNativeGreedyF32 asserts token-for-token greedy equality on the
// F32 model, where wasm and native arithmetic must agree exactly.
func TestNativeGreedyF32(t *testing.T) {
	dir := nativeRef(t)
	model := verifyModelPath(t, "stories260K.gguf")
	m, err := testInst.LoadModel(filepath.Base(model))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()

	for _, prompt := range []string{"Once upon a time", "The little dog"} {
		res, err := ctx.Generate(prompt, llama.Params{NPredict: 32, Temperature: 0})
		if err != nil {
			t.Fatal(err)
		}
		native := nativeGreedy(t, dir, model, prompt, 32)
		if want := prompt + res.Text; !strings.HasPrefix(strings.TrimPrefix(native, " "), want) &&
			strings.TrimPrefix(native, " ") != want {
			t.Errorf("greedy diverges from native on an F32 model:\n  go:     %q\n  native: %q",
				want, native)
		}
		if err := ctx.Reset(); err != nil {
			t.Fatal(err)
		}
	}
}

// TestNativeGreedyQuantized documents the divergence point on quantized
// models. Different kernels produce slightly different logits, so exact
// equality is not the contract; generating plausibly and stopping
// cleanly is. The common-prefix length is logged for tracking.
func TestNativeGreedyQuantized(t *testing.T) {
	dir := nativeRef(t)
	for _, name := range []string{"qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"} {
		model := verifyModelPath(t, name)
		if model == "" {
			continue
		}
		m, err := testInst.LoadModel(filepath.Base(model))
		if err != nil {
			t.Fatal(err)
		}
		ctx, err := m.NewContext(llama.ContextParams{NCtx: 256})
		if err != nil {
			t.Fatal(err)
		}
		for _, prompt := range []string{"日本の首都は", "Once upon a time"} {
			res, err := ctx.Generate(prompt, llama.Params{NPredict: 32, Temperature: 0})
			if err != nil {
				t.Fatal(err)
			}
			if res.NDecoded == 0 || !utf8.ValidString(res.Text) {
				t.Errorf("%s %q: degenerate generation %q", name, prompt, res.Text)
			}
			native := nativeGreedy(t, dir, model, prompt, 32)
			goFull := prompt + res.Text
			t.Logf("%s %q: common prefix %d runes\n  go:     %q\n  native: %q",
				name, prompt, commonPrefixRunes(goFull, strings.TrimPrefix(native, " ")), goFull, native)
			if err := ctx.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		ctx.Close()
		m.Close()
	}
}

// TestNativeModelMetadata compares Info() against the fields
// llama-bench reports from the same GGUF.
func TestNativeModelMetadata(t *testing.T) {
	dir := nativeRef(t)
	for _, model := range verifyModels(t) {
		m, err := testInst.LoadModel(filepath.Base(model))
		if err != nil {
			t.Fatal(err)
		}
		info, err := m.Info()
		if err != nil {
			t.Fatal(err)
		}
		m.Close()
		out := nativeTool(t, dir, "llama-bench", "-m", model, "-p", "0", "-n", "1", "-r", "1", "-t", "1")
		row := ""
		for _, ln := range strings.Split(out, "\n") {
			if strings.HasPrefix(ln, "|") && !strings.Contains(ln, "model") && !strings.Contains(ln, "---") {
				row = ln
				break
			}
		}
		if row == "" {
			t.Fatalf("no result row in llama-bench output:\n%s", out)
		}
		cols := strings.Split(row, "|")
		desc := strings.TrimSpace(cols[1])
		params := strings.TrimSpace(cols[3])
		if info.Desc != desc {
			t.Errorf("%s: desc %q, native %q", filepath.Base(model), info.Desc, desc)
		}
		if want := formatParams(info.NParams); want != params {
			t.Errorf("%s: params %s (from %d), native %s", filepath.Base(model), want, info.NParams, params)
		}
	}
}

// nativeGreedy runs llama-completion deterministically and returns the
// combined prompt+completion text.
func nativeGreedy(t *testing.T, dir, model, prompt string, n int) string {
	t.Helper()
	out := nativeTool(t, dir, "llama-completion",
		"-m", model, "-p", prompt, "-n", fmt.Sprint(n),
		"--temp", "0", "-no-cnv", "--simple-io", "-t", "1")
	return strings.TrimRight(out, "\n")
}

// verifyModels lists the test models present in testdata.
func verifyModels(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, name := range []string{"stories260K.gguf", "qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"} {
		if p := verifyModelPath(t, name); p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		t.Skip("no test models in testdata; run `make testdata`")
	}
	return out
}

// verifyModelPath returns the absolute path of a testdata model, or ""
// when absent. stories260K is required; the rest are optional extras.
func verifyModelPath(t *testing.T, name string) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); err != nil {
		if name == "stories260K.gguf" {
			t.Skipf("test model missing: %v", err)
		}
		return ""
	}
	return p
}

// parseIDList parses llama-tokenize --ids output: `[1, 403, 407]`.
// Foreign tool output, parsed totally: anything unexpected fails the
// test rather than passing silently.
func parseIDList(t *testing.T, out string) []int32 {
	t.Helper()
	s := strings.TrimSpace(out)
	i, j := strings.LastIndex(s, "["), strings.LastIndex(s, "]")
	if i < 0 || j <= i {
		t.Fatalf("no ID list in llama-tokenize output: %q", out)
	}
	body := strings.TrimSpace(s[i+1 : j])
	if body == "" {
		return nil
	}
	var ids []int32
	for _, f := range strings.Split(body, ",") {
		var v int32
		if _, err := fmt.Sscanf(strings.TrimSpace(f), "%d", &v); err != nil {
			t.Fatalf("bad token ID %q in %q: %v", f, out, err)
		}
		ids = append(ids, v)
	}
	return ids
}

func int32SlicesEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func commonPrefixRunes(a, b string) int {
	ar, br := []rune(a), []rune(b)
	n := 0
	for n < len(ar) && n < len(br) && ar[n] == br[n] {
		n++
	}
	return n
}

// formatParams renders a parameter count the way llama-bench does:
// millions with two decimals below a billion ("630.17 M", "0.29 M"),
// billions above.
func formatParams(n uint64) string {
	if n >= 1e9 {
		return fmt.Sprintf("%.2f B", float64(n)/1e9)
	}
	return fmt.Sprintf("%.2f M", float64(n)/1e6)
}
