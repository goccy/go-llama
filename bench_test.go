package llama_test

// Throughput benchmarks, comparable with native llama-bench via
// scripts/bench-compare.sh. Both regimes matter and they are different:
// prompt processing (pp) is one batched pass over many tokens; token
// generation (tg) is one token at a time, which is what an interactive
// caller feels.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llama "github.com/goccy/go-llama"
)

// benchContext loads the benchmark model and one context outside the
// timed region.
func benchContext(b *testing.B, nCtx uint32) (*llama.Model, *llama.Context) {
	b.Helper()
	m, err := testInst.LoadModel(benchModelName(b))
	if err != nil {
		b.Fatalf("LoadModel: %v", err)
	}
	b.Cleanup(func() { m.Close() })
	ctx, err := m.NewContext(llama.ContextParams{NCtx: nCtx})
	if err != nil {
		b.Fatalf("NewContext: %v", err)
	}
	b.Cleanup(func() { ctx.Close() })
	return m, ctx
}

func benchModelName(b *testing.B) string {
	b.Helper()
	p := modelPath()
	if _, err := os.Stat(p); err != nil {
		b.Skipf("benchmark model missing (%v); run `make testdata` or set GO_LLAMA_TEST_MODEL", err)
	}
	return filepath.Base(p)
}

// BenchmarkDecode reports token-generation throughput (native
// llama-bench's tg). b.N is one generation's token budget: restarting
// would re-run prompt evaluation and time the wrong thing.
func BenchmarkDecode(b *testing.B) {
	_, ctx := benchContext(b, 2048)
	b.ResetTimer()
	res, err := ctx.Generate("Once upon a time", llama.Params{NPredict: b.N, Temperature: 0})
	b.StopTimer()
	if err != nil {
		b.Fatalf("Generate: %v", err)
	}
	if res.NDecoded != b.N {
		b.Fatalf("wanted %d tokens, decoded %d (%s)", b.N, res.NDecoded, res.Reason)
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkPromptEval reports prompt-processing throughput (native
// llama-bench's pp): a long prompt, one predicted token.
func BenchmarkPromptEval(b *testing.B) {
	m, ctx := benchContext(b, 2048)
	prompt := strings.Repeat("Once upon a time there was a little girl who loved to play outside. ", 60)
	// ggml's mul_mat takes its two-row vec_dot path only for an even
	// batch (ne11 % 2 == 0), and llama-bench's pp sizes are even, so
	// pad the prompt until its token count is even (and the 512-token
	// batches it splits into are all even).
	for i := 0; i < 8; i++ {
		toks, err := m.Tokenize(prompt, true, false)
		if err != nil {
			b.Fatalf("Tokenize: %v", err)
		}
		if len(toks)%2 == 0 {
			break
		}
		prompt += " and"
	}
	b.ResetTimer()
	nPrompt := 0
	for b.Loop() {
		b.StopTimer()
		if err := ctx.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		b.StartTimer()
		res, err := ctx.Generate(prompt, llama.Params{NPredict: 1, Temperature: 0})
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		nPrompt = res.NPrompt
	}
	b.ReportMetric(float64(nPrompt)*float64(b.N)/b.Elapsed().Seconds(), "prompt_tok/s")
}

// BenchmarkDecodeDeep reports token-generation throughput with the KV
// cache already holding GO_LLAMA_BENCH_DEPTH tokens (default 2048), the
// regime native llama-bench's `tg64 @ d2048` measures: the prompt is
// decoded once outside the timed region and every iteration reuses it
// through CachePrompt, so only the decode loop is timed (the engine's
// own decode timing, as llama-bench reports it).
func BenchmarkDecodeDeep(b *testing.B) {
	depth := 2048
	if s := os.Getenv("GO_LLAMA_BENCH_DEPTH"); s != "" {
		if _, err := fmt.Sscan(s, &depth); err != nil {
			b.Fatalf("GO_LLAMA_BENCH_DEPTH: %v", err)
		}
	}
	m, ctx := benchContext(b, uint32(depth+b.N+64))
	unit := "Once upon a time there was a little girl who loved to play outside. "
	toks, err := m.Tokenize(unit, false, false)
	if err != nil {
		b.Fatalf("Tokenize: %v", err)
	}
	prompt := strings.Repeat(unit, (depth+len(toks)-1)/len(toks))
	all, err := m.Tokenize(prompt, false, false)
	if err != nil {
		b.Fatalf("Tokenize: %v", err)
	}
	if len(all) > depth {
		all = all[:depth]
	}
	if prompt, err = m.Detokenize(all, false); err != nil {
		b.Fatalf("Detokenize: %v", err)
	}
	if _, err := ctx.Generate(prompt, llama.Params{NPredict: 1, Temperature: 0}); err != nil {
		b.Fatalf("Generate (fill): %v", err)
	}
	b.ResetTimer()
	res, err := ctx.Generate(prompt, llama.Params{NPredict: b.N, Temperature: 0, CachePrompt: true, IgnoreEOS: true})
	b.StopTimer()
	if err != nil {
		b.Fatalf("Generate: %v", err)
	}
	if res.NDecoded != b.N {
		b.Fatalf("wanted %d tokens, decoded %d (%s)", b.N, res.NDecoded, res.Reason)
	}
	if res.NCached < res.NPrompt-8 {
		b.Fatalf("CachePrompt reused %d of %d prompt tokens", res.NCached, res.NPrompt)
	}
	b.ReportMetric(float64(res.NPrompt), "depth")
	b.ReportMetric(float64(b.N)/(res.Timings.DecodeMS/1000), "tok/s")
}
