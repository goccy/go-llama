package llama_test

import (
	"os"
	"strconv"
	"testing"
	"time"

	llama "github.com/goccy/go-llama"
)

// TestModernBertSpeed times one CLS embedding (prompt processing only) for
// 32 / 128 / 512 synthetic tokens. Opt-in via MODERNBERT_GGUF.
func TestModernBertSpeed(t *testing.T) {
	name := os.Getenv("MODERNBERT_GGUF")
	if name == "" {
		t.Skip("set MODERNBERT_GGUF")
	}
	m, err := testInst.LoadModel(name)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	threads := uint32(8)
	if s := os.Getenv("MODERNBERT_THREADS"); s != "" {
		v, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		threads = uint32(v)
	}
	ctx, err := m.NewContext(llama.ContextParams{NCtx: 1024, NBatch: 1024, NUBatch: 1024, NThreads: threads, Embeddings: true})
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	for _, n := range []int{32, 128, 512} {
		toks := make([]int32, n)
		toks[0], toks[n-1] = 1, 2
		for i := 1; i < n-1; i++ {
			toks[i] = int32(1000 + (i*37)%50000)
		}
		for i := 0; i < 2; i++ { // warm-up
			if _, err := ctx.EmbedTokens(toks, false); err != nil {
				t.Fatal(err)
			}
		}
		const reps = 5
		best := time.Hour
		var total time.Duration
		for i := 0; i < reps; i++ {
			t0 := time.Now()
			if _, err := ctx.EmbedTokens(toks, false); err != nil {
				t.Fatal(err)
			}
			d := time.Since(t0)
			total += d
			if d < best {
				best = d
			}
		}
		avg := total / reps
		t.Logf("%s threads=%d T=%4d  avg %7.1f ms  best %7.1f ms  (%.0f tok/s avg)", name, threads, n, float64(avg.Microseconds())/1000, float64(best.Microseconds())/1000, float64(n)/avg.Seconds())
	}
}
