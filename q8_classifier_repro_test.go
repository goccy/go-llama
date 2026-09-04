package llama_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"testing"

	llama "github.com/goccy/go-llama"
)

// TestQ8ClassifierShape drives the aigateway autorouter shape against the
// real q8_0 classifier model (Qwen2.5-1.5B-Instruct q8_0): a chat-template
// prefix evaluated into the KV cache, a user suffix evaluated on top, and
// ScoreChoices over the 12 stage-1 labels.
//
// It is an engine-correctness A/B, not a model-quality test: run once with
// GOAMD64=v1 (pure-Go lowering, golden) and SCORES_OUT=<file>, then with
// GOAMD64=v2 (SIMD splices + assembly overrides) and SCORES_GOLDEN=<file>.
// The two runs must agree on every per-token label NLL within tolerance.
//
// Gated by Q8_MODEL (path to qwen2.5-1.5b-instruct-q8_0.gguf).
func TestQ8ClassifierShape(t *testing.T) {
	path := os.Getenv("Q8_MODEL")
	if path == "" {
		t.Skip("set Q8_MODEL")
	}
	t.Logf("goarch: %s", runtime.GOARCH)

	inst, err := llama.New()
	if err != nil {
		t.Fatal(err)
	}
	defer inst.Close()
	m, err := inst.LoadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	labels := []string{"coding", "debugging", "math", "qa", "chat", "summarize", "translate", "write", "extract", "analyze", "agent", "other"}
	prefix := "<|im_start|>system\nユーザーの依頼を coding, debugging, math, qa, chat, summarize, translate, write, extract, analyze, agent, other のいずれかに分類し、ラベルだけを出力してください。<|im_end|>\n" +
		"<|im_start|>user\n関数を書いて CSV を集計したい<|im_end|>\n<|im_start|>assistant\ncoding<|im_end|>\n" +
		"<|im_start|>user\nこの文章を英語にして<|im_end|>\n<|im_start|>assistant\ntranslate<|im_end|>\n" +
		"<|im_start|>user\n"
	const post = "<|im_end|>\n<|im_start|>assistant\n"

	cases := []struct {
		name   string
		suffix string
	}{
		{"ja-coding", "React のコンポーネントで一覧画面を実装してほしい。ページネーションもつけて。" + post},
		{"ja-debugging", "このスタックトレースの原因を調査して直してほしい。nil pointer dereference が出ている。" + post},
		{"ja-translate", "次の段落を自然な英語に翻訳してください。" + post},
		// Long suffix: crosses NBatch=128 in the suffix eval like real
		// gateway prompts do.
		{"ja-long", "経費精算の承認フローで差戻しが発生したときに申請者へ通知メールが飛ばない不具合を調査しています。再現手順と関連するコードの場所も知りたいです。ログには特にエラーが出ていません。どこから手を付けるべきか、調査の進め方も含めて提案してください。通知基盤は SES で、テンプレートは DB 管理です。ワークフロー エンジンは内製で、状態遷移のイベントから非同期でメール送信ジョブが積まれる構成です。" + post},
	}

	got := map[string]map[string]float64{}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx, err := m.NewContext(llama.ContextParams{NCtx: 2048, NBatch: 128, NThreads: 4, NSeqMax: 16})
			if err != nil {
				t.Fatal(err)
			}
			defer ctx.Close()
			if _, err := ctx.Eval(prefix, true, true); err != nil {
				t.Fatal(err)
			}
			if _, err := ctx.Eval(c.suffix, false, true); err != nil {
				t.Fatal(err)
			}
			scores, err := ctx.ScoreChoices(labels)
			if err != nil {
				t.Fatal(err)
			}
			per := map[string]float64{}
			type ls struct {
				label string
				nll   float64
			}
			all := make([]ls, len(labels))
			for i, s := range scores {
				nll := s.NLL / float64(s.NTokens)
				per[labels[i]] = nll
				all[i] = ls{labels[i], nll}
			}
			got[c.name] = per
			sort.Slice(all, func(i, j int) bool { return all[i].nll < all[j].nll })
			line := ""
			for _, e := range all {
				line += fmt.Sprintf("%s=%.3f ", e.label, e.nll)
			}
			t.Logf("scores (per-token NLL asc): %s", line)
			for _, e := range all {
				if math.IsNaN(e.nll) || math.IsInf(e.nll, 0) {
					t.Errorf("non-finite NLL for %s", e.label)
				}
			}
		})
	}

	if out := os.Getenv("SCORES_OUT"); out != "" {
		b, _ := json.MarshalIndent(got, "", " ")
		if err := os.WriteFile(out, b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote golden scores to %s", out)
	}
	if golden := os.Getenv("SCORES_GOLDEN"); golden != "" {
		b, err := os.ReadFile(golden)
		if err != nil {
			t.Fatal(err)
		}
		var want map[string]map[string]float64
		if err := json.Unmarshal(b, &want); err != nil {
			t.Fatal(err)
		}
		// Fast-math (FMA, accumulation order) shifts NLLs slightly between
		// lowerings; a broken kernel shifts them by whole nats. 0.2 per-token
		// NLL separates the two regimes comfortably.
		const tol = 0.2
		for cname, wantPer := range want {
			for label, w := range wantPer {
				g, ok := got[cname][label]
				if !ok {
					t.Errorf("%s/%s missing in this run", cname, label)
					continue
				}
				if math.Abs(g-w) > tol {
					t.Errorf("%s/%s: NLL %v (this run) vs %v (golden), |Δ|=%.3f > %.1f", cname, label, g, w, math.Abs(g-w), tol)
				}
			}
		}
	}
}
