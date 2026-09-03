package perfgate

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

const llamaBenchJSON = `[
  {"n_prompt": 512, "n_gen": 0, "avg_ts": 280.5, "samples_ts": [278.1, 283.9, 279.4]},
  {"n_prompt": 0, "n_gen": 64, "avg_ts": 68.2, "samples_ts": [67.0, 69.3, 68.3]}
]`

func TestParseLlamaBenchBestOfSamples(t *testing.T) {
	got, err := ParseLlamaBench([]byte(llamaBenchJSON))
	if err != nil {
		t.Fatal(err)
	}
	if got[MetricPP] != 283.9 || got[MetricTG] != 69.3 {
		t.Fatalf("got %v, want pp=283.9 tg=69.3", got)
	}
}

func TestParseLlamaBenchFallsBackToAverage(t *testing.T) {
	got, err := ParseLlamaBench([]byte(`[{"n_prompt":512,"n_gen":0,"avg_ts":100},{"n_prompt":0,"n_gen":64,"avg_ts":50}]`))
	if err != nil {
		t.Fatal(err)
	}
	if got[MetricPP] != 100 || got[MetricTG] != 50 {
		t.Fatalf("got %v", got)
	}
}

func TestParseLlamaBenchRejectsMissingEntry(t *testing.T) {
	if _, err := ParseLlamaBench([]byte(`[{"n_prompt":512,"n_gen":0,"avg_ts":100}]`)); err == nil {
		t.Fatal("missing tg entry accepted")
	}
}

const goBenchOut = `goos: linux
goarch: arm64
BenchmarkDecode-4   	      64	   9611632 ns/op	       104.0 tok/s
BenchmarkDecode-4   	      64	   9371895 ns/op	       106.7 tok/s
BenchmarkDecode-4   	      64	   9384765 ns/op	       106.6 tok/s
BenchmarkPromptEval-4    	       3	6579516831 ns/op	       136.9 prompt_tok/s
PASS
ok  	github.com/goccy/go-llama	41.2s
`

func TestParseGoBenchBestOfCount(t *testing.T) {
	tg, err := ParseGoBench(goBenchOut, "tok/s")
	if err != nil {
		t.Fatal(err)
	}
	if tg["BenchmarkDecode"] != 106.7 {
		t.Fatalf("decode best = %v, want 106.7", tg["BenchmarkDecode"])
	}
	if _, has := tg["BenchmarkPromptEval"]; has {
		t.Fatal("prompt_tok/s line matched the tok/s unit")
	}
	pp, err := ParseGoBench(goBenchOut, "prompt_tok/s")
	if err != nil {
		t.Fatal(err)
	}
	if pp["BenchmarkPromptEval"] != 136.9 {
		t.Fatalf("prompt best = %v", pp["BenchmarkPromptEval"])
	}
}

func TestParseGoBenchRejectsNoLines(t *testing.T) {
	if _, err := ParseGoBench("PASS\nok x 1s\n", "tok/s"); err == nil {
		t.Fatal("empty output accepted")
	}
}

func TestCompareGatesOnRatio(t *testing.T) {
	base := &ArchResult{Metrics: map[string]Measurement{
		MetricTG: {GoTokS: 50, NativeTokS: 100, Ratio: 0.5},
		MetricPP: {GoTokS: 80, NativeTokS: 320, Ratio: 0.25},
	}}
	cur := ArchResult{Metrics: map[string]Measurement{
		// tg: ratio 0.46 = -8% vs 0.5 → within a 10% gate.
		MetricTG: {GoTokS: 46, NativeTokS: 100, Ratio: 0.46},
		// pp: ratio 0.20 = -20% vs 0.25 → regression, even though the
		// raw tok/s went UP (a faster runner).
		MetricPP: {GoTokS: 100, NativeTokS: 500, Ratio: 0.20},
	}}
	vs := Compare(cur, base, 0.10)
	if len(vs) != 2 || vs[0].Metric != MetricPP || vs[1].Metric != MetricTG {
		t.Fatalf("verdict order = %+v", vs)
	}
	if !vs[0].Regressed || vs[1].Regressed {
		t.Fatalf("pp must regress, tg must not: %+v", vs)
	}
	if math.Abs(vs[0].Change-(-0.20)) > 1e-9 || math.Abs(vs[1].Change-(-0.08)) > 1e-9 {
		t.Fatalf("changes = %v %v", vs[0].Change, vs[1].Change)
	}
	// Exactly at the threshold is not a regression (strictly below fails).
	edge := ArchResult{Metrics: map[string]Measurement{MetricTG: {Ratio: 0.45}}}
	if Compare(edge, base, 0.10)[0].Regressed {
		t.Fatal("ratio exactly at baseline*(1-threshold) must pass")
	}
}

func TestCompareWithoutBaselineRecordsOnly(t *testing.T) {
	cur := ArchResult{Metrics: map[string]Measurement{MetricTG: {Ratio: 0.1}}}
	for _, base := range []*ArchResult{nil, {Metrics: map[string]Measurement{}}} {
		vs := Compare(cur, base, 0.10)
		if vs[0].Regressed || vs[0].BaselineRatio != 0 {
			t.Fatalf("no-baseline verdict = %+v", vs[0])
		}
	}
}

func TestBaselineRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "baseline.json")
	empty, err := LoadBaseline(p)
	if err != nil || len(empty) != 0 {
		t.Fatalf("missing file: %v %v", empty, err)
	}
	want := Baseline{"arm64": {
		Metrics:  map[string]Measurement{MetricTG: {GoTokS: 1, NativeTokS: 2, Ratio: 0.5}},
		Recorded: Provenance{Engine: "v0.3.2", LlamaCPP: "11924d4c"},
	}}
	if err := SaveBaseline(p, want); err != nil {
		t.Fatal(err)
	}
	got, err := LoadBaseline(p)
	if err != nil {
		t.Fatal(err)
	}
	if got["arm64"].Metrics[MetricTG] != want["arm64"].Metrics[MetricTG] || got["arm64"].Recorded.LlamaCPP != "11924d4c" {
		t.Fatalf("round trip = %+v", got)
	}
}

func TestSummaryMarksRegression(t *testing.T) {
	vs := []Verdict{
		{Metric: MetricPP, Current: Measurement{GoTokS: 1, NativeTokS: 4, Ratio: 0.25}, BaselineRatio: 0.5, Change: -0.5, Regressed: true},
		{Metric: MetricTG, Current: Measurement{GoTokS: 1, NativeTokS: 2, Ratio: 0.5}},
	}
	s := Summary("amd64", vs, 0.10)
	for _, want := range []string{"**REGRESSION**", "recorded (no baseline)", "-50.0%", "(amd64)"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary missing %q:\n%s", want, s)
		}
	}
}
