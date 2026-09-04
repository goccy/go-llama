// Package perfgate compares go-llama's throughput against a native
// llama.cpp build measured on the same machine, and gates CI on the
// ratio: a change that makes go-llama more than a threshold slower
// RELATIVE TO NATIVE than the recorded baseline fails.
//
// Comparing ratios rather than raw tok/s cancels the runner-to-runner
// speed spread (the CI pool mixes CPU generations); native and go-llama
// are always measured back to back on the same runner.
package perfgate

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Metric names, shared by baseline files, results and the summary.
const (
	MetricTG = "tg" // token generation: one token at a time (decode)
	MetricPP = "pp" // prompt processing: one batched pass over a long prompt
)

// Measurement is one metric on one architecture: both throughputs and
// their ratio (go-llama / native).
type Measurement struct {
	GoTokS     float64 `json:"go_tok_s"`
	NativeTokS float64 `json:"native_tok_s"`
	Ratio      float64 `json:"ratio"`
}

// Provenance records what produced an architecture's numbers.
type Provenance struct {
	Engine   string `json:"engine,omitempty"`    // llamawasm2go module version
	LlamaCPP string `json:"llama_cpp,omitempty"` // native llama.cpp commit
	CPU      string `json:"cpu,omitempty"`
	Run      string `json:"run,omitempty"` // CI run URL
	Date     string `json:"date,omitempty"`
}

// ArchResult is one architecture's measurements.
type ArchResult struct {
	Metrics  map[string]Measurement `json:"metrics"`
	Recorded Provenance             `json:"recorded"`
}

// Baseline maps Key(arch, cpu) to its recorded result.
type Baseline map[string]ArchResult

// Key names a baseline entry: the architecture and the runner's CPU
// model. The hosted runner pool mixes CPU generations whose native
// llama.cpp and go-llama throughputs do not scale alike (a Zen 4 VM
// that masks AVX-512 keeps native at AVX2 while decode bandwidth
// differs from Zen 3), so a ratio recorded on one model is not a bar
// for another. An entry is judged only against a run on the same
// model; other models are recorded and pass. An empty cpu keys by
// architecture alone.
func Key(arch, cpu string) string {
	cpu = strings.Join(strings.Fields(cpu), " ")
	if cpu == "" {
		return arch
	}
	return arch + "/" + cpu
}

// Verdict is the outcome of comparing one metric against the baseline.
type Verdict struct {
	Metric        string
	Current       Measurement
	BaselineRatio float64 // 0 when the baseline has no entry
	Change        float64 // (current.Ratio / BaselineRatio) - 1; 0 when no baseline
	Regressed     bool
}

// ParseLlamaBench extracts native tok/s from `llama-bench -o json`
// output: the entry with n_gen > 0 and n_prompt == 0 is tg, the one with
// n_prompt > 0 and n_gen == 0 is pp. Best of the per-repetition samples
// is used (the go-llama side is also best-of), falling back to avg_ts
// when samples are absent.
func ParseLlamaBench(data []byte) (map[string]float64, error) {
	var rows []struct {
		NPrompt   int       `json:"n_prompt"`
		NGen      int       `json:"n_gen"`
		AvgTS     float64   `json:"avg_ts"`
		SamplesTS []float64 `json:"samples_ts"`
	}
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("llama-bench json: %w", err)
	}
	out := map[string]float64{}
	for _, r := range rows {
		best := r.AvgTS
		for _, s := range r.SamplesTS {
			if s > best {
				best = s
			}
		}
		switch {
		case r.NGen > 0 && r.NPrompt == 0:
			out[MetricTG] = best
		case r.NPrompt > 0 && r.NGen == 0:
			out[MetricPP] = best
		}
	}
	if _, ok := out[MetricTG]; !ok {
		return nil, errors.New("llama-bench json: no tg entry (n_gen > 0, n_prompt == 0)")
	}
	if _, ok := out[MetricPP]; !ok {
		return nil, errors.New("llama-bench json: no pp entry (n_prompt > 0, n_gen == 0)")
	}
	return out, nil
}

// goBenchLine matches a `go test -bench` result line, e.g.
//
//	BenchmarkDecode-4   64   9371895 ns/op   106.7 tok/s
var goBenchLine = regexp.MustCompile(`^(Benchmark\w+?)(?:-\d+)?\s+\d+\s+(.*)$`)

// ParseGoBench extracts the best (highest) value of a reported metric
// unit per benchmark from `go test -bench` output — with -count=N every
// run is a line, and best-of-N is the noise-robust choice for a gate.
// unit is the ReportMetric unit ("tok/s", "prompt_tok/s").
func ParseGoBench(text, unit string) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(strings.NewReader(text))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		m := goBenchLine.FindStringSubmatch(strings.TrimSpace(sc.Text()))
		if m == nil {
			continue
		}
		fields := strings.Fields(m[2])
		for i := 1; i < len(fields); i++ {
			if fields[i] != unit {
				continue
			}
			v, err := strconv.ParseFloat(fields[i-1], 64)
			if err != nil {
				return nil, fmt.Errorf("go bench line %q: %w", sc.Text(), err)
			}
			if v > out[m[1]] {
				out[m[1]] = v
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("go bench output: no Benchmark line reporting %q", unit)
	}
	return out, nil
}

// Compare evaluates every metric of cur against base (the same
// architecture's baseline entry; nil when none is recorded) at the given
// regression threshold (0.10 = 10%). A metric regresses when its ratio
// to native fell below baseline.ratio * (1 - threshold). Verdicts are
// returned in a stable metric order.
func Compare(cur ArchResult, base *ArchResult, threshold float64) []Verdict {
	names := make([]string, 0, len(cur.Metrics))
	for n := range cur.Metrics {
		names = append(names, n)
	}
	sort.Strings(names)
	var out []Verdict
	for _, n := range names {
		v := Verdict{Metric: n, Current: cur.Metrics[n]}
		if base != nil {
			if b, ok := base.Metrics[n]; ok && b.Ratio > 0 {
				v.BaselineRatio = b.Ratio
				v.Change = v.Current.Ratio/b.Ratio - 1
				v.Regressed = v.Current.Ratio < b.Ratio*(1-threshold)
			}
		}
		out = append(out, v)
	}
	return out
}

// LoadBaseline reads a baseline file; a missing file is an empty
// baseline (the gate then records without judging).
func LoadBaseline(path string) (Baseline, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Baseline{}, nil
	}
	if err != nil {
		return nil, err
	}
	var b Baseline
	if len(strings.TrimSpace(string(data))) == 0 {
		return Baseline{}, nil
	}
	if err := json.Unmarshal(data, &b); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return b, nil
}

// SaveBaseline writes the baseline with stable formatting.
func SaveBaseline(path string, b Baseline) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// Summary renders the verdicts as a Markdown table (for the CI step
// summary and the log).
func Summary(arch string, vs []Verdict, threshold float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "### go-llama vs native llama.cpp (%s), gate: ratio must stay within %.0f%% of baseline\n\n", arch, threshold*100)
	b.WriteString("| metric | native tok/s | go-llama tok/s | ratio | baseline ratio | change | result |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---|\n")
	for _, v := range vs {
		baseline, change, result := "—", "—", "recorded (no baseline)"
		if v.BaselineRatio > 0 {
			baseline = fmt.Sprintf("%.3f", v.BaselineRatio)
			change = fmt.Sprintf("%+.1f%%", v.Change*100)
			result = "ok"
			if v.Regressed {
				result = "**REGRESSION**"
			}
		}
		fmt.Fprintf(&b, "| %s | %.1f | %.1f | %.3f | %s | %s | %s |\n",
			v.Metric, v.Current.NativeTokS, v.Current.GoTokS, v.Current.Ratio, baseline, change, result)
	}
	return b.String()
}
