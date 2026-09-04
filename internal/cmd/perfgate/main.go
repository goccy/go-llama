// Command perfgate is the CI entry point of internal/perfgate.
//
//	perfgate compare -arch ARCH -native native.json -tg tg.txt -pp pp.txt \
//	    -baseline bench/baseline.json -out result.json [-threshold 0.10] \
//	    [-summary $GITHUB_STEP_SUMMARY] [-engine V] [-llama-cpp SHA] [-cpu S] [-run URL]
//
// exits 1 when any metric's ratio to native fell more than threshold
// below the recorded baseline for the same architecture and CPU model;
// with no baseline for that pair it records and passes.
//
//	perfgate baseline -baseline bench/baseline.json -result result.json
//
// merges a compare result (downloaded from the CI artifact) into the
// baseline file — the deliberate, reviewable way to move the bar.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/goccy/go-llama/internal/perfgate"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "compare":
		err = compare(os.Args[2:])
	case "baseline":
		err = baseline(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "perfgate:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: perfgate compare|baseline [flags]")
	os.Exit(2)
}

// result is what compare writes: the arch's measurements plus the
// gate's verdict, so the baseline command can merge it verbatim.
type result struct {
	Arch      string              `json:"arch"`
	Threshold float64             `json:"threshold"`
	Result    perfgate.ArchResult `json:"result"`
	Regressed []string            `json:"regressed"`
}

func compare(args []string) error {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	arch := fs.String("arch", "", "architecture key (amd64, arm64)")
	nativePath := fs.String("native", "", "llama-bench -o json output")
	tgPath := fs.String("tg", "", "go test -bench BenchmarkDecode output")
	ppPath := fs.String("pp", "", "go test -bench BenchmarkPromptEval output")
	basePath := fs.String("baseline", "bench/baseline.json", "baseline file")
	outPath := fs.String("out", "", "result json to write")
	threshold := fs.Float64("threshold", 0.10, "allowed relative drop of the go/native ratio")
	summaryPath := fs.String("summary", "", "append the Markdown summary to this file (GITHUB_STEP_SUMMARY)")
	engine := fs.String("engine", "", "engine module version (provenance)")
	llamaCPP := fs.String("llama-cpp", "", "native llama.cpp commit (provenance)")
	cpu := fs.String("cpu", "", "runner CPU (provenance)")
	run := fs.String("run", "", "CI run URL (provenance)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *arch == "" || *nativePath == "" || *tgPath == "" || *ppPath == "" || *outPath == "" {
		return fmt.Errorf("compare: -arch, -native, -tg, -pp and -out are required")
	}
	nativeData, err := os.ReadFile(*nativePath)
	if err != nil {
		return err
	}
	native, err := perfgate.ParseLlamaBench(nativeData)
	if err != nil {
		return err
	}
	tgOut, err := os.ReadFile(*tgPath)
	if err != nil {
		return err
	}
	tg, err := perfgate.ParseGoBench(string(tgOut), "tok/s")
	if err != nil {
		return err
	}
	ppOut, err := os.ReadFile(*ppPath)
	if err != nil {
		return err
	}
	pp, err := perfgate.ParseGoBench(string(ppOut), "prompt_tok/s")
	if err != nil {
		return err
	}
	goTG, ok := tg["BenchmarkDecode"]
	if !ok {
		return fmt.Errorf("no BenchmarkDecode result in %s", *tgPath)
	}
	goPP, ok := pp["BenchmarkPromptEval"]
	if !ok {
		return fmt.Errorf("no BenchmarkPromptEval result in %s", *ppPath)
	}
	cur := perfgate.ArchResult{
		Metrics: map[string]perfgate.Measurement{
			perfgate.MetricTG: {GoTokS: goTG, NativeTokS: native[perfgate.MetricTG], Ratio: goTG / native[perfgate.MetricTG]},
			perfgate.MetricPP: {GoTokS: goPP, NativeTokS: native[perfgate.MetricPP], Ratio: goPP / native[perfgate.MetricPP]},
		},
		Recorded: perfgate.Provenance{Engine: *engine, LlamaCPP: *llamaCPP, CPU: *cpu, Run: *run, Date: time.Now().UTC().Format("2006-01-02")},
	}
	base, err := perfgate.LoadBaseline(*basePath)
	if err != nil {
		return err
	}
	var baseArch *perfgate.ArchResult
	key := perfgate.Key(*arch, *cpu)
	if b, ok := base[key]; ok {
		baseArch = &b
	}
	verdicts := perfgate.Compare(cur, baseArch, *threshold)
	res := result{Arch: *arch, Threshold: *threshold, Result: cur}
	for _, v := range verdicts {
		if v.Regressed {
			res.Regressed = append(res.Regressed, v.Metric)
		}
	}
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*outPath, append(data, '\n'), 0o644); err != nil {
		return err
	}
	summary := perfgate.Summary(key, verdicts, *threshold)
	fmt.Print(summary)
	if *summaryPath != "" {
		f, err := os.OpenFile(*summaryPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		if _, err := f.WriteString(summary + "\n"); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	if baseArch == nil {
		fmt.Printf("perfgate: no baseline for %s in %s — recorded only. Seed it with:\n  go run ./internal/cmd/perfgate baseline -baseline %s -result %s\n", key, *basePath, *basePath, *outPath)
	}
	if len(res.Regressed) > 0 {
		return fmt.Errorf("%s: %v regressed more than %.0f%% relative to native vs the baseline", key, res.Regressed, *threshold*100)
	}
	return nil
}

func baseline(args []string) error {
	fs := flag.NewFlagSet("baseline", flag.ExitOnError)
	basePath := fs.String("baseline", "bench/baseline.json", "baseline file to update")
	resPath := fs.String("result", "", "compare result json (from the CI artifact)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *resPath == "" {
		return fmt.Errorf("baseline: -result is required")
	}
	data, err := os.ReadFile(*resPath)
	if err != nil {
		return err
	}
	var res result
	if err := json.Unmarshal(data, &res); err != nil {
		return fmt.Errorf("%s: %w", *resPath, err)
	}
	if res.Arch == "" {
		return fmt.Errorf("%s: no arch", *resPath)
	}
	base, err := perfgate.LoadBaseline(*basePath)
	if err != nil {
		return err
	}
	base[perfgate.Key(res.Arch, res.Result.Recorded.CPU)] = res.Result
	if err := perfgate.SaveBaseline(*basePath, base); err != nil {
		return err
	}
	fmt.Printf("perfgate: %s baseline updated from %s\n", res.Arch, *resPath)
	return nil
}
