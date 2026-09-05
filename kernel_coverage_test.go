package llama_test

import (
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/goccy/llamawasm2go/base"
)

// kernelIndex is the bundle's kernel index (kernels/asm/kernels.json in
// llama-wasm): every exported kernel a native body replaces, with what it
// computes, the tensor type it serves and the architectures carrying a body.
type kernelIndex struct {
	Kernels []struct {
		Export string   `json:"export"`
		Role   string   `json:"role"`
		Quant  string   `json:"quant"`
		Arches []string `json:"arches"`
	} `json:"kernels"`
}

// nativeKernels returns, per tensor type, the exports with a native body
// for arch in each role ("gemv", "gemm", "vec_dot").
func nativeKernels(t *testing.T, arch string) map[string]map[string]string {
	t.Helper()
	var idx kernelIndex
	if err := json.Unmarshal(base.AsmKernels, &idx); err != nil {
		t.Fatalf("bundle kernel index: %v", err)
	}
	out := map[string]map[string]string{}
	for _, k := range idx.Kernels {
		for _, a := range k.Arches {
			if a != arch {
				continue
			}
			if out[k.Quant] == nil {
				out[k.Quant] = map[string]string{}
			}
			out[k.Quant][k.Role] = k.Export
		}
	}
	return out
}

// TestKernelCoverage maps every weight tensor of the test model to the
// kernel it runs on and reports whether the bundle's native bodies reach
// it: a repacked tensor needs the type's repack GEMV and GEMM overrides, a
// non-repacked quantized tensor its vec_dot override. The per-type table
// is logged (an inspection tool, not a gate: which types a model uses is
// the model's choice); tensors whose type has repack kernels but that the backend
// did not repack (rows not a multiple of 8, or a shared embedding) are
// reported, not failed. Run with GO_LLAMA_TEST_MODEL to inspect a real
// model; the default tiny model is f32/f16 and exercises no quant kernel.
func TestKernelCoverage(t *testing.T) {
	m := load(t)
	defer m.Close()
	tensors, err := m.Tensors()
	if err != nil {
		t.Fatal(err)
	}
	if len(tensors) == 0 {
		t.Fatal("no tensors reported")
	}
	native := nativeKernels(t, runtime.GOARCH)

	type row struct {
		path     string // "repack" or "vec_dot"
		kernels  []string
		asm      bool
		count    int
		params   int64
		notRepak int // tensors of a repackable type left on the per-row path
	}
	rows := map[string]*row{}
	var total int64
	for _, tn := range tensors {
		if len(tn.Shape) < 2 {
			continue // embeddings-as-lookups and 1-D norms are not matmuls
		}
		switch tn.Type {
		case "f32", "f16", "bf16":
			continue
		}
		params := int64(1)
		for _, d := range tn.Shape {
			params *= d
		}
		total += params
		key := tn.Type
		var kernels []string
		asm := true
		if tn.Repacked() {
			key += " (repack)"
			for _, role := range []string{"gemv", "gemm"} {
				if k, ok := native[tn.Type][role]; ok {
					kernels = append(kernels, k)
				} else {
					asm = false
					kernels = append(kernels, "no "+role+" body")
				}
			}
		} else {
			key += " (vec_dot)"
			if k, ok := native[tn.Type]["vec_dot"]; ok {
				kernels = append(kernels, k)
			} else {
				asm = false
				kernels = append(kernels, "no vec_dot body")
			}
		}
		r := rows[key]
		if r == nil {
			r = &row{kernels: kernels, asm: asm}
			if tn.Repacked() {
				r.path = "repack"
			} else {
				r.path = "vec_dot"
			}
			rows[key] = r
		}
		r.count++
		r.params += params
		if _, ok := native[tn.Type]["gemv"]; !tn.Repacked() && ok {
			r.notRepak++
		}
	}
	if total == 0 {
		t.Skip("no quantized weight tensors: the model exercises no quant kernel")
	}
	keys := make([]string, 0, len(rows))
	for k := range rows {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return rows[keys[i]].params > rows[keys[j]].params })
	var covered int64
	var sb strings.Builder
	fmt.Fprintf(&sb, "kernel coverage on %s (%s):\n", runtime.GOARCH, modelPath())
	for _, k := range keys {
		r := rows[k]
		status := "native asm"
		if !r.asm {
			status = "no native body (transpiled wasm code)"
		} else {
			covered += r.params
		}
		fmt.Fprintf(&sb, "  %-20s %3d tensors %5.1f%% of quant params  %s  %s\n", k, r.count, 100*float64(r.params)/float64(total), status, strings.Join(r.kernels, ", "))
		if r.notRepak > 0 {
			fmt.Fprintf(&sb, "  %-20s %3d tensors of a repackable type run the per-row dot (rows not a multiple of 8, or shared with a lookup)\n", "", r.notRepak)
		}
	}
	fmt.Fprintf(&sb, "  native asm reaches %.1f%% of the quantized parameters", 100*float64(covered)/float64(total))
	t.Log(sb.String())
}
