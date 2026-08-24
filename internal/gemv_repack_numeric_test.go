package internal

// Numeric regression tests for the engine's repacked q8_0 gemv
// kernel: the bundle's debug export runs the real (spliced, on
// arm64) kernel over linear memory, and the results are compared
// against a straight-line Go reference of the q8_0x4 semantics.
//
// These tests exist because the kernel's fused-loop form once froze
// the activation scale at loop entry — every block was scaled by the
// first block's d — while all-ones smoke inputs still passed. The
// varying-scale cases below fail loudly on that whole defect class.

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	wasm2go "github.com/goccy/llamawasm2go"
	"github.com/goccy/llamawasm2go/base"
)

// repackF16Table is the f16->f32 table's base address in the pinned
// engine bundle. The transpiler derives it from the engine's init
// loop and logs it at build time ("f16 table auto-detected at ...",
// recorded in the llama-wasm release build log); it moves with any
// engine rebuild, so re-read it from the log (or the bundle's
// load32_splat memarg offsets) when bumping the bundle dependency.
const repackF16Table = 8793040

func f16Bits(f float32) uint16 {
	// Exact for the small power-of-two scales the tests use.
	switch f {
	case 1.0:
		return 0x3C00
	case 0.5:
		return 0x3800
	case 2.0:
		return 0x4000
	case 0.25:
		return 0x3400
	}
	panic("unsupported scale")
}

func f16Val(b uint16) float32 {
	sign := uint32(b&0x8000) << 16
	mag := uint32(b&0x7fff) << 13
	return math.Float32frombits(sign|mag) * float32(math.Pow(2, 112))
}

// fillF16Table builds ggml's f32<-f16 lookup table in guest memory: a
// bare engine never runs ggml_init (that happens at model load), so
// the table region is zero and every table-gathered scale would
// collapse to 0 without this.
func fillF16Table(mem []byte) {
	for i := 0; i < 1<<16; i++ {
		binary.LittleEndian.PutUint32(mem[repackF16Table+4*i:], math.Float32bits(f16Val(uint16(i))))
	}
}

// gemvHarness provisions an engine whose top 8 MiB of linear memory
// serve as the test arena: weights at vx, activations at vy, the
// result vector at sPtr.
type gemvHarness struct {
	mem          []byte
	g            *base.Module
	vx, vy, sPtr int64
}

func newGemvHarness(t *testing.T) *gemvHarness {
	t.Helper()
	m, err := NewEngine(Options{})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	g := m.g
	mem := g.Memory
	fillF16Table(mem)
	base := int64(g.MemSize.Load()) - (8 << 20)
	if base <= 0 {
		t.Fatalf("memory too small: %d", len(mem))
	}
	return &gemvHarness{mem: mem, g: g, vx: base, vy: base + 1<<20, sPtr: base + 2<<20}
}

// putWeightBlock writes one block_q8_0x4 (4 interleaved columns): 4
// f16 column scales then 128 quantized bytes.
func (h *gemvHarness) putWeightBlock(block int64, scales [4]float32, qs func(i int) int8) {
	blk := h.vx + block*136
	for j, s := range scales {
		binary.LittleEndian.PutUint16(h.mem[blk+int64(2*j):], f16Bits(s))
	}
	for i := 0; i < 128; i++ {
		h.mem[blk+8+int64(i)] = byte(qs(i))
	}
}

// putActBlock writes one block_q8_0: an f16 scale then 32 quantized
// bytes.
func (h *gemvHarness) putActBlock(block int64, scale float32, qs func(i int) int8) {
	blk := h.vy + block*34
	binary.LittleEndian.PutUint16(h.mem[blk:], f16Bits(scale))
	for i := 0; i < 32; i++ {
		h.mem[blk+2+int64(i)] = byte(qs(i))
	}
}

func (h *gemvHarness) run(n int32, nc int) {
	wasm2go.DbgGemvQ8_0_4x4(h.g, n, h.sPtr, int64(nc), h.vx, h.vy, 1, int32(nc))
}

func (h *gemvHarness) col(j int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(h.mem[h.sPtr+int64(4*j):]))
}

func assertCols(t *testing.T, h *gemvHarness, nc int, want func(j int) float32) {
	t.Helper()
	for j := 0; j < nc; j++ {
		got, w := h.col(j), want(j)
		diff := math.Abs(float64(got - w))
		if tol := 1e-3 * (1 + math.Abs(float64(w))); diff > tol {
			t.Errorf("col %d: got %g want %g (diff %g)", j, got, w, diff)
		}
	}
}

// TestGemvRepackNumeric drives random blocks through the kernel and
// checks every output column against the reference dot product.
func TestGemvRepackNumeric(t *testing.T) {
	h := newGemvHarness(t)
	const (
		nb = 6 // blocks per row
		nc = 8 // output columns (2 groups of 4)
	)
	scales := []float32{1.0, 0.5, 2.0, 0.25}
	rng := rand.New(rand.NewSource(7))

	qb := make([][][]int8, nc/4) // [group][block][128]
	dB := make([][][]float32, nc/4)
	for x := 0; x < nc/4; x++ {
		qb[x] = make([][]int8, nb)
		dB[x] = make([][]float32, nb)
		for l := 0; l < nb; l++ {
			var s [4]float32
			for j := range s {
				s[j] = scales[rng.Intn(len(scales))]
			}
			dB[x][l] = s[:]
			q := make([]int8, 128)
			for i := range q {
				q[i] = int8(rng.Intn(255) - 127)
			}
			qb[x][l] = q
			blk := int64(x*nb + l)
			h.putWeightBlock(blk, s, func(i int) int8 { return q[i] })
		}
	}
	qa := make([][]int8, nb)
	dA := make([]float32, nb)
	for l := 0; l < nb; l++ {
		dA[l] = scales[rng.Intn(len(scales))]
		q := make([]int8, 32)
		for i := range q {
			q[i] = int8(rng.Intn(255) - 127)
		}
		qa[l] = q
		h.putActBlock(int64(l), dA[l], func(i int) int8 { return q[i] })
	}

	h.run(nb*32, nc)
	assertCols(t, h, nc, func(col int) float32 {
		x, j := col/4, col%4
		var want float32
		for l := 0; l < nb; l++ {
			sum := int32(0)
			for k := 0; k < 8; k++ {
				for i := 0; i < 4; i++ {
					sum += int32(qb[x][l][k*16+j*4+i]) * int32(qa[l][k*4+i])
				}
			}
			want += float32(sum) * dB[x][l][j] * dA[l]
		}
		return want
	})
}

// TestGemvRepackUnitBlock: one block, unit scales, all-ones quants —
// every column must come out exactly 32. A sentinel prefill
// distinguishes "stored zeros" from "never stored".
func TestGemvRepackUnitBlock(t *testing.T) {
	h := newGemvHarness(t)
	h.putWeightBlock(0, [4]float32{1, 1, 1, 1}, func(int) int8 { return 1 })
	h.putActBlock(0, 1, func(int) int8 { return 1 })
	for i := 0; i < 16; i++ {
		h.mem[h.sPtr+int64(i)] = 0xEE
	}
	h.run(32, 4)
	assertCols(t, h, 4, func(int) float32 { return 32 })
}

// TestGemvRepackWeightScaleAdvance: two blocks whose WEIGHT scales
// differ (1 then 2). want 32*1+32*2 = 96; 32 or 64 would mean a lost
// or frozen second block.
func TestGemvRepackWeightScaleAdvance(t *testing.T) {
	h := newGemvHarness(t)
	for l := int64(0); l < 2; l++ {
		s := float32(1)
		if l == 1 {
			s = 2
		}
		h.putWeightBlock(l, [4]float32{s, s, s, s}, func(int) int8 { return 1 })
		h.putActBlock(l, 1, func(int) int8 { return 1 })
	}
	h.run(64, 4)
	assertCols(t, h, 4, func(int) float32 { return 96 })
}

// TestGemvRepackActScaleAdvance: two blocks whose ACTIVATION scales
// differ (1 then 2) — the exact case the fused loop once broke by
// freezing the entry-time scale. want 96; frozen-first gives 64,
// frozen-last 128.
func TestGemvRepackActScaleAdvance(t *testing.T) {
	h := newGemvHarness(t)
	for l := int64(0); l < 2; l++ {
		s := float32(1)
		if l == 1 {
			s = 2
		}
		h.putWeightBlock(l, [4]float32{1, 1, 1, 1}, func(int) int8 { return 1 })
		h.putActBlock(l, s, func(int) int8 { return 1 })
	}
	h.run(64, 4)
	assertCols(t, h, 4, func(int) float32 { return 96 })
}
