package internal

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	wasm2go "github.com/goccy/llamawasm2go"
)

// TestVecDotQ5KTile pins the two calling conventions of the engine's
// q5_K x q8_K dot to each other. type_traits gives Q5_K nrows = 2, so the
// mul_mat driver calls the dot with nrc == 2 for a 2x2 tile (two weight
// rows against two activation columns, results at s[0], s[1], s[bs],
// s[bs+1]) whenever the shapes allow and with nrc == 1 otherwise. The two
// must agree bit for bit or a decode's results depend on the batch shape.
// On arm64 and AVX2 the tile is the native kernel's; on the pure-Go path
// (GOAMD64=v1) it is the wasm body's four-dot fallback, which no other
// test reaches: every Q5_K tensor of the models the suite runs is
// repacked and never takes the per-row dot.
func TestVecDotQ5KTile(t *testing.T) {
	const (
		n        = 1024 // four super-blocks
		q5kBlock = 176  // d f16 | dmin f16 | scales[12] | qh[32] | qs[128]
		q8kBlock = 292  // d f32 | qs[256] | bsums[16] i16
	)
	h := newGemvHarness(t)
	rng := rand.New(rand.NewSource(5))
	nb := n / 256
	x0, x1 := h.vx, h.vx+int64(nb*q5kBlock)
	y0, y1 := h.vy, h.vy+int64(nb*q8kBlock)
	fillQ5K := func(at int64) {
		for b := 0; b < nb; b++ {
			blk := at + int64(b*q5kBlock)
			binary.LittleEndian.PutUint16(h.mem[blk:], f16Bits([]float32{0.25, 0.5, 1, 2}[rng.Intn(4)]))
			binary.LittleEndian.PutUint16(h.mem[blk+2:], f16Bits([]float32{0.25, 0.5, 1, 2}[rng.Intn(4)]))
			for i := 4; i < q5kBlock; i++ {
				h.mem[blk+int64(i)] = byte(rng.Intn(256))
			}
		}
	}
	fillQ8K := func(at int64) {
		for b := 0; b < nb; b++ {
			blk := at + int64(b*q8kBlock)
			binary.LittleEndian.PutUint32(h.mem[blk:], math.Float32bits(float32(rng.Intn(200)+1)/64))
			var sums [16]int16
			for i := 0; i < 256; i++ {
				q := int8(rng.Intn(255) - 127)
				h.mem[blk+4+int64(i)] = byte(q)
				sums[i/16] += int16(q)
			}
			for j, s := range sums {
				binary.LittleEndian.PutUint16(h.mem[blk+260+int64(2*j):], uint16(s))
			}
		}
	}
	fillQ5K(x0)
	fillQ5K(x1)
	fillQ8K(y0)
	fillQ8K(y1)

	const bs = 16 // floats between the tile's rows
	s := h.sPtr
	for i := 0; i < 2*bs; i++ {
		binary.LittleEndian.PutUint32(h.mem[s+int64(4*i):], math.Float32bits(12345))
	}
	wasm2go.DbgVecDotQ5KQ8K(h.g, n, s, bs, x0, int64(nb*q5kBlock), y0, int64(nb*q8kBlock), 2)
	tile := [4]float32{h.col(0), h.col(1), h.col(bs), h.col(bs + 1)}

	single := func(x, y int64) float32 {
		wasm2go.DbgVecDotQ5KQ8K(h.g, n, s+256, 0, x, 0, y, 0, 1)
		return math.Float32frombits(binary.LittleEndian.Uint32(h.mem[s+256:]))
	}
	want := [4]float32{single(x0, y0), single(x1, y0), single(x0, y1), single(x1, y1)}
	for i := range tile {
		if tile[i] != want[i] {
			t.Errorf("tile lane %d = %v, single dot %v: the nrc == 2 and nrc == 1 paths must be bit-identical", i, tile[i], want[i])
		}
		if math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) || want[i] == 0 {
			t.Errorf("single dot %d = %v: not a plausible dot of random blocks", i, want[i])
		}
	}
	// The tile writes exactly its four lanes.
	for _, i := range []int{2, 3, bs - 1, bs + 2} {
		if v := h.col(i); v != 12345 {
			t.Errorf("s[%d] = %v: written by the tile", i, v)
		}
	}
}
