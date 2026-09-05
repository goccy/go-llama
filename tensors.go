package llama

// TensorInfo describes one weight tensor of a loaded model and the kernel
// path the CPU backend gave it.
type TensorInfo struct {
	Name string `json:"name"`
	// Type is the ggml type name ("q4_K", "q8_0", "f16", ...).
	Type string `json:"type"`
	// Shape is ne[0..]: the row length first.
	Shape []int64 `json:"ne"`
	// Buffer names the backend buffer holding the tensor: "CPU_REPACK" for
	// tensors the CPU backend repacked for its GEMV/GEMM kernels (rows a
	// multiple of the repack width, a repackable type), "CPU" otherwise.
	Buffer string `json:"buffer"`
	// VecDotType is the activation type the tensor's dot product quantizes
	// to ("q8_K" for K-quants, "q8_0" / "q8_1" for the legacy types).
	VecDotType string `json:"vec_dot_type"`
}

// Repacked reports whether the CPU backend repacked the tensor for its
// GEMV/GEMM kernels; tensors that are not repacked run the per-row dot.
func (t TensorInfo) Repacked() bool { return t.Buffer == "CPU_REPACK" }

// Tensors lists the model's weight tensors with the buffer each landed in,
// which decides the kernel path (repacked GEMV/GEMM versus per-row dot).
func (m *Model) Tensors() ([]TensorInfo, error) {
	if err := m.use("model tensors"); err != nil {
		return nil, err
	}
	js, err := m.inst.e().LlamaModelTensors(m.h)
	if err != nil {
		return nil, err
	}
	var out struct {
		envelope
		Tensors []TensorInfo `json:"tensors"`
	}
	if err := decode("model tensors", js, &out); err != nil {
		return nil, err
	}
	return out.Tensors, nil
}
