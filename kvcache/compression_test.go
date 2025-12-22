package kvcache

import (
	"testing"

	"github.com/ollama/ollama/kvcache/wavelet"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

type mockBackend struct {
	ml.Backend
}

func (m *mockBackend) NewContextSize(n int) ml.Context {
	return &mockContext{}
}

type mockContext struct {
	ml.Context
}

func (m *mockContext) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	return &mockTensor{shape: shape}
}

type mockTensor struct {
	ml.Tensor
	shape []int
}

func (m *mockTensor) Dim(n int) int { return m.shape[n] }
func (m *mockTensor) Floats() []float32 {
	size := 1
	for _, d := range m.shape {
		size *= d
	}
	return make([]float32, size)
}

func TestCompressionIntegration(t *testing.T) {
	c := NewCausalCache(nil)
	backend := &mockBackend{}
	c.Init(backend, ml.DTypeF32, 1, 100, 10)

	c.compressionEnabled = true
	c.compressionThreshold = 10
	c.compressionConfig = &wavelet.CodecConfig{
		Levels:    2,
		Threshold: 0.01,
	}

	// Manually set up tensors for layer 0
	c.keys[0] = &mockTensor{shape: []int{128, 1, 100}}
	c.values[0] = &mockTensor{shape: []int{128, 1, 100}}

	// Simulate some data
	_ = input.Batch{
		Positions: []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		Sequences: make([]int, 16),
	}
	
	// We need a real context for StartForward if we want it to work
	// But for this test, we can just manually set up the cells
	for i := 0; i < 16; i++ {
		c.cells[i].pos = int32(i)
		c.cells[i].seqMask.set(0)
	}
	
	// Manually trigger compression
	c.compressOldSegments(15)

	// Check if old cells are compressed
	compressedCount := 0
	for i := 0; i < 16; i++ {
		if c.cells[i].compressed {
			compressedCount++
		}
	}

	if compressedCount == 0 {
		t.Errorf("expected some cells to be compressed, got 0")
	}
	
	t.Logf("Compressed %d cells", compressedCount)
}
