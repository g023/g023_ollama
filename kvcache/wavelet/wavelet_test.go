package wavelet

import (
	"math"
	"testing"
)

func TestHaarTransform(t *testing.T) {
	h := &HaarTransform{}
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	levels := 2

	coeffs := h.Decompose(data, levels)
	if coeffs.Levels != levels {
		t.Errorf("expected %d levels, got %d", levels, coeffs.Levels)
	}

	reconstructed := h.Reconstruct(coeffs, levels)
	for i := range data {
		if math.Abs(float64(data[i]-reconstructed[i])) > 1e-5 {
			t.Errorf("mismatch at index %d: expected %f, got %f", i, data[i], reconstructed[i])
		}
	}
}

func TestSIMDHaarTransform(t *testing.T) {
	s := &SIMDHaarTransform{}
	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i)
	}
	levels := 4

	coeffs := s.Decompose(data, levels)
	reconstructed := s.Reconstruct(coeffs, levels)
	
	for i := range data {
		if math.Abs(float64(data[i]-reconstructed[i])) > 1e-3 {
			t.Errorf("mismatch at index %d: expected %f, got %f", i, data[i], reconstructed[i])
		}
	}
}

func BenchmarkHaarDecompose(b *testing.B) {
	h := &HaarTransform{}
	data := make([]float32, 4096)
	for i := 0; i < b.N; i++ {
		h.Decompose(data, 5)
	}
}

func BenchmarkSIMDHaarDecompose(b *testing.B) {
	s := &SIMDHaarTransform{}
	data := make([]float32, 4096)
	for i := 0; i < b.N; i++ {
		s.Decompose(data, 5)
	}
}
