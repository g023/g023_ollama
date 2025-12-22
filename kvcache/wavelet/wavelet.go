package wavelet

import (
	"math"
)

// WaveletTransform defines the interface for wavelet decomposition and reconstruction.
type WaveletTransform interface {
	Decompose(data []float32, levels int) *WaveletCoefficients
	Reconstruct(coeffs *WaveletCoefficients, targetLevel int) []float32
}

// HaarTransform implements the fast Haar wavelet transform.
type HaarTransform struct{}

// Decompose performs multi-level Haar wavelet decomposition.
func (h *HaarTransform) Decompose(data []float32, levels int) *WaveletCoefficients {
	n := len(data)
	if n == 0 {
		return nil
	}

	// Ensure n is a power of 2 for simplicity in this implementation
	// In production, we'd handle arbitrary lengths with padding
	paddedN := 1
	for paddedN < n {
		paddedN <<= 1
	}

	current := make([]float32, paddedN)
	copy(current, data)

	coeffs := &WaveletCoefficients{
		Details: make([][]float32, levels),
		Shape:   []int{n},
		Levels:  levels,
	}

	for l := 0; l < levels; l++ {
		size := paddedN >> uint(l)
		if size < 2 {
			coeffs.Levels = l
			coeffs.Details = coeffs.Details[:l]
			break
		}

		half := size / 2
		approx := make([]float32, half)
		detail := make([]float32, half)

		for i := 0; i < half; i++ {
			val1 := current[2*i]
			val2 := current[2*i+1]
			approx[i] = (val1 + val2) * float32(math.Sqrt(0.5))
			detail[i] = (val1 - val2) * float32(math.Sqrt(0.5))
		}

		coeffs.Details[l] = detail
		copy(current, approx)
	}

	coeffs.Approximation = current[:paddedN>>uint(coeffs.Levels)]
	return coeffs
}

// Reconstruct performs inverse Haar wavelet transform.
func (h *HaarTransform) Reconstruct(coeffs *WaveletCoefficients, targetLevel int) []float32 {
	if coeffs == nil {
		return nil
	}

	if targetLevel > coeffs.Levels {
		targetLevel = coeffs.Levels
	}

	current := make([]float32, len(coeffs.Approximation))
	copy(current, coeffs.Approximation)

	for l := targetLevel - 1; l >= 0; l-- {
		detail := coeffs.Details[l]
		size := len(current)
		next := make([]float32, size*2)

		for i := 0; i < size; i++ {
			a := current[i]
			d := detail[i]
			next[2*i] = (a + d) * float32(math.Sqrt(0.5))
			next[2*i+1] = (a - d) * float32(math.Sqrt(0.5))
		}
		current = next
	}

	// Trim padding if necessary
	if len(current) > coeffs.Shape[0] {
		return current[:coeffs.Shape[0]]
	}
	return current
}
