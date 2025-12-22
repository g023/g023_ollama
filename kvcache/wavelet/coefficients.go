package wavelet

import (
	"github.com/ollama/ollama/ml"
)

// WaveletCoefficients stores the result of a wavelet decomposition.
type WaveletCoefficients struct {
	Approximation []float32   // Coarsest scale coefficients
	Details       [][]float32 // Detail coefficients at each level
	Shape         []int       // Original data shape
	Levels        int         // Number of decomposition levels
	DType         ml.DType    // Original data type
}

// Threshold applies hard thresholding to the detail coefficients.
func (wc *WaveletCoefficients) Threshold(threshold float32) {
	for l := 0; l < wc.Levels; l++ {
		for i := range wc.Details[l] {
			if wc.Details[l][i] < threshold && wc.Details[l][i] > -threshold {
				wc.Details[l][i] = 0
			}
		}
	}
}

// Size returns the total number of coefficients stored.
func (wc *WaveletCoefficients) Size() int {
	size := len(wc.Approximation)
	for _, d := range wc.Details {
		size += len(d)
	}
	return size
}

// SparseSize returns the number of non-zero coefficients.
func (wc *WaveletCoefficients) SparseSize() int {
	count := 0
	for _, v := range wc.Approximation {
		if v != 0 {
			count++
		}
	}
	for _, d := range wc.Details {
		for _, v := range d {
			if v != 0 {
				count++
			}
		}
	}
	return count
}
