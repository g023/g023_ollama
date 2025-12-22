package wavelet

// SIMDHaarTransform provides an optimized Haar transform using manual loop unrolling
// and hints for the compiler to use SIMD instructions.
type SIMDHaarTransform struct {
	HaarTransform
}

// Decompose implements an optimized version of Haar decomposition.
func (s *SIMDHaarTransform) Decompose(data []float32, levels int) *WaveletCoefficients {
	// In a real production environment, we would use Go assembly or 
	// specialized intrinsics for AVX2/NEON.
	// Here we implement a highly unrolled version that Go's compiler can vectorize.
	
	n := len(data)
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

	sqrt05 := float32(0.70710678118)

	for l := 0; l < levels; l++ {
		size := paddedN >> uint(l)
		if size < 8 { // Fallback to standard for small sizes
			return s.HaarTransform.Decompose(data, levels)
		}

		half := size / 2
		approx := make([]float32, half)
		detail := make([]float32, half)

		// Unrolled loop for better vectorization
		for i := 0; i < half-3; i += 4 {
			v0_1 := current[2*i]
			v0_2 := current[2*i+1]
			v1_1 := current[2*(i+1)]
			v1_2 := current[2*(i+1)+1]
			v2_1 := current[2*(i+2)]
			v2_2 := current[2*(i+2)+1]
			v3_1 := current[2*(i+3)]
			v3_2 := current[2*(i+3)+1]

			approx[i] = (v0_1 + v0_2) * sqrt05
			detail[i] = (v0_1 - v0_2) * sqrt05
			approx[i+1] = (v1_1 + v1_2) * sqrt05
			detail[i+1] = (v1_1 - v1_2) * sqrt05
			approx[i+2] = (v2_1 + v2_2) * sqrt05
			detail[i+2] = (v2_1 - v2_2) * sqrt05
			approx[i+3] = (v3_1 + v3_2) * sqrt05
			detail[i+3] = (v3_1 - v3_2) * sqrt05
		}
		
		// Handle remainder
		for i := (half / 4) * 4; i < half; i++ {
			v1 := current[2*i]
			v2 := current[2*i+1]
			approx[i] = (v1 + v2) * sqrt05
			detail[i] = (v1 - v2) * sqrt05
		}

		coeffs.Details[l] = detail
		current = approx
	}

	coeffs.Approximation = current
	return coeffs
}
