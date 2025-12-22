package wavelet

import (
	"github.com/ollama/ollama/ml"
)

// CompressionStrategy defines how thresholding is applied.
type CompressionStrategy int

const (
	ThresholdAbsolute CompressionStrategy = iota
	ThresholdRelative
	ThresholdAdaptive
)

// CodecConfig configures the wavelet compression.
type CodecConfig struct {
	Levels       int
	Threshold    float32
	Strategy     CompressionStrategy
	QuantizeBits int
}

// Codec handles the end-to-end compression and decompression.
type Codec struct {
	Transform WaveletTransform
	Config    CodecConfig
	Seeds     map[uint64]*WaveletCoefficients // Fractal seeds for recurring patterns
}

func NewCodec(config CodecConfig) *Codec {
	return &Codec{
		Transform: &SIMDHaarTransform{}, // Use SIMD by default
		Config:    config,
		Seeds:     make(map[uint64]*WaveletCoefficients),
	}
}

func (c *Codec) Compress(data []float32, dtype ml.DType) *WaveletCoefficients {
	coeffs := c.Transform.Decompose(data, c.Config.Levels)
	if coeffs == nil {
		return nil
	}
	coeffs.DType = dtype

	// Apply thresholding based on strategy
	switch c.Config.Strategy {
	case ThresholdAbsolute:
		coeffs.Threshold(c.Config.Threshold)
	case ThresholdRelative:
		// TODO: Implement relative thresholding based on max coefficient
		coeffs.Threshold(c.Config.Threshold)
	}

	return coeffs
}

func (c *Codec) Decompress(coeffs *WaveletCoefficients, targetLevel int) []float32 {
	return c.Transform.Reconstruct(coeffs, targetLevel)
}
