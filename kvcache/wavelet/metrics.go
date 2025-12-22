package wavelet

import (
	"sync/atomic"
	"time"
)

// Metrics tracks wavelet compression performance.
type Metrics struct {
	TotalCompressions   atomic.Uint64
	TotalDecompressions atomic.Uint64
	TotalSavedBytes     atomic.Uint64
	TotalOriginalBytes  atomic.Uint64
	CompressionTime     atomic.Int64 // nanoseconds
	DecompressionTime   atomic.Int64 // nanoseconds
}

var GlobalMetrics Metrics

func RecordCompression(original, compressed uint64, duration time.Duration) {
	GlobalMetrics.TotalCompressions.Add(1)
	GlobalMetrics.TotalOriginalBytes.Add(original)
	if original > compressed {
		GlobalMetrics.TotalSavedBytes.Add(original - compressed)
	}
	GlobalMetrics.CompressionTime.Add(duration.Nanoseconds())
}

func RecordDecompression(duration time.Duration) {
	GlobalMetrics.TotalDecompressions.Add(1)
	GlobalMetrics.DecompressionTime.Add(duration.Nanoseconds())
}

func (m *Metrics) GetCompressionRatio() float64 {
	orig := m.TotalOriginalBytes.Load()
	if orig == 0 {
		return 1.0
	}
	saved := m.TotalSavedBytes.Load()
	return float64(orig) / float64(orig-saved)
}
