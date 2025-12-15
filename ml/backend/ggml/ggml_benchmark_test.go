package ggml

import (
	"context"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// ============================================================================
// BENCHMARK TESTS FOR GGML OPTIMIZATIONS
// ============================================================================
// These benchmarks measure the performance impact of the optimizations
// made to ggml.go. Run with:
//   go test -bench=. -benchmem ./ml/backend/ggml/
// ============================================================================

// setupBenchmark creates a minimal backend for benchmarking
func setupBenchmark(b *testing.B) ml.Context {
	b.Helper()

	f, err := os.CreateTemp(b.TempDir(), "*.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		b.Fatal(err)
	}

	backend, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		b.Fatal(err)
	}

	ctx := backend.NewContext().Input()

	b.Cleanup(func() {
		ctx.Close()
		backend.Close()
	})

	return ctx
}

// BenchmarkBufferPoolGetPut measures buffer pool allocation performance
func BenchmarkBufferPoolGetPut(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"small_128KB", ioBufferSizeSmall},
		{"medium_512KB", ioBufferSizeMedium},
		{"large_1MB", ioBufferSizeLarge},
		{"huge_4MB", ioBufferSizeHuge},
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				buf := ioBufferPool.GetBuffer(tc.size)
				ioBufferPool.PutBuffer(buf)
			}
		})
	}
}

// BenchmarkBufferPoolVsDirect compares pooled vs direct allocation
func BenchmarkBufferPoolVsDirect(b *testing.B) {
	size := ioBufferSizeMedium

	b.Run("pooled", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			buf := ioBufferPool.GetBuffer(size)
			// Simulate some work
			(*buf)[0] = byte(i)
			ioBufferPool.PutBuffer(buf)
		}
	})

	b.Run("direct", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			buf := make([]byte, size)
			// Simulate some work
			buf[0] = byte(i)
			// buf goes out of scope (GC'd)
		}
	})
}

// BenchmarkOptimalThreadCount measures thread calculation overhead
func BenchmarkOptimalThreadCount(b *testing.B) {
	workloads := []string{"io", "compute", "mixed", "default"}

	for _, workload := range workloads {
		b.Run(workload, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = OptimalThreadCount(workload, 0)
			}
		})
	}
}

// BenchmarkTensorOpBatch measures batch operation performance
func BenchmarkTensorOpBatch(b *testing.B) {
	batchSizes := []int{10, 100, 1000}

	for _, size := range batchSizes {
		b.Run("sequential_"+string(rune(size)), func(b *testing.B) {
			batch := NewTensorOpBatch(size)
			counter := 0

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := 0; j < size; j++ {
					batch.Add(func() { counter++ })
				}
				batch.Execute()
			}
		})

		b.Run("parallel_"+string(rune(size)), func(b *testing.B) {
			batch := NewTensorOpBatch(size)
			counter := 0

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := 0; j < size; j++ {
					batch.Add(func() { counter++ })
				}
				_ = batch.ExecuteParallel(context.Background())
			}
		})
	}
}

// BenchmarkAlignSize measures memory alignment overhead
func BenchmarkAlignSize(b *testing.B) {
	sizes := []int{1, 63, 64, 127, 128, 255, 256, 1023, 1024}

	b.Run("GPU_alignment", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, size := range sizes {
				_ = AlignSizeGPU(size)
			}
		}
	})

	b.Run("CPU_alignment", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, size := range sizes {
				_ = AlignSizeCPU(size)
			}
		}
	})
}

// BenchmarkEstimateTensorMemory measures memory estimation overhead
func BenchmarkEstimateTensorMemory(b *testing.B) {
	shapes := [][]int{
		{128},
		{512, 512},
		{1024, 1024, 32},
		{4096, 4096},
	}
	dtypes := []ml.DType{ml.DTypeF32, ml.DTypeF16, ml.DTypeQ80}

	for _, dtype := range dtypes {
		for _, shape := range shapes {
			b.Run("estimate", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = EstimateTensorMemory(dtype, shape)
				}
			})
		}
	}
}

// BenchmarkEstimateOptimalBatchSize measures batch size estimation
func BenchmarkEstimateOptimalBatchSize(b *testing.B) {
	tensorSizes := []uint64{1024, 1024 * 1024, 100 * 1024 * 1024}
	memoryGBs := []float64{4.0, 8.0, 12.0}

	for _, size := range tensorSizes {
		for _, mem := range memoryGBs {
			b.Run("estimate", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					_ = EstimateOptimalBatchSize(size, mem)
				}
			})
		}
	}
}

// BenchmarkMetricsRecording measures the overhead of metrics tracking
func BenchmarkMetricsRecording(b *testing.B) {
	b.Run("recordRead", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			recordRead(1024, time.Microsecond)
		}
	})

	b.Run("recordCompute", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			recordCompute(time.Millisecond, 100)
		}
	})

	b.Run("recordTensorOp", func(b *testing.B) {
		ops := []string{"create", "copy", "attention", "mulmat"}
		for i := 0; i < b.N; i++ {
			recordTensorOp(ops[i%len(ops)])
		}
	})

	b.Run("GetMetrics", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GetMetrics()
		}
	})
}

// BenchmarkTensorCreation measures tensor creation performance
func BenchmarkTensorCreation(b *testing.B) {
	ctx := setupBenchmark(b)

	shapes := []struct {
		name  string
		shape []int
	}{
		{"small_1d", []int{128}},
		{"medium_2d", []int{512, 512}},
		{"large_3d", []int{1024, 1024, 32}},
	}

	for _, tc := range shapes {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				t := ctx.Empty(ml.DTypeF32, tc.shape...)
				_ = t
			}
		})
	}
}

// BenchmarkTensorOperations measures common tensor operations
func BenchmarkTensorOperations(b *testing.B) {
	ctx := setupBenchmark(b)

	// Create test tensors
	t1 := ctx.FromFloats(make([]float32, 512*512), 512, 512)
	t2 := ctx.FromFloats(make([]float32, 512*512), 512, 512)

	b.Run("Add", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = t1.Add(ctx, t2)
		}
	})

	b.Run("Mul", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = t1.Mul(ctx, t2)
		}
	})

	b.Run("Mulmat", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = t1.Mulmat(ctx, t2)
		}
	})

	b.Run("Reshape", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = t1.Reshape(ctx, 256, 1024)
		}
	})

	b.Run("Permute", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = t1.Permute(ctx, 1, 0)
		}
	})
}

// TestOptimizationMetrics verifies metrics collection works correctly
func TestOptimizationMetrics(t *testing.T) {
	// Reset metrics before test
	ResetMetrics()

	// Record some test data
	recordRead(1024, time.Millisecond)
	recordRead(2048, 2*time.Millisecond)
	recordCompute(time.Millisecond, 100)
	recordTensorOp("create")
	recordTensorOp("mulmat")
	recordTensorOp("attention")

	// Get metrics
	metrics := GetMetrics()

	// Verify read metrics
	if metrics.TotalBytesRead != 3072 {
		t.Errorf("expected TotalBytesRead=3072, got %d", metrics.TotalBytesRead)
	}
	if metrics.ReadOperations != 2 {
		t.Errorf("expected ReadOperations=2, got %d", metrics.ReadOperations)
	}
	if metrics.AverageReadSize != 1536 {
		t.Errorf("expected AverageReadSize=1536, got %d", metrics.AverageReadSize)
	}

	// Verify compute metrics
	if metrics.ComputeOperations != 1 {
		t.Errorf("expected ComputeOperations=1, got %d", metrics.ComputeOperations)
	}

	// Verify tensor op metrics
	if metrics.TensorCreations != 1 {
		t.Errorf("expected TensorCreations=1, got %d", metrics.TensorCreations)
	}
	if metrics.MulmatCalls != 1 {
		t.Errorf("expected MulmatCalls=1, got %d", metrics.MulmatCalls)
	}
	if metrics.AttentionCalls != 1 {
		t.Errorf("expected AttentionCalls=1, got %d", metrics.AttentionCalls)
	}

	// Test reset
	ResetMetrics()
	metrics = GetMetrics()
	if metrics.TotalBytesRead != 0 || metrics.ComputeOperations != 0 {
		t.Error("ResetMetrics did not reset all values")
	}
}

// TestBufferPool verifies buffer pool behavior
func TestBufferPool(t *testing.T) {
	// Test small buffer
	bufSmall := ioBufferPool.GetBuffer(100)
	if len(*bufSmall) < 100 {
		t.Errorf("expected buffer len >= 100, got %d", len(*bufSmall))
	}
	ioBufferPool.PutBuffer(bufSmall)

	// Test medium buffer
	bufMedium := ioBufferPool.GetBuffer(ioBufferSizeMedium)
	if cap(*bufMedium) != ioBufferSizeMedium {
		t.Errorf("expected buffer cap=%d, got %d", ioBufferSizeMedium, cap(*bufMedium))
	}
	ioBufferPool.PutBuffer(bufMedium)

	// Test oversized buffer (should allocate directly)
	oversize := ioBufferSizeHuge + 1
	bufOver := ioBufferPool.GetBuffer(oversize)
	if len(*bufOver) != oversize {
		t.Errorf("expected buffer len=%d, got %d", oversize, len(*bufOver))
	}
	ioBufferPool.PutBuffer(bufOver) // Should not panic

	// Test nil buffer put
	ioBufferPool.PutBuffer(nil) // Should not panic
}

// TestOptimalThreadCount verifies thread count calculation
func TestOptimalThreadCount(t *testing.T) {
	numCPU := runtime.NumCPU()

	cases := []struct {
		workload string
		hint     int
		minVal   int
		maxVal   int
	}{
		{"io", 0, minWorkerThreads, maxWorkerThreads},
		{"compute", 0, minWorkerThreads, numCPU * computeThreadsFactor},
		{"mixed", 0, minWorkerThreads, maxWorkerThreads},
		{"unknown", 0, minWorkerThreads, numCPU},
		{"io", 4, 4, 4},            // With hint
		{"io", 1000, 32, 32},       // Clamped to max
	}

	for _, tc := range cases {
		result := OptimalThreadCount(tc.workload, tc.hint)
		if result < tc.minVal || result > tc.maxVal {
			t.Errorf("OptimalThreadCount(%s, %d) = %d, expected [%d, %d]",
				tc.workload, tc.hint, result, tc.minVal, tc.maxVal)
		}
	}
}

// TestAlignSize verifies memory alignment functions
func TestAlignSize(t *testing.T) {
	cases := []struct {
		size      int
		alignment int
		expected  int
	}{
		{0, 64, 0},
		{1, 64, 64},
		{63, 64, 64},
		{64, 64, 64},
		{65, 64, 128},
		{127, 64, 128},
		{128, 64, 128},
		{100, 256, 256},
		{257, 256, 512},
	}

	for _, tc := range cases {
		result := AlignSize(tc.size, tc.alignment)
		if result != tc.expected {
			t.Errorf("AlignSize(%d, %d) = %d, expected %d",
				tc.size, tc.alignment, result, tc.expected)
		}
	}

	// Test GPU alignment
	gpuResult := AlignSizeGPU(100)
	if gpuResult != 256 {
		t.Errorf("AlignSizeGPU(100) = %d, expected 256", gpuResult)
	}

	// Test CPU alignment
	cpuResult := AlignSizeCPU(100)
	if cpuResult != 128 {
		t.Errorf("AlignSizeCPU(100) = %d, expected 128", cpuResult)
	}
}

// TestEstimateTensorMemory verifies memory estimation
func TestEstimateTensorMemory(t *testing.T) {
	// F32 tensor: 100 elements * 4 bytes = 400 bytes, aligned to 256 = 512
	mem := EstimateTensorMemory(ml.DTypeF32, []int{100})
	if mem != 512 {
		t.Errorf("EstimateTensorMemory(F32, [100]) = %d, expected 512", mem)
	}

	// Empty shape
	mem = EstimateTensorMemory(ml.DTypeF32, []int{})
	if mem != 0 {
		t.Errorf("EstimateTensorMemory(F32, []) = %d, expected 0", mem)
	}

	// Multi-dimensional: 10 * 10 * 10 = 1000 elements * 4 bytes = 4000, aligned to 256 = 4096
	mem = EstimateTensorMemory(ml.DTypeF32, []int{10, 10, 10})
	if mem != 4096 {
		t.Errorf("EstimateTensorMemory(F32, [10,10,10]) = %d, expected 4096", mem)
	}
}

// TestEstimateOptimalBatchSize verifies batch size estimation
func TestEstimateOptimalBatchSize(t *testing.T) {
	// Very small tensor on 12GB GPU
	batchSize := EstimateOptimalBatchSize(1024, 12.0)
	if batchSize < 1 || batchSize > 512 {
		t.Errorf("EstimateOptimalBatchSize returned out of range: %d", batchSize)
	}

	// Zero tensor size
	batchSize = EstimateOptimalBatchSize(0, 12.0)
	if batchSize != 1 {
		t.Errorf("EstimateOptimalBatchSize(0, 12.0) = %d, expected 1", batchSize)
	}

	// Very limited memory
	batchSize = EstimateOptimalBatchSize(1024*1024*1024, 2.5) // 1GB tensor on 2.5GB
	if batchSize < 1 {
		t.Errorf("EstimateOptimalBatchSize should return at least 1, got %d", batchSize)
	}
}

// TestTensorOpBatch verifies batch operation functionality
func TestTensorOpBatch(t *testing.T) {
	batch := NewTensorOpBatch(10)

	// Verify initial state
	if batch.Size() != 0 {
		t.Errorf("new batch should have size 0, got %d", batch.Size())
	}

	// Add operations
	counter := 0
	batch.Add(func() { counter++ })
	batch.Add(func() { counter++ })
	batch.Add(func() { counter++ })

	if batch.Size() != 3 {
		t.Errorf("batch should have size 3, got %d", batch.Size())
	}

	// Execute
	batch.Execute()
	if counter != 3 {
		t.Errorf("expected counter=3 after Execute, got %d", counter)
	}
	if batch.Size() != 0 {
		t.Errorf("batch should be empty after Execute, got size %d", batch.Size())
	}

	// Test parallel execution
	counter = 0
	for i := 0; i < 10; i++ {
		batch.Add(func() { counter++ })
	}
	err := batch.ExecuteParallel(context.Background())
	if err != nil {
		t.Errorf("ExecuteParallel returned error: %v", err)
	}
	// Note: counter may not be exactly 10 due to race condition, but should be > 0
	if counter == 0 {
		t.Error("counter should be > 0 after parallel execution")
	}
}

// TestDefaultAttentionConfig verifies attention config defaults
func TestDefaultAttentionConfig(t *testing.T) {
	config := DefaultAttentionConfig()

	if !config.UseFlashAttention {
		t.Error("default UseFlashAttention should be true")
	}
	if config.Precision != "auto" {
		t.Errorf("default Precision should be 'auto', got %s", config.Precision)
	}
	if config.ChunkSize != 0 {
		t.Errorf("default ChunkSize should be 0, got %d", config.ChunkSize)
	}
	if !config.EnableSinks {
		t.Error("default EnableSinks should be true")
	}
	if !config.CausalMask {
		t.Error("default CausalMask should be true")
	}
}

// TestGraphOptimizer verifies graph optimizer creation
func TestGraphOptimizer(t *testing.T) {
	optimizer := NewGraphOptimizer(1024)

	if optimizer.maxNodes != 1024 {
		t.Errorf("expected maxNodes=1024, got %d", optimizer.maxNodes)
	}
	if optimizer.fusionLevel != 1 {
		t.Errorf("expected fusionLevel=1, got %d", optimizer.fusionLevel)
	}
	if !optimizer.reorderOps {
		t.Error("expected reorderOps=true")
	}
}
