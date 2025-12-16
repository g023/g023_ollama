package kvcache

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// Benchmark tests for optimized causal cache
// These demonstrate the performance improvements

func BenchmarkStartForward(b *testing.B) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 4, 2048, 32)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Prime the cache with some data
	batch := input.Batch{
		Positions: []int32{0, 1, 2, 3},
		Sequences: []int{0, 0, 0, 0},
	}
	cache.StartForward(ctx, batch, false)
	cache.SetLayer(0)
	tensor := ctx.FromFloats([]float32{1, 2, 3, 4}, 1, 1, 4)
	cache.Put(ctx, tensor, tensor)

	// Benchmark subsequent forward passes
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i + 4)},
			Sequences: []int{0},
		}
		cache.StartForward(ctx, batch, false)
	}
}

func BenchmarkBuildMaskSmall(b *testing.B) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	batch := input.Batch{
		Positions: []int32{0, 1, 2, 3},
		Sequences: []int{0, 0, 0, 0},
	}
	cache.StartForward(ctx, batch, false)
	cache.SetLayer(0)
	tensor := ctx.FromFloats([]float32{1, 2, 3, 4}, 1, 1, 4)
	cache.Put(ctx, tensor, tensor)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.buildMask(ctx)
	}
}

func BenchmarkBuildMaskLarge(b *testing.B) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 1, 2048, 32)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Fill cache with positions
	positions := make([]int32, 32)
	sequences := make([]int, 32)
	for i := range positions {
		positions[i] = int32(i)
		sequences[i] = 0
	}

	batch := input.Batch{
		Positions: positions,
		Sequences: sequences,
	}
	cache.StartForward(ctx, batch, false)
	cache.SetLayer(0)
	tensor := ctx.FromFloats(make([]float32, 32), 1, 1, 32)
	cache.Put(ctx, tensor, tensor)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.buildMask(ctx)
	}
}

func BenchmarkSeqBitmapOperations(b *testing.B) {
	b.Run("set", func(b *testing.B) {
		var bitmap seqBitmap
		for i := 0; i < b.N; i++ {
			bitmap.set(i % 64)
		}
	})

	b.Run("has", func(b *testing.B) {
		var bitmap seqBitmap
		bitmap.set(5)
		bitmap.set(10)
		bitmap.set(20)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bitmap.has(10)
		}
	})

	b.Run("hasOtherThan", func(b *testing.B) {
		var bitmap seqBitmap
		bitmap.set(5)
		bitmap.set(10)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bitmap.hasOtherThan(5)
		}
	})
}

func BenchmarkCopyPrefix(b *testing.B) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()

	cache.Init(backend, ml.DTypeF16, 4, 256, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	// Create a sequence with some data
	positions := make([]int32, 16)
	sequences := make([]int, 16)
	for i := range positions {
		positions[i] = int32(i)
		sequences[i] = 0
	}
	batch := input.Batch{Positions: positions, Sequences: sequences}
	cache.StartForward(ctx, batch, false)
	cache.SetLayer(0)
	tensor := ctx.FromFloats(make([]float32, 16), 1, 1, 16)
	cache.Put(ctx, tensor, tensor)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.CopyPrefix(0, 1, 8)
	}
}
