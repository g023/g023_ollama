package nn

import (
	"reflect"
	"sync"
	"testing"
)

// =============================================================================
// BENCHMARK TESTS FOR ATTENTION OPTIMIZATIONS
// =============================================================================
//
// This file provides comprehensive benchmarks for the optimized attention.go.
// Run with: go test -bench=. -benchmem -benchtime=5s ./ml/nn/
//
// Target hardware: NVIDIA RTX 3060 (12GB), Intel Xeon E5-1650, 64GB RAM
// =============================================================================

// =============================================================================
// TYPE FOR BENCHMARKING REFLECT OPERATIONS
// =============================================================================

type benchTensorType struct {
	dims []int
}

// =============================================================================
// BENCHMARK: SDPA Capability Check (Type Assertion Caching)
// =============================================================================

func BenchmarkReflectTypeOf(b *testing.B) {
	tensor := &benchTensorType{dims: []int{64, 8, 512}}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = reflect.TypeOf(tensor)
	}
}

func BenchmarkReflectTypeOfWithSyncMapLookup(b *testing.B) {
	tensor := &benchTensorType{dims: []int{64, 8, 512}}
	cache := sync.Map{}
	cache.Store(reflect.TypeOf(tensor), true)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		t := reflect.TypeOf(tensor)
		_, _ = cache.Load(t)
	}
}

func BenchmarkTypeAssertionDirect(b *testing.B) {
	var tensor interface{} = &benchTensorType{dims: []int{64, 8, 512}}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = tensor.(*benchTensorType)
	}
}

// =============================================================================
// BENCHMARK: Validation Mode Comparison
// =============================================================================

func BenchmarkValidationEnabled(b *testing.B) {
	SetAttentionValidationMode(ValidationEnabled)
	defer SetAttentionValidationMode(ValidationEnabled)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mode := GetAttentionValidationMode()
		_ = mode == ValidationEnabled
	}
}

func BenchmarkValidationDisabled(b *testing.B) {
	SetAttentionValidationMode(ValidationDisabled)
	defer SetAttentionValidationMode(ValidationEnabled)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		mode := GetAttentionValidationMode()
		_ = mode == ValidationDisabled
	}
}

// =============================================================================
// BENCHMARK: Atomic Operations
// =============================================================================

func BenchmarkAtomicLoadValidationMode(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = validationMode.Load()
	}
}

func BenchmarkAtomicStoreValidationMode(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		validationMode.Store(int32(ValidationEnabled))
	}
}

// =============================================================================
// BENCHMARK: sync.Map Operations
// =============================================================================

func BenchmarkSyncMapLoad(b *testing.B) {
	cache := sync.Map{}
	key := reflect.TypeOf(&benchTensorType{})
	cache.Store(key, true)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = cache.Load(key)
	}
}

func BenchmarkSyncMapStore(b *testing.B) {
	cache := sync.Map{}
	key := reflect.TypeOf(&benchTensorType{})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		cache.Store(key, true)
	}
}

func BenchmarkSyncMapLoadOrStore(b *testing.B) {
	cache := sync.Map{}
	key := reflect.TypeOf(&benchTensorType{})

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		cache.LoadOrStore(key, true)
	}
}

// =============================================================================
// BENCHMARK: Error String Allocation Comparison
// =============================================================================

func BenchmarkPreAllocatedErrorString(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = errDkMismatch
	}
}

// =============================================================================
// UNIT TESTS FOR OPTIMIZATION CORRECTNESS
// =============================================================================

func TestValidationModeDefault(t *testing.T) {
	// Reset to default
	SetAttentionValidationMode(ValidationEnabled)

	mode := GetAttentionValidationMode()
	if mode != ValidationEnabled {
		t.Errorf("expected ValidationEnabled, got %v", mode)
	}
}

func TestValidationModeDisabled(t *testing.T) {
	SetAttentionValidationMode(ValidationDisabled)
	defer SetAttentionValidationMode(ValidationEnabled)

	mode := GetAttentionValidationMode()
	if mode != ValidationDisabled {
		t.Errorf("expected ValidationDisabled, got %v", mode)
	}
}

func TestValidationModeOnce(t *testing.T) {
	SetAttentionValidationMode(ValidationOnce)
	defer SetAttentionValidationMode(ValidationEnabled)

	mode := GetAttentionValidationMode()
	if mode != ValidationOnce {
		t.Errorf("expected ValidationOnce, got %v", mode)
	}
}

func TestResetAttentionCaches(t *testing.T) {
	// Store something in caches
	testType := reflect.TypeOf(&benchTensorType{})
	sdpaCapabilityCache.Store(testType, true)
	validationOnceMap.Store(testType, true)

	// Reset
	ResetAttentionCaches()

	// Verify caches are cleared (by checking they're new empty maps)
	// Note: sync.Map doesn't have a Len() method, so we verify by checking
	// that a stored value is no longer there
	if _, exists := sdpaCapabilityCache.Load(testType); exists {
		t.Error("sdpaCapabilityCache should be empty after reset")
	}
	if _, exists := validationOnceMap.Load(testType); exists {
		t.Error("validationOnceMap should be empty after reset")
	}
}

func TestPreAllocatedErrorStrings(t *testing.T) {
	// Verify error strings are non-empty and correct
	if errDkMismatch == "" {
		t.Error("errDkMismatch should not be empty")
	}
	if errKvHeadsMismatch == "" {
		t.Error("errKvHeadsMismatch should not be empty")
	}
	if errSeqLenMismatch == "" {
		t.Error("errSeqLenMismatch should not be empty")
	}
	if errNilKV == "" {
		t.Error("errNilKV should not be empty")
	}

	// Verify they contain meaningful text
	if errDkMismatch != "d_k dimension mismatch between query and key" {
		t.Errorf("unexpected errDkMismatch: %s", errDkMismatch)
	}
}

func TestSDPACapabilityCacheStorage(t *testing.T) {
	ResetAttentionCaches()

	testType := reflect.TypeOf(&benchTensorType{})

	// Initially should not be cached
	if _, exists := sdpaCapabilityCache.Load(testType); exists {
		t.Error("cache should be empty initially")
	}

	// Store a value
	sdpaCapabilityCache.Store(testType, true)

	// Should be retrievable
	val, exists := sdpaCapabilityCache.Load(testType)
	if !exists {
		t.Error("value should exist after storing")
	}
	if !val.(bool) {
		t.Error("value should be true")
	}
}

// =============================================================================
// CONCURRENCY TESTS
// =============================================================================

func TestConcurrentValidationModeAccess(t *testing.T) {
	done := make(chan bool)
	iterations := 100

	// Writers
	for i := 0; i < 50; i++ {
		go func() {
			for j := 0; j < iterations; j++ {
				SetAttentionValidationMode(ValidationDisabled)
				SetAttentionValidationMode(ValidationEnabled)
			}
			done <- true
		}()
	}

	// Readers
	for i := 0; i < 50; i++ {
		go func() {
			for j := 0; j < iterations; j++ {
				_ = GetAttentionValidationMode()
			}
			done <- true
		}()
	}

	// Wait for all
	for i := 0; i < 100; i++ {
		<-done
	}
}

func TestConcurrentCacheAccess(t *testing.T) {
	ResetAttentionCaches()
	done := make(chan bool)
	iterations := 100
	testType := reflect.TypeOf(&benchTensorType{})

	// Writers
	for i := 0; i < 50; i++ {
		go func() {
			for j := 0; j < iterations; j++ {
				sdpaCapabilityCache.Store(testType, true)
			}
			done <- true
		}()
	}

	// Readers
	for i := 0; i < 50; i++ {
		go func() {
			for j := 0; j < iterations; j++ {
				_, _ = sdpaCapabilityCache.Load(testType)
			}
			done <- true
		}()
	}

	// Wait for all
	for i := 0; i < 100; i++ {
		<-done
	}
}

// =============================================================================
// COMPARATIVE BENCHMARKS: OLD vs NEW PATTERNS
// =============================================================================

// Simulates old validation pattern with fmt.Errorf
func BenchmarkOldValidationPattern(b *testing.B) {
	d1, d2 := 64, 64

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if d1 != d2 {
			// Old pattern would do: panic(fmt.Errorf("mismatch %v vs %v", d1, d2))
		}
		// Check passes, no allocation
	}
}

// Simulates new validation pattern with pre-allocated error
func BenchmarkNewValidationPattern(b *testing.B) {
	d1, d2 := 64, 64

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if d1 != d2 {
			panic(errDkMismatch)
		}
		// Check passes, no allocation
	}
}

// =============================================================================
// ADDITIONAL PERFORMANCE TESTS
// =============================================================================

func BenchmarkNilCheck(b *testing.B) {
	var ptr *benchTensorType = nil

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ptr != nil
	}
}

func BenchmarkCombinedNilCheck(b *testing.B) {
	var key, value *benchTensorType = &benchTensorType{}, &benchTensorType{}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = key != nil && value != nil
	}
}

// Benchmark comparing bool check overhead
func BenchmarkBoolCheck(b *testing.B) {
	hasKV := true

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		if hasKV {
			// Do something
		}
	}
}

