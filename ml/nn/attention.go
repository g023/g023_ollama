package nn

import (
	"reflect"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
)

// =============================================================================
// ATTENTION OPTIMIZATION CONFIGURATION
// =============================================================================

// AttentionValidationMode controls dimension validation behavior.
// Disable after warmup in production for maximum performance.
type AttentionValidationMode int32

const (
	// ValidationEnabled performs full dimension checks (default, safe)
	ValidationEnabled AttentionValidationMode = iota
	// ValidationDisabled skips dimension checks (production mode after warmup)
	ValidationDisabled
	// ValidationOnce validates first call only, then auto-disables
	ValidationOnce
)

var (
	// validationMode controls whether dimension validation is performed.
	// Use SetAttentionValidationMode() to change. Default: ValidationEnabled
	validationMode atomic.Int32

	// validationOncePerType tracks first-call validation per tensor type
	validationOnceMap sync.Map // map[reflect.Type]bool

	// sdpaCapabilityCache caches SDPA interface check results per type
	// Eliminates repeated type assertions in hot inference loops
	sdpaCapabilityCache sync.Map // map[reflect.Type]bool
)

// Pre-allocated error messages to avoid allocations in hot path
var (
	errDkMismatch      = "d_k dimension mismatch between query and key"
	errKvHeadsMismatch = "kv_heads dimension mismatch between key and value"
	errSeqLenMismatch  = "seq_len_k dimension mismatch between key and value"
	errNilKV           = "key & value tensors must be provided if cache is nil"
)

// SetAttentionValidationMode configures dimension validation behavior.
// Call with ValidationDisabled after model warmup for production inference.
//
//go:noinline
func SetAttentionValidationMode(mode AttentionValidationMode) {
	validationMode.Store(int32(mode))
}

// GetAttentionValidationMode returns the current validation mode.
//
//go:noinline
func GetAttentionValidationMode() AttentionValidationMode {
	return AttentionValidationMode(validationMode.Load())
}

// ResetAttentionCaches clears all internal caches.
// Useful for testing or when tensor implementations change.
//
//go:noinline
func ResetAttentionCaches() {
	validationOnceMap = sync.Map{}
	sdpaCapabilityCache = sync.Map{}
}

// =============================================================================
// INTERNAL OPTIMIZATION HELPERS
// =============================================================================

// checkSDPACapability performs cached SDPA interface check.
// Uses sync.Map for thread-safe caching across concurrent inference.
//
//go:inline
func checkSDPACapability(query ml.Tensor) (ml.ScaledDotProductAttention, bool) {
	// Fast path: check type cache first
	queryType := reflect.TypeOf(query)
	if cached, ok := sdpaCapabilityCache.Load(queryType); ok {
		if cached.(bool) {
			// Type supports SDPA, do the assertion (can't cache interface value)
			sdpa, _ := query.(ml.ScaledDotProductAttention)
			return sdpa, true
		}
		return nil, false
	}

	// Slow path: first time for this type, check and cache result
	sdpa, ok := query.(ml.ScaledDotProductAttention)
	sdpaCapabilityCache.Store(queryType, ok)
	return sdpa, ok
}

// validateDimensions checks tensor dimension compatibility.
// Panics with pre-allocated error messages to minimize allocations.
//
//go:inline
func validateDimensions(query, key, value ml.Tensor) {
	// Ordered for most likely failure first (d_k usually wrong if any)
	if query.Dim(0) != key.Dim(0) {
		panic(errDkMismatch)
	}
	if key.Dim(1) != value.Dim(1) {
		panic(errKvHeadsMismatch)
	}
	if key.Dim(2) != value.Dim(2) {
		panic(errSeqLenMismatch)
	}
}

// shouldValidate determines if validation should be performed based on mode.
//
//go:inline
func shouldValidate(query ml.Tensor) bool {
	mode := AttentionValidationMode(validationMode.Load())
	switch mode {
	case ValidationDisabled:
		return false
	case ValidationOnce:
		queryType := reflect.TypeOf(query)
		if _, validated := validationOnceMap.Load(queryType); validated {
			return false
		}
		validationOnceMap.Store(queryType, true)
		return true
	default: // ValidationEnabled
		return true
	}
}

// =============================================================================
// PUBLIC ATTENTION API
// =============================================================================

// Attention implements scaled dot-product attention for transformer models:
// Attention(Q, K, V) = softmax(QK^T/√d_k)V
//
// This implementation is optimized for inference with:
//   - Cached SDPA interface checks to eliminate repeated type assertions
//   - Configurable dimension validation (disable after warmup for production)
//   - Pre-allocated error messages to minimize allocations
//   - Optimized control flow for branch prediction
//
// Parameters:
//   - ctx: Context for tensor operations
//   - query: Query tensor (Q) with shape [d_k, heads, seq_len_q]
//   - key: Key tensor (K) with shape [d_k, kv_heads, seq_len_k], can be nil to read from cache only
//   - value: Value tensor (V) with shape [d_v, kv_heads, seq_len_k], can be nil to read from cache only
//   - scale: Scaling factor, typically 1/√d_k where d_k is the key dimension
//   - cache: KV cache to store key/value and get past history, can be nil to only use provided key/value
//
// Returns:
//
//	Attention output with shape [d_v, heads, seq_len_q]
//
//go:noinline
func Attention(ctx ml.Context, query, key, value ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	// Direct call to core function - no intermediate wrapper overhead
	return attentionCore(ctx, query, key, value, nil, nil, scale, cache)
}

// AttentionWithSinks implements attention with sink tokens for streaming/infinite context.
//
//go:noinline
func AttentionWithSinks(ctx ml.Context, query, key, value, sinks ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return attentionCore(ctx, query, key, value, sinks, nil, scale, cache)
}

// AttentionWithVMLA implements attention with VMLA (Value Matrix Linear Attention) projection.
// Used for certain architectures like DeepSeek2 that apply an additional linear transformation.
//
//go:noinline
func AttentionWithVMLA(ctx ml.Context, query, key, value, sinks ml.Tensor, vmla ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	return attentionCore(ctx, query, key, value, sinks, vmla, scale, cache)
}

// =============================================================================
// CORE ATTENTION IMPLEMENTATION
// =============================================================================

// attentionCore is the unified, optimized attention implementation.
// All public functions delegate here to ensure consistent optimization.
//
//go:noinline
func attentionCore(ctx ml.Context, query, key, value, sinks, vmla ml.Tensor, scale float64, cache kvcache.Cache) ml.Tensor {
	// ==========================================================================
	// PHASE 1: Input Processing & Cache Update
	// ==========================================================================

	// Register query for computation graph
	ctx.Forward(query)

	// Handle key/value input - optimized branch ordering
	hasKV := key != nil && value != nil
	if hasKV {
		// Validation (when enabled)
		if shouldValidate(query) {
			validateDimensions(query, key, value)
		}

		// Register K/V for computation and update cache
		ctx.Forward(key, value)
		if cache != nil {
			cache.Put(ctx, key, value)
		}
	} else if cache == nil {
		panic(errNilKV)
	}

	// ==========================================================================
	// PHASE 2: Cache Retrieval
	// ==========================================================================

	var mask ml.Tensor
	if cache != nil {
		key, value, mask = cache.Get(ctx)
	}

	// ==========================================================================
	// PHASE 3: Attention Computation (SDPA or Manual Path)
	// ==========================================================================

	// Check for hardware-accelerated SDPA support (cached check)
	if sdpa, ok := checkSDPACapability(query); ok {
		// Fast path: hardware-accelerated attention
		cacheConfigApplied := cache != nil
		return sdpa.ScaledDotProductAttention(ctx, key, value, mask, sinks, vmla, scale, cacheConfigApplied)
	}

	// Fallback path: manual attention computation
	return computeManualAttention(ctx, query, key, value, mask, vmla, scale)
}

// computeManualAttention performs the attention computation using individual tensor ops.
// This path is taken when the tensor type doesn't support fused SDPA.
// Optimized for minimal intermediate allocations and efficient operation ordering.
//
//go:noinline
func computeManualAttention(ctx ml.Context, query, key, value, mask, vmla ml.Tensor, scale float64) ml.Tensor {
	// Permute Q and K for batch matrix multiplication
	// Shape: [d_k, heads, seq_len] -> [d_k, seq_len, heads, batch]
	q := query.Permute(ctx, 0, 2, 1, 3)
	k := key.Permute(ctx, 0, 2, 1, 3)

	// Value needs contiguous memory after permute for efficient matmul
	// Shape: [d_v, kv_heads, seq_len] -> [seq_len, kv_heads, d_v, batch]
	v := value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	// Compute attention scores: QK^T with full precision for numerical stability
	kq := k.MulmatFullPrec(ctx, q)

	// Apply scaling factor: 1/√d_k
	kq = kq.Scale(ctx, scale)

	// Apply attention mask if present (e.g., causal mask)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}

	// Softmax to get attention weights
	kq = kq.Softmax(ctx)

	// Weighted sum of values
	kqv := v.Mulmat(ctx, kq)

	// Apply VMLA projection if present (DeepSeek2 architecture)
	if vmla != nil {
		kqv = vmla.Mulmat(ctx, kqv)
	}

	// Final permute back to expected output shape and make contiguous
	return kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
}
