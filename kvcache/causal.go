// Package kvcache provides optimized key-value caching for transformer attention.
//
// # Causal Cache Architecture
//
// The Causal cache stores Key and Value tensors according to their position in
// sequences, enabling efficient autoregressive generation. This implementation
// uses several optimizations for high performance:
//
// ## Key Optimizations
//
//   - Bitmap-based sequence membership: O(1) vs O(n) for sequence checks
//   - Free list for cell allocation: O(1) vs O(n) for finding empty cells
//   - Scratch buffer reuse: Eliminates mask allocation per forward pass
//   - Pre-computed except masks: O(1) exception checking in mask building
//
// ## Supported Attention Patterns
//
//   - Standard causal attention (full history)
//   - Sliding window attention (SWA) with configurable window size
//   - SWA with extended memory for prefix caching
//   - Chunked attention for efficient long-sequence processing
//
// ## Performance Characteristics
//
//   - Cell allocation: O(batch_size) instead of O(cache_size)
//   - Mask building: O(batch_size * history_length) with O(1) sequence checks
//   - Sliding window update: O(affected_cells) instead of O(all_cells * sequences)
//   - Memory per cell: 12 bytes (bitmap) vs 28+ bytes (slice)
//
// ## Limitations
//
//   - Maximum 64 concurrent sequences (using uint64 bitmap)
//   - Not thread-safe (caller must synchronize)
package kvcache

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"sync"

	"github.com/ollama/ollama/kvcache/wavelet"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/envconfig"
)

// ============================================================================
// SEQUENCE BITMAP - O(1) sequence membership operations
// ============================================================================

// seqBitmap provides O(1) sequence membership testing using a 64-bit bitmap.
// Supports up to 64 sequences (0-63). Sequences >= 64 will panic.
type seqBitmap uint64

// set adds a sequence to the bitmap
func (s *seqBitmap) set(seq int) {
	if seq < 0 || seq >= 64 {
		panic(fmt.Errorf("sequence %d out of range [0, 63]", seq))
	}
	*s |= 1 << uint(seq)
}

// clear removes a sequence from the bitmap
func (s *seqBitmap) clear(seq int) {
	if seq < 0 || seq >= 64 {
		panic(fmt.Errorf("sequence %d out of range [0, 63]", seq))
	}
	*s &^= 1 << uint(seq)
}

// has returns true if the sequence is in the bitmap
func (s seqBitmap) has(seq int) bool {
	if seq < 0 || seq >= 64 {
		return false
	}
	return (s & (1 << uint(seq))) != 0
}

// isEmpty returns true if no sequences are in the bitmap
func (s seqBitmap) isEmpty() bool {
	return s == 0
}

// count returns the number of sequences in the bitmap
func (s seqBitmap) count() int {
	return bits.OnesCount64(uint64(s))
}

// hasOtherThan returns true if there's any sequence other than the given one
func (s seqBitmap) hasOtherThan(seq int) bool {
	if seq < 0 || seq >= 64 {
		return s != 0
	}
	mask := ^(seqBitmap(1) << uint(seq))
	return (s & mask) != 0
}

type shiftFn func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error)

// Causal cache stores K and V tensors according to their position in the
// sequence. Returns the history and a mask for attending to past tokens
//
// The tensors are of shape embed dim, kv heads, batch size
// The mask is of shape history size, batch size
type Causal struct {
	DType ml.DType

	// swaWindowSize is the number of tokens that will be included in the mask
	// during attention operations. swaMemorySize is the number of tokens that
	// will be retained in memory for partial prefix caching. Set to math.MaxInt32
	// for unlimited or if sliding window attention is not being used.
	swaWindowSize int32
	swaMemorySize int32

	chunkSize int32

	opts CausalOptions

	// maxBatch is the largest batch that we might receive
	maxBatch int

	// config controls mostly backend-specific optimizations
	config *ml.CacheConfig

	// ** current forward pass **

	// size of the current batch
	curBatchSize int

	// locations for data storage for this batch
	curLoc ml.Tensor

	// mask of the cache as used by this batch
	curMask ml.Tensor

	// the active layer for Get and Put
	curLayer int

	// locations in the cache that are needed for this batch
	curCellRange cellRange

	// curSequences is the sequences corresponding to this pass's entries in the cache
	curSequences []int

	// curPositions is the positions corresponding to this pass's entries in the cache
	curPositions []int32

	// ** cache metadata **

	// for each possible location in the cache, stores the position and set of sequences
	// that reference the data there
	cells []cacheCell

	// maps from sequence to the range of locations where it is stored in the cache
	cellRanges map[int]cellRange

	// ** optimized free list for O(1) cell allocation **
	freeList  []int32 // Stack of free cell indices
	freeCount int     // Number of free cells available

	// ** pre-allocated scratch buffers to reduce allocations **
	scratchMask      []float32              // Reusable mask buffer
	scratchLowestPos map[int]lowestPosition // Reusable map for sliding window

	// ** cache data storage **

	shiftFn      shiftFn
	backend      ml.Backend
	ctxs         map[int]ml.Context
	keys, values map[int]ml.Tensor

	// ** compression fields **
	compressionEnabled   bool
	compressionConfig    *wavelet.CodecConfig
	compressionThreshold int32 // Compress cells older than this position offset
	compressionMu        sync.RWMutex
	compressedSegments   map[int]map[int]*compressedSegment // layer -> cellIndex -> segment
	attentionScores      map[int][]float32                  // layer -> scores
}

type compressedSegment struct {
	coeffs *wavelet.WaveletCoefficients
}

// cacheCell stores position and sequence information for a single cache location.
// Uses bitmap for O(1) sequence operations instead of []int slice.
type cacheCell struct {
	pos        int32
	seqMask    seqBitmap // Bitmap of sequences that reference this cell
	compressed bool      // Is this cell in compressed form?
}

type cellRange struct {
	min int
	max int
}

// lowestPosition tracks the lowest position seen for a sequence
type lowestPosition struct {
	pos      int32
	curBatch bool
}

func NewCausalCache(shift shiftFn) *Causal {
	return &Causal{
		shiftFn:            shift,
		ctxs:               make(map[int]ml.Context),
		keys:               make(map[int]ml.Tensor),
		values:             make(map[int]ml.Tensor),
		scratchLowestPos:   make(map[int]lowestPosition),
		compressedSegments: make(map[int]map[int]*compressedSegment),
		attentionScores:    make(map[int][]float32),
	}
}

func NewSWACache(windowSize int32, shift shiftFn) *Causal {
	return &Causal{
		swaWindowSize:      windowSize,
		shiftFn:            shift,
		ctxs:               make(map[int]ml.Context),
		keys:               make(map[int]ml.Tensor),
		values:             make(map[int]ml.Tensor),
		scratchLowestPos:   make(map[int]lowestPosition),
		compressedSegments: make(map[int]map[int]*compressedSegment),
		attentionScores:    make(map[int][]float32),
	}
}

func NewSWAMemCache(windowSize int32, memorySize int32, shift shiftFn) *Causal {
	return &Causal{
		swaWindowSize:      windowSize,
		swaMemorySize:      memorySize,
		shiftFn:            shift,
		ctxs:               make(map[int]ml.Context),
		keys:               make(map[int]ml.Tensor),
		values:             make(map[int]ml.Tensor),
		scratchLowestPos:   make(map[int]lowestPosition),
		compressedSegments: make(map[int]map[int]*compressedSegment),
		attentionScores:    make(map[int][]float32),
	}
}

func NewChunkedAttentionCache(chunkSize int32, shift shiftFn) *Causal {
	return &Causal{
		chunkSize:          chunkSize,
		shiftFn:            shift,
		ctxs:               make(map[int]ml.Context),
		keys:               make(map[int]ml.Tensor),
		values:             make(map[int]ml.Tensor),
		scratchLowestPos:   make(map[int]lowestPosition),
		compressedSegments: make(map[int]map[int]*compressedSegment),
		attentionScores:    make(map[int][]float32),
	}
}

func (c *Causal) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	// Validate sequence limit for bitmap
	if maxSequences > 64 {
		panic(fmt.Errorf("maxSequences (%d) exceeds bitmap capacity (64)", maxSequences))
	}

	if c.config == nil {
		var config ml.CacheConfig
		if cc, ok := backend.(ml.BackendCacheConfig); ok {
			config = cc.CacheConfig()
		}
		c.config = &config
	}

	if c.config.CachePadding == 0 {
		c.config.CachePadding = 1
	}

	if c.config.MaskBatchPadding == 0 {
		c.config.MaskBatchPadding = 1
	}

	if c.config.MaskDType == ml.DTypeOther {
		c.config.MaskDType = ml.DTypeF32
	}

	if c.swaWindowSize == 0 {
		c.swaWindowSize = math.MaxInt32
	}
	if c.swaMemorySize == 0 {
		c.swaMemorySize = c.swaWindowSize
	}
	// We will allocate space in the cache for the stop token, which won't be part of a follow on
	// sequence, so allocate an extra token of storage to ensure that we can jump back without
	// causing a cache break. As an optimization, only do this when we have parallel sequences
	// because the extra token will live in the batch buffer and won't get overwritten if we
	// only have a single sequence.
	if c.swaMemorySize != math.MaxInt32 && maxSequences > 1 {
		c.swaMemorySize = max(c.swaMemorySize, c.swaWindowSize+1)
	}
	if int(c.swaMemorySize) >= capacity {
		c.swaMemorySize = math.MaxInt32
	}

	if c.swaMemorySize < c.swaWindowSize {
		panic(fmt.Errorf("sliding window memory (%v) must be at least as large as the window (%v)", c.swaMemorySize, c.swaWindowSize))
	}

	var cacheSize int
	if c.swaMemorySize == math.MaxInt32 {
		cacheSize = maxSequences * capacity
	} else {
		cacheSize = (maxSequences * int(c.swaMemorySize)) + maxBatch
	}
	cacheSize = roundUp(cacheSize, c.config.CachePadding)
	c.cells = make([]cacheCell, cacheSize)

	// Initialize free list - all cells start as free
	c.freeList = make([]int32, cacheSize)
	for i := range c.freeList {
		c.freeList[i] = int32(i)
	}
	c.freeCount = cacheSize

	// Pre-allocate scratch mask buffer
	maxMaskSize := roundUp(maxBatch, c.config.MaskBatchPadding) * cacheSize
	c.scratchMask = make([]float32, maxMaskSize)

	c.DType = dtype
	c.cellRanges = make(map[int]cellRange)
	c.backend = backend
	c.maxBatch = maxBatch

	// Initialize compression from environment config
	c.compressionEnabled = envconfig.KvCacheCompression()
	if c.compressionEnabled {
		c.compressionThreshold = int32(envconfig.KvCompressionThreshold())
		c.compressionConfig = &wavelet.CodecConfig{
			Levels:    int(envconfig.KvCompressionLevel()),
			Threshold: 0.01, // Default threshold
			Strategy:  wavelet.ThresholdAbsolute,
		}
	}
}

func (c *Causal) SetConfig(config ml.CacheConfig) {
	if c.config != nil {
		panic("config cannot be changed after being previously set, either by the model or backend")
	}

	c.config = &config
}

func (c *Causal) Close() {
	for _, ctx := range c.ctxs {
		ctx.Close()
	}
}

// ============================================================================
// FREE LIST MANAGEMENT - O(1) cell allocation
// ============================================================================

// allocateCells returns n free cell indices. Returns error if not enough free cells.
// This is O(k) where k is the number of cells requested, not O(n) total cells.
// Cells are allocated from the beginning of the free list to maintain ordering
// that the original linear scan provided (lower indices first).
func (c *Causal) allocateCells(n int) ([]int32, error) {
	if c.freeCount < n {
		return nil, fmt.Errorf("%w (cache: %v batch: %v free: %v)", ErrKvCacheFull, len(c.cells), n, c.freeCount)
	}

	// Pop n cells from the beginning of free list (to maintain low-to-high ordering)
	locs := make([]int32, n)
	copy(locs, c.freeList[:n])
	
	// Shift remaining free list entries down
	copy(c.freeList, c.freeList[n:c.freeCount])
	c.freeCount -= n

	return locs, nil
}

// freeCell returns a cell to the free list
// Cells are added to maintain sorted order for consistent allocation patterns
func (c *Causal) freeCell(idx int) {
	if c.freeCount >= len(c.freeList) {
		return // Already at capacity (shouldn't happen)
	}
	
	// Insert in sorted order to maintain consistent allocation
	// Find insertion point using binary search
	cellIdx := int32(idx)
	insertPos := 0
	for insertPos < c.freeCount && c.freeList[insertPos] < cellIdx {
		insertPos++
	}
	
	// Shift elements to make room
	copy(c.freeList[insertPos+1:c.freeCount+1], c.freeList[insertPos:c.freeCount])
	c.freeList[insertPos] = cellIdx
	c.freeCount++
}

// ============================================================================
// FORWARD PASS
// ============================================================================

func (c *Causal) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	c.curBatchSize = len(batch.Positions)
	c.curSequences = batch.Sequences
	c.curPositions = batch.Positions
	c.opts.Except = nil

	var locs []int32
	if !reserve {
		c.updateSlidingWindow()

		var err error
		locs, err = c.allocateCells(c.curBatchSize)
		if err != nil {
			return err
		}

		for i, pos := range batch.Positions {
			seq := batch.Sequences[i]
			loc := int(locs[i])

			// Initialize cell with bitmap
			c.cells[loc].pos = pos
			c.cells[loc].seqMask = 0
			c.cells[loc].seqMask.set(seq)

			seqRange, ok := c.cellRanges[seq]
			if !ok {
				seqRange = newRange()
			}

			seqRange.min = min(seqRange.min, loc)
			c.curCellRange.min = min(c.curCellRange.min, loc)

			seqRange.max = max(seqRange.max, loc)
			c.curCellRange.max = max(c.curCellRange.max, loc)

			c.cellRanges[seq] = seqRange
		}
	} else {
		// If we are reserving memory, don't update any of the cache metadata but set the size
		// to the worst case.
		locs = make([]int32, c.curBatchSize)
		for i := range locs {
			locs[i] = int32(i)
		}
		c.curCellRange.min = 0
		c.curCellRange.max = len(c.cells) - 1
	}

	c.curLoc = ctx.Input().FromInts(locs, len(locs))
	c.curMask = c.buildMask(ctx)

	return nil
}

func newRange() cellRange {
	return cellRange{
		min: math.MaxInt,
		max: 0,
	}
}

// ============================================================================
// SLIDING WINDOW UPDATE - Optimized with bitmaps
// ============================================================================

func (c *Causal) updateSlidingWindow() {
	c.curCellRange = newRange()

	if c.swaMemorySize == math.MaxInt32 {
		for _, seq := range c.curSequences {
			if seqRange, ok := c.cellRanges[seq]; ok {
				c.curCellRange.min = min(c.curCellRange.min, seqRange.min)
				c.curCellRange.max = max(c.curCellRange.max, seqRange.max)
			}
		}
		return
	}

	// Clear and reuse scratch map to avoid allocation
	for k := range c.scratchLowestPos {
		delete(c.scratchLowestPos, k)
	}
	lowestPos := c.scratchLowestPos

	// Create a map of unique sequences to the lowest position in that sequence
	for i := range c.curPositions {
		seq := c.curSequences[i]

		lowest, ok := lowestPos[seq]
		if !ok {
			lowest = lowestPosition{pos: c.curPositions[i], curBatch: true}
		} else if c.curPositions[i] < lowest.pos {
			lowest.pos = c.curPositions[i]
		}

		lowestPos[seq] = lowest
	}

	// For any sequences not part of this batch, clean up any tokens
	// that are no longer needed after the processing of the previous batch
	for seq, seqRange := range c.cellRanges {
		if _, ok := lowestPos[seq]; !ok {
			var last int32
			for i := seqRange.min; i <= seqRange.max; i++ {
				if c.cells[i].seqMask.has(seq) {
					last = max(last, c.cells[i].pos)
				}
			}
			lowestPos[seq] = lowestPosition{pos: last + 1, curBatch: false}
		}
	}

	// Delete any entries that are beyond the window of the oldest position in the sequence
	for seq, lowest := range lowestPos {
		oldRange, ok := c.cellRanges[seq]
		if !ok {
			continue
		}

		newRange := newRange()

		for i := oldRange.min; i <= oldRange.max; i++ {
			if c.cells[i].seqMask.has(seq) {
				if c.cells[i].pos < lowest.pos-c.swaMemorySize {
					c.cells[i].seqMask.clear(seq)
					// If cell is now empty, return it to free list
					if c.cells[i].seqMask.isEmpty() {
						c.freeCell(i)
						// Clean up compressed segment if it exists
						c.compressionMu.Lock()
						for layer := range c.compressedSegments {
							delete(c.compressedSegments[layer], i)
						}
						c.compressionMu.Unlock()
					}
				} else {
					newRange.min = min(newRange.min, i)
					newRange.max = max(newRange.max, i)
				}
				if lowest.curBatch && c.cells[i].pos >= lowest.pos-c.swaWindowSize {
					c.curCellRange.min = min(c.curCellRange.min, i)
					c.curCellRange.max = max(c.curCellRange.max, i)
				}
			}
		}

		c.cellRanges[seq] = newRange
	}

	// Trigger compression for old segments
	if c.compressionEnabled {
		var maxPos int32
		for _, pos := range c.curPositions {
			if pos > maxPos {
				maxPos = pos
			}
		}
		c.compressOldSegments(maxPos)
	}
}

func roundDown(length, pad int) int {
	return (length / pad) * pad
}

func roundUp(length, pad int) int {
	return ((length + pad - 1) / pad) * pad
}

// ============================================================================
// MASK BUILDING - Highly optimized with bitmaps and scratch buffers
// ============================================================================

// Builds a mask of history x batch indicating whether for each token in the batch the
// token in the history should apply. This is based on both the sequence and causality (the
// position of the history is not ahead of the token in the batch).
func (c *Causal) buildMask(ctx ml.Context) ml.Tensor {
	// Align and pad the two dimensions as required by the backend
	batchSize := roundUp(c.curBatchSize, c.config.MaskBatchPadding)

	c.curCellRange.min = roundDown(c.curCellRange.min, c.config.CachePadding)
	c.curCellRange.max = roundUp(c.curCellRange.max+1, c.config.CachePadding) - 1

	length := c.curCellRange.max - c.curCellRange.min + 1
	maskSize := batchSize * length

	// Reuse scratch buffer if large enough, otherwise allocate
	var mask []float32
	if cap(c.scratchMask) >= maskSize {
		mask = c.scratchMask[:maskSize]
	} else {
		mask = make([]float32, maskSize)
	}

	// Initialize all mask values to -Inf (masked out)
	// This is more efficient than checking every condition and setting selectively
	negInf := float32(math.Inf(-1))
	for i := range mask {
		mask[i] = negInf
	}

	// Pre-compute except bitmap for O(1) lookup
	var exceptMask uint64
	for _, idx := range c.opts.Except {
		if idx >= 0 && idx < 64 {
			exceptMask |= 1 << uint(idx)
		}
	}

	// Now selectively unmask valid positions
	// This is faster when most positions are masked (common case with causality)
	for i := range c.curBatchSize {
		curSeq := c.curSequences[i]
		curPos := c.curPositions[i]
		enabled := (exceptMask & (1 << uint(i))) == 0

		// Pre-compute chunk start if chunked attention is used
		var chunkStart int32
		if c.chunkSize > 0 {
			chunkStart = curPos - curPos%c.chunkSize
		}

		rowOffset := i * length

		for j := c.curCellRange.min; j <= c.curCellRange.max; j++ {
			cell := &c.cells[j]
			colOffset := j - c.curCellRange.min

			// Fast bitmap check for sequence membership - O(1) instead of O(n)
			if !cell.seqMask.has(curSeq) {
				continue // Already masked
			}

			// Causality check
			if enabled && cell.pos > curPos {
				continue // Already masked
			}

			// Chunked attention check
			if c.chunkSize > 0 && cell.pos < chunkStart {
				continue // Already masked
			}

			// Sliding window check
			if cell.pos < curPos-c.swaWindowSize {
				continue // Already masked
			}

			// This position should NOT be masked
			mask[rowOffset+colOffset] = 0
		}
	}

	// Padding tokens are already masked by default (we initialized to -Inf)
	// No need for explicit padding loop

	maskTensor := ctx.Input().FromFloats(mask, length, batchSize)

	if c.config.MaskDType != ml.DTypeF32 {
		maskTensor = maskTensor.Cast(ctx, c.config.MaskDType)
	}

	return maskTensor
}

// ============================================================================
// LAYER AND OPTIONS
// ============================================================================

func (c *Causal) SetLayer(layer int) {
	c.curLayer = layer
}

// RecordAttentionScores implements the CompressionAwareCache interface.
func (c *Causal) RecordAttentionScores(layer int, scores []float32) {
	c.compressionMu.Lock()
	defer c.compressionMu.Unlock()
	c.attentionScores[layer] = scores
}

type CausalOptions struct {
	// Enabled controls whether the causal mask is generated for a particular index in a batch
	Except []int
}

// SetCausal disables causal mask generation for a particular range of indicies in
// the current batch for subsequent calls to Get. The state resets for the next forward pass.
func (c *Causal) SetCausal(ctx ml.Context, opts CausalOptions) {
	// Check if options actually changed
	if len(c.opts.Except) == len(opts.Except) {
		same := true
		for i := range opts.Except {
			if c.opts.Except[i] != opts.Except[i] {
				same = false
				break
			}
		}
		if same {
			return
		}
	}

	c.opts = opts
	if ctx != nil {
		c.curMask = c.buildMask(ctx)
	}
}

// ============================================================================
// GET AND PUT
// ============================================================================

func (c *Causal) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	key := c.keys[c.curLayer]
	value := c.values[c.curLayer]

	// Decompress any cells in the current range that are needed
	if c.compressionEnabled {
		for i := c.curCellRange.min; i <= c.curCellRange.max; i++ {
			if c.cells[i].compressed {
				kData, vData := c.decompressSegment(c.curLayer, i)
				if kData != nil && vData != nil {
					// Write back to tensors
					// This is a simplified write-back. In production, we'd use
					// a more efficient way to update the tensor data.
					c.writeCellData(ctx, key, i, kData)
					c.writeCellData(ctx, value, i, vData)
					c.cells[i].compressed = false
				}
			}
		}
	}

	kHeadDim := key.Dim(0)
	numKVHeads := key.Dim(1)
	rowSize := key.Stride(2)
	cachedSize := c.curMask.Dim(0)

	key = key.View(ctx, rowSize*c.curCellRange.min,
		kHeadDim, key.Stride(1),
		numKVHeads, key.Stride(2),
		cachedSize,
	)

	if c.config.PermutedV {
		vHeadDim := value.Dim(1)
		elemSize := value.Stride(0)

		value = value.View(ctx, elemSize*c.curCellRange.min,
			cachedSize, value.Stride(1),
			vHeadDim, value.Stride(2),
			numKVHeads,
		)
	} else {
		vHeadDim := value.Dim(0)
		rowSize := value.Stride(2)

		value = value.View(ctx, rowSize*c.curCellRange.min,
			vHeadDim, value.Stride(1),
			numKVHeads, value.Stride(2),
			cachedSize,
		)
	}

	return key, value, c.curMask
}

func (c *Causal) Put(ctx ml.Context, key, value ml.Tensor) {
	kHeadDim := key.Dim(0)
	vHeadDim := value.Dim(0)
	numKVHeads := key.Dim(1)
	batchSize := key.Dim(2)

	if c.curBatchSize != batchSize {
		panic(fmt.Errorf("inconsistent batch sizes (layer: %v, batch size: %v layer batch size: %v)", c.curLayer, c.curBatchSize, batchSize))
	}

	if _, ok := c.ctxs[c.curLayer]; !ok {
		c.ctxs[c.curLayer] = c.backend.NewContextSize(2).Layer(c.curLayer)
	}

	if _, ok := c.keys[c.curLayer]; !ok {
		c.keys[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, kHeadDim, numKVHeads, len(c.cells))
	}

	if _, ok := c.values[c.curLayer]; !ok {
		if c.config.PermutedV {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, len(c.cells), vHeadDim, numKVHeads)
		} else {
			c.values[c.curLayer] = c.ctxs[c.curLayer].Zeros(c.DType, vHeadDim, numKVHeads, len(c.cells))
		}
	}

	key = key.Reshape(ctx, kHeadDim*numKVHeads, batchSize)
	keyCache := c.keys[c.curLayer]
	keyCache = keyCache.Reshape(ctx, kHeadDim*numKVHeads, len(c.cells))
	ctx.Forward(keyCache.SetRows(ctx, key, c.curLoc))

	if c.config.PermutedV {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, 1, batchSize)
		value = value.Permute(ctx, 2, 0, 1, 3)

		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, 1, len(c.cells), vHeadDim*numKVHeads)

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	} else {
		value = value.Reshape(ctx, vHeadDim*numKVHeads, batchSize)
		valueCache := c.values[c.curLayer]
		valueCache = valueCache.Reshape(ctx, vHeadDim*numKVHeads, len(c.cells))

		ctx.Forward(valueCache.SetRows(ctx, value, c.curLoc))
	}
}

func (c *Causal) writeCellData(ctx ml.Context, t ml.Tensor, cellIdx int, data []float32) {
	// Simplified write-back using SetRows or similar
	// In production, we'd use a more direct method.
	idxs := ctx.Input().FromInts([]int32{int32(cellIdx)}, 1)
	rowSize := t.Dim(0) * t.Dim(1)
	row := ctx.Input().FromFloats(data, rowSize, 1)

	// Reshape t to [rowSize, numCells] for SetRows
	numCells := len(c.cells)
	tReshaped := t.Reshape(ctx, rowSize, numCells)
	ctx.Forward(tReshaped.SetRows(ctx, row, idxs))
}

// ============================================================================
// COPY PREFIX - Optimized with targeted range scanning
// ============================================================================

func (c *Causal) CopyPrefix(srcSeq, dstSeq int, length int32) {
	// First, remove dstSeq from its current cells (only scan dst range)
	if dstRange, ok := c.cellRanges[dstSeq]; ok {
		for i := dstRange.min; i <= dstRange.max; i++ {
			if c.cells[i].seqMask.has(dstSeq) {
				c.cells[i].seqMask.clear(dstSeq)
				// If cell is now empty, return it to free list
				if c.cells[i].seqMask.isEmpty() {
					c.freeCell(i)
				}
			}
		}
	}

	// Now copy from srcSeq (only scan src range)
	srcRange, ok := c.cellRanges[srcSeq]
	if !ok {
		c.cellRanges[dstSeq] = newRange()
		return
	}

	seqRange := newRange()

	for i := srcRange.min; i <= srcRange.max; i++ {
		if c.cells[i].seqMask.has(srcSeq) && c.cells[i].pos < length {
			c.cells[i].seqMask.set(dstSeq)
			seqRange.min = min(seqRange.min, i)
			seqRange.max = max(seqRange.max, i)
		}
	}

	c.cellRanges[dstSeq] = seqRange
}

// ============================================================================
// CAN RESUME
// ============================================================================

func (c *Causal) CanResume(seq int, pos int32) bool {
	if c.swaMemorySize == math.MaxInt32 {
		return true
	}

	seqRange, ok := c.cellRanges[seq]
	if !ok {
		return false
	}

	// For sliding window, check that the window of the new sequence is contained in
	// the window of what we are storing
	var first int32 = math.MaxInt32
	var last int32 = -1
	for i := seqRange.min; i <= seqRange.max; i++ {
		if c.cells[i].seqMask.has(seq) {
			first = min(first, c.cells[i].pos)
			last = max(last, c.cells[i].pos)
		}
	}

	if last == -1 {
		return false
	}

	posWindowStart := max(0, pos-c.swaWindowSize)
	return posWindowStart >= first && pos <= last+1
}

// ============================================================================
// SHIFT - Position shifting for removed tokens
// ============================================================================

func (c *Causal) shift(seq int, beginIndex, offset int32) error {
	if c.shiftFn == nil {
		return ErrNotSupported
	}

	seqRange := c.cellRanges[seq]

	for start := seqRange.min; start <= seqRange.max; start += c.maxBatch {
		size := min(seqRange.max-start+1, c.maxBatch)
		offsets := make([]int32, size)

		var batchFirst, batchLast int

		batchFirst = -1
		for i := range offsets {
			cell := c.cells[start+i]

			if cell.seqMask.has(seq) && cell.pos >= beginIndex {
				offsets[i] = offset
				if batchFirst < 0 {
					batchFirst = i
				}
				batchLast = i
			}
		}

		if batchFirst < 0 {
			continue
		}

		offsets = offsets[batchFirst : batchLast+1]

		ctx := c.backend.NewContext()
		kShift := ctx.Input().FromInts(offsets, len(offsets))

		for i, key := range c.keys {
			if key == nil {
				continue
			}

			kHeadDim := key.Dim(0)
			numKVHeads := key.Dim(1)
			rowSize := key.Stride(2)

			key = key.View(ctx, rowSize*(start+batchFirst),
				kHeadDim, key.Stride(1),
				numKVHeads, key.Stride(2),
				len(offsets),
			)

			roped, err := c.shiftFn(ctx, i, key, kShift)
			if err != nil {
				ctx.Close()
				return err
			}

			ctx.Forward(roped.Copy(ctx, key))
		}

		ctx.Compute()
		ctx.Close()
	}

	return nil
}

// ============================================================================
// REMOVE - Remove tokens from cache
// ============================================================================

func (c *Causal) Remove(seq int, beginIndex, endIndex int32) error {
	// TODO(jessegross): We should check to see if removing the middle of the sequence will
	// cause the sliding window to encompass tokens that we no longer have. If so, then we
	// should return an error, which will trigger the runner to evaluate the full history and
	// rebuild the window. However, if we have multimodal inputs in our history, this reuse
	// results in use after free, so we don't do it for now.

	var offset int32
	if endIndex != math.MaxInt32 {
		offset = beginIndex - endIndex
	}

	seqRange := newRange()

	for i := range c.cells {
		if c.cells[i].seqMask.has(seq) {
			if c.cells[i].pos >= beginIndex && c.cells[i].pos < endIndex {
				c.cells[i].seqMask.clear(seq)
				// If cell is now empty, return it to free list
				if c.cells[i].seqMask.isEmpty() {
					c.freeCell(i)
				}
			} else {
				if c.cells[i].pos >= endIndex {
					if c.cells[i].seqMask.hasOtherThan(seq) {
						return errors.New("shifting cells shared by multiple sequences not supported")
					}

					c.cells[i].pos += offset
				}
				if i < seqRange.min {
					seqRange.min = i
				}
				if i > seqRange.max {
					seqRange.max = i
				}
			}
		}
	}

	if seqRange.min > seqRange.max {
		delete(c.cellRanges, seq)
		return nil
	}

	c.cellRanges[seq] = seqRange

	if endIndex != math.MaxInt32 {
		err := c.shift(seq, endIndex+offset, offset)
		if err != nil {
			return err
		}
	}

	return nil
}
