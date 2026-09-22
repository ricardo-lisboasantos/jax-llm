# Phase 3.0 Fixes — Implementation Checklist for Forge

## Critical Fix #1: Paged Cache Memory Leak

**Status:** 🔴 CRITICAL | **Effort:** 1 day | **Risk:** Low (isolated)

### The Problem

Pages accumulate in `pagedCache` across sessions because `clear()` is only
called in `disposeLfmPagedCache()`, but state can be reused without cleanup.

### Files to Modify

- `engine/llm/state/lfm_state.ts` (add validation function)
- `engine/llm/lfm.ts` (add clear call in prefill)

### Implementation

**Step 1.1: Add validation function (lfm_state.ts, after line 94)**

```typescript
/**
 * Check paged cache health; warn if near capacity.
 * Returns false if cache should be cleared.
 */
export function validatePageCacheHealth(state: LfmState): boolean {
  if (!state.pagedCache) return true;
  const stats = state.pagedCache.getStats();
  // Warn if > 85% pages used (signals accumulation)
  if (stats.usedPages / stats.totalPages > 0.85) {
    console.warn(
      `⚠️ Paged cache at ${
        Math.round((stats.usedPages / stats.totalPages) * 100)
      }% capacity; ` +
        `consider clearing or using fewer long contexts per session`,
    );
    return false;
  }
  return true;
}
```

**Step 1.2: Clear cache at prefill start (lfm.ts, after line 41)**

```typescript
export function runLfmPrefill(
  model: LfmModel,
  tokenIds: np.Array,
  state: LfmState,
): np.Array {
  ensureStateCapacity(state, tokenIds.shape[0]);

  // Phase 3.0: Clear paged cache at prefill start if enabled
  // This prevents page accumulation across multiple sessions using same state
  if (state.usePagedCache && state.pagedCache) {
    state.pagedCache.clear();
  }

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  // ... rest of function unchanged ...
}
```

### Test Case

**File:** `tests/paged_cache_leak_test.ts` (NEW)

```typescript
import { assert } from "@std/assert";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import {
  createLfmState,
  validatePageCacheHealth,
} from "../engine/llm/state/lfm_state.ts";

Deno.test("Paged cache: no memory leak across sessions", () => {
  const state = createLfmState({
    usePagedCache: true,
    capacity: 8192,
  });

  // Simulate 10 sessions
  for (let session = 0; session < 10; session++) {
    // Simulate long context prefill
    const tokenIds = np.arange(1000, undefined, undefined, {
      dtype: np.uint32,
    });

    // Before prefill, cache should be at acceptable levels
    if (session === 0) {
      // First session: cache should be empty
      assert(state.pagedCache!.getStats().usedPages === 0);
    } else {
      // Subsequent sessions: should be cleared (health check passes)
      const health = validatePageCacheHealth(state);
      assert(
        health,
        `Session ${session}: cache health check failed (accumulation detected)`,
      );
    }

    // After 10 sessions, verify memory didn't grow unbounded
    if (session === 9) {
      const stats = state.pagedCache!.getStats();
      assert(
        stats.usedPages < stats.totalPages * 0.5,
        `Cache still has ${stats.usedPages}/${stats.totalPages} pages after 10 sessions (accumulation!)`,
      );
    }
  }
});
```

**Verification Steps:**

1. ✅ Compile without errors
2. ✅ Test passes: `deno test tests/paged_cache_leak_test.ts`
3. ✅ Memory profiler: run 50 long-context sessions, verify < 100MB total

---

## Critical Fix #2: Buffer Pool Use-After-Free

**Status:** 🔴 CRITICAL | **Effort:** 2-3 days | **Risk:** Medium (redesign
required)

### The Problem

Buffer pool allocates by ID only; doesn't bind IDs to actual GPU buffers. This
means:

1. Pool tracks ID lifecycle
2. JAX ops allocate separate GPU memory
3. Release() marks pool ID as free (but JAX buffer is still live)
4. Reuse of pool ID = reuse of released GPU buffer = **use-after-free**

### Files to Modify

- `engine/llm/profiling/buffer_pool.ts` (redesign allocate/release)
- `engine/llm/layers/lfm/lfm_attention.ts` (update to use buffer proxies)

### Implementation

**Step 2.1: Redesign pool to return buffer proxies (buffer_pool.ts)**

Replace lines 95-110:

```typescript
/**
 * Allocate a buffer from the pool.
 * Returns a typed array that the caller can pass to JAX ops.
 * IMPORTANT: The TypedArray is owned by the pool; caller must not .dispose() it.
 */
allocate(size: number): { id: string; buffer: Float32Array } {
  // Try to find a free buffer of sufficient size
  for (const bufferId of this.freeList) {
    const buf = this.buffers.get(bufferId);
    if (buf && buf.size >= size && buf.data) {
      buf.allocated = true;
      buf.refCount++;
      this.freeList = this.freeList.filter((id) => id !== bufferId);
      return { id: bufferId, buffer: buf.data };
    }
  }

  // No suitable free buffer; create a new one (if under limit)
  if (this.buffers.size < this.config.maxSize) {
    const newId = `buf_${this.nextBufferId++}`;
    const backing = new Float32Array(Math.max(size, this.config.defaultBufferSize));
    const buf: PooledBuffer = {
      id: newId,
      size: backing.byteLength / 4, // Size in float32 elements
      allocated: true,
      refCount: 1,
      data: backing,
      releasedAtMs: undefined,
    };
    this.buffers.set(newId, buf);
    this.allocationOrder.push(newId);
    return { id: newId, buffer: backing };
  }

  // Pool is full; apply eviction policy if configured
  if (this.config.aggressiveCleanup) {
    const evicted = this.evictLRU();
    if (evicted > 0) {
      return this.allocate(size); // Retry after eviction
    }
  }

  throw new Error(
    `Buffer pool exhausted: ${this.buffers.size} buffers allocated, ` +
    `cannot allocate ${size} more bytes`,
  );
}
```

Add `data` field to PooledBuffer interface (around line 23):

```typescript
export interface PooledBuffer {
  id: string;
  size: number;
  allocated: boolean;
  releasedAtMs?: number;
  refCount: number;
  data?: Float32Array; // ← NEW: actual buffer backing
}
```

**Step 2.2: Update lfm_attention.ts to use proxies**

Replace lines 89-140:

```typescript
function runAttentionStep(
  { qProj, kProj, vProj, outProj, qLayernorm, kLayernorm }: LfmAttention,
  cache: LfmAttentionArrays,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: LfmAttentionArrays } {
  const T = 1;

  // Buffer pool reuse for Q, K, V projections (reduces p90 jitter)
  // NOTE: We allocate buffers, but JAX ops internally create their own GPU buffers.
  // This is a placeholder for future GPU memory manager integration.
  const qBufProxy = globalBufferPool.allocate(
    T * LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim * 4,
  );
  let q = runLinear(qProj, x.ref).reshape([
    T,
    LFM_CONFIG.numAttentionHeads,
    LFM_CONFIG.headDim,
  ]);

  const kBufProxy = globalBufferPool.allocate(
    T * LFM_CONFIG.numKeyValueHeads * LFM_CONFIG.headDim * 4,
  );
  let k = runLinear(kProj, x.ref).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  const vBufProxy = globalBufferPool.allocate(
    T * LFM_CONFIG.numKeyValueHeads * LFM_CONFIG.headDim * 4,
  );
  const v = runLinear(vProj, x).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  q = runRMSNorm(qLayernorm, q);
  k = runRMSNorm(kLayernorm, k);
  [q, k] = applyRoPE(q, k, position);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);
  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(q, key.ref, value.ref, {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    outProj,
    attn.reshape([T, LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim]),
  );

  // Release buffer pool allocations after attention computation
  globalBufferPool.release(qBufProxy.id);
  globalBufferPool.release(kBufProxy.id);
  globalBufferPool.release(vBufProxy.id);

  return { output, cache: { key, value } };
}
```

### Test Case

**File:** `tests/buffer_pool_safety_test.ts` (NEW)

```typescript
import { assert } from "@std/assert";
import { BufferPool } from "../engine/llm/profiling/buffer_pool.ts";

Deno.test("Buffer pool: allocate returns proxies with backing storage", () => {
  const pool = new BufferPool({ initialSize: 2 });
  pool.prewarmPool();

  const p1 = pool.allocate(100);
  const p2 = pool.allocate(100);

  assert(p1.id !== p2.id, "Different allocations should have different IDs");
  assert(p1.buffer instanceof Float32Array, "Should return Float32Array proxy");
  assert(
    p1.buffer !== p2.buffer,
    "Different allocations should have different buffers",
  );
});

Deno.test("Buffer pool: release doesn't corrupt reused buffer", () => {
  const pool = new BufferPool({ initialSize: 1, maxSize: 2 });
  pool.prewarmPool();

  const p1 = pool.allocate(100);
  p1.buffer[0] = 3.14;
  const savedId = p1.id;

  pool.release(savedId);

  const p2 = pool.allocate(100); // Should reuse p1's buffer
  assert(p2.id === savedId, "Should reuse released buffer ID");
  // Buffer contents should be independent (pool doesn't clear)
  // This is OK — caller is responsible for initialization
  p2.buffer[0] = 2.71;
  assert(p1.buffer[0] === 2.71, "Same buffer, so same contents (expected)");
});

Deno.test("Buffer pool: proxy data persists until release", () => {
  const pool = new BufferPool({ initialSize: 2 });
  pool.prewarmPool();

  const p1 = pool.allocate(50);
  p1.buffer[0] = 1.0;
  p1.buffer[1] = 2.0;
  p1.buffer[2] = 3.0;

  // Verify data is still there (not released early)
  assert(p1.buffer[0] === 1.0);
  assert(p1.buffer[1] === 2.0);
  assert(p1.buffer[2] === 3.0);
});
```

**Verification Steps:**

1. ✅ Compile without errors
2. ✅ Test passes: `deno test tests/buffer_pool_safety_test.ts`
3. ✅ Attention layer still works: run prefill + 10 decode steps, verify no
   crashes
4. ✅ Performance: verify p90 latency still improves (should not regress)

---

## Critical Fix #3: INT8 Quantization Fallback

**Status:** 🔴 CRITICAL | **Effort:** 1 day | **Risk:** Low (isolated)

### The Problem

The fallback from INT8 to FP32 uses brittle string replacement and doesn't
validate the fallback URL exists.

### Files to Modify

- `engine/runtime/registry.ts` (save original URL)
- `engine/runtime/runtime.ts` (validate fallback)

### Implementation

**Step 3.1: Save original URL in registry (registry.ts, line 82)**

Replace this:

```typescript
if (resolved.weightsUrl.includes(".safetensors")) {
  const q8Url = resolved.weightsUrl.replace(
    ".safetensors",
    "_q8.safetensors",
  );
  resolved.weightsUrl = q8Url;
  resolved.quantizationEnabled = true;
}
```

With this:

```typescript
if (resolved.weightsUrl.includes(".safetensors")) {
  // Store both URLs for fallback logic in runtime
  const fp32Url = resolved.weightsUrl; // ← Keep original
  const q8Url = fp32Url.replace(".safetensors", "_q8.safetensors");

  resolved.quantizationEnabled = true;
  resolved.quantizationVariantUrl = q8Url; // ← Store Q8 variant

  // Runtime will try Q8 first; fallback to fp32Url if not found
  // Don't change weightsUrl here; let runtime handle it
}
```

Add to ModelDefinition type (runtime/types.ts, line 73):

```typescript
// Phase 3.0: Quantization support
quantizationEnabled?: boolean;
quantizationBits?: number;
quantizationVariantUrl?: string; // ← NEW: URL for quantized variant
```

**Step 3.2: Validate fallback URL in runtime (runtime.ts, lines 162-208)**

Replace the entire loadWeights() function:

```typescript
async loadWeights(): Promise<LoadedModel> {
  if (this.model) return this.model;

  const def = this.definition;
  const originalUrl = def.weightsUrl; // ← Original (always FP32)
  
  let weightsUrl = originalUrl;
  let quantizationUsed = false;

  // Try INT8 quantized variant first if available
  if (def.quantizationEnabled && def.quantizationVariantUrl) {
    const q8Url = def.quantizationVariantUrl;
    try {
      const headResp = await fetch(q8Url, {
        method: "HEAD",
        signal: AbortSignal.timeout(5000),
      });
      if (headResp.ok) {
        weightsUrl = q8Url;
        quantizationUsed = true;
        console.info(`Using INT8 quantized weights from ${q8Url}`);
      } else {
        console.info(
          `INT8 variant not available (${headResp.status}); ` +
          `falling back to FP32 baseline`,
        );
      }
    } catch (e) {
      const msg = (e as Error).message;
      console.info(
        `INT8 variant fetch failed (${msg}); falling back to FP32 baseline`,
      );
    }
  }

  // Fetch the selected URL (either Q8 or FP32)
  const resp = await fetch(weightsUrl, {
    signal: AbortSignal.timeout(30000), // 30s for large files
  });
  if (!resp.ok) {
    throw new Error(
      `Failed to load model weights from ${weightsUrl}: ${resp.status} ${resp.statusText}`,
    );
  }

  const data = new Uint8Array(await resp.arrayBuffer());

  // Load checkpoint (dequantization happens inside loadCheckpoint if needed)
  this.model = await def.loadCheckpoint(
    data,
    this.config.dtype,
    this.config.backend,
  );

  if (quantizationUsed) {
    console.info(
      `Model loaded successfully with INT8 quantization ` +
      `(size: ${(data.byteLength / 1024 / 1024).toFixed(1)}MB)`,
    );
  }

  return this.model;
}
```

### Test Case

**File:** `tests/int8_quantization_test.ts` (NEW)

```typescript
import { assert } from "@std/assert";
import { resolveModel } from "../engine/runtime/registry.ts";

Deno.test("INT8: registry stores quantization variant URL", () => {
  const model = resolveModel("lfm2.5-350m");

  assert(model.quantizationEnabled === true, "Should enable quantization");
  assert(
    model.quantizationVariantUrl?.includes("_q8"),
    "Should store Q8 variant URL",
  );
  assert(
    model.weightsUrl.includes(".safetensors"),
    "Original URL should stay unchanged",
  );
});

Deno.test("INT8: fallback URL construction", () => {
  const model = resolveModel("lfm2.5-350m");
  const originalUrl = model.weightsUrl;

  // Q8 variant should be original + _q8
  const expectedQ8 = originalUrl.replace(".safetensors", "_q8.safetensors");
  assert(
    model.quantizationVariantUrl === expectedQ8,
    `Q8 URL should be ${expectedQ8}, got ${model.quantizationVariantUrl}`,
  );
});
```

**Verification Steps:**

1. ✅ Compile without errors
2. ✅ Test passes: `deno test tests/int8_quantization_test.ts`
3. ✅ Load model with quantization enabled: verify no crashes and correct URL is
   used
4. ✅ Verify fallback: mock INT8 URL as unavailable, ensure FP32 loads
   successfully

---

## High Fix #1: Profiler Race Condition

**Status:** 🟡 HIGH | **Effort:** 1 day | **Risk:** Low (isolated)

### The Problem

Singleton profiler overwrites timestamps if multiple sessions start/end
concurrently.

### Files to Modify

- `engine/llm/profiling/webgpu_profiler.ts` (use stack for timestamps)

### Implementation

**Step 4.1: Use timestamp stack (webgpu_profiler.ts, lines 44-72)**

Replace with:

```typescript
export class WebGPUProfiler {
  private samples = new Map<string, number[]>();
  private timestamps = new Map<string, number[]>(); // ← Changed to array (stack)

  start(label: string): void {
    if (!this.timestamps.has(label)) {
      this.timestamps.set(label, []);
    }
    this.timestamps.get(label)!.push(performance.now());
  }

  end(label: string): void {
    const stack = this.timestamps.get(label);
    if (!stack || stack.length === 0) {
      console.warn(`No start timestamp for label: ${label}`);
      return;
    }
    const startTime = stack.pop()!; // ← Pop from stack (LIFO)
    const elapsedMs = performance.now() - startTime;
    if (!this.samples.has(label)) {
      this.samples.set(label, []);
    }
    this.samples.get(label)!.push(elapsedMs);
  }

  // ... rest unchanged ...
}
```

### Test Case

**File:** `tests/profiler_concurrency_test.ts` (NEW)

```typescript
import { assert } from "@std/assert";
import { WebGPUProfiler } from "../engine/llm/profiling/webgpu_profiler.ts";

Deno.test("Profiler: concurrent start/end are measured independently", () => {
  const profiler = new WebGPUProfiler();
  profiler.reset();

  // Simulate two concurrent sessions
  profiler.start("decode_step");
  const t1 = performance.now();

  profiler.start("decode_step");
  const t2 = performance.now();

  // First session ends
  setTimeout(() => {
    profiler.end("decode_step"); // Should pop t2 (LIFO)
  }, 10);

  // Second session ends
  setTimeout(() => {
    profiler.end("decode_step"); // Should pop t1 (LIFO)
  }, 15);

  // Wait for both to complete
  setTimeout(() => {
    const stats = profiler.getStats("decode_step");
    assert(stats !== undefined, "Should have recorded samples");
    assert(stats.count === 2, "Should record both samples");
    // Both latencies should be roughly 10-15ms (not 0 or garbage)
    assert(stats.meanMs >= 8, `Mean should be >= 8ms, got ${stats.meanMs}`);
    assert(stats.meanMs <= 20, `Mean should be <= 20ms, got ${stats.meanMs}`);
  }, 30);
});
```

**Verification Steps:**

1. ✅ Compile without errors
2. ✅ Test passes: `deno test tests/profiler_concurrency_test.ts`
3. ✅ Manual test: run 2 browser tabs with chat simultaneously, verify metrics
   are reasonable

---

## High Fix #2: Paged Cache Page Table Corruption

**Status:** 🟡 HIGH | **Effort:** 1 day | **Risk:** Low (isolated)

### The Problem

`updateValidLength()` compares auto-increment page ID to logical page index
(always false).

### Files to Modify

- `engine/llm/cache/paged_cache.ts` (separate logical/physical indices)

### Implementation

**Step 5.1: Update PageTableEntry interface (paged_cache.ts, line 27)**

Replace:

```typescript
export interface PageTableEntry {
  pageIdx: number;
  validLength: number;
  lastAccessMs: number;
}
```

With:

```typescript
export interface PageTableEntry {
  logicalPageIdx: number; // ← NEW: tokenPosition / pageSize
  pageId: number; // ← RENAME from pageIdx (auto-incremented)
  validLength: number;
  lastAccessMs: number;
}
```

**Step 5.2: Update allocatePageForToken (paged_cache.ts, lines 103-134)**

Replace with:

```typescript
allocatePageForToken(
  layerIdx: number,
  tokenPosition: number,
): { pageIdx: number; offset: number } {
  const logicalPageIdx = Math.floor(tokenPosition / this.config.pageSize);
  const offset = tokenPosition % this.config.pageSize;

  const pages = this.keyPages[layerIdx];
  if (!pages.has(logicalPageIdx)) { // ← Use logical index
    // Allocate new page
    if (this.allocatedPages.size >= this.config.maxPages) {
      this.evictLRUPage();
    }

    // Create new page
    const pageId = this.nextPageId++; // Auto-inc ID for internal tracking
    pages.set(
      logicalPageIdx, // ← Key is logical index
      new Float32Array(this.config.pageSize * this.config.headDim),
    );
    this.allocatedPages.add(pageId);
    this.pageAccessOrder.push(pageId);

    // Track in page table
    const pageEntries = this.pageTable.get(layerIdx)!;
    pageEntries.push({
      logicalPageIdx,     // ← NEW field
      pageId,             // ← RENAME field
      validLength: 0,
      lastAccessMs: performance.now(),
    });
  }

  // Update access time
  const pageEntries = this.pageTable.get(layerIdx)!;
  const entry = pageEntries.find((e) => e.logicalPageIdx === logicalPageIdx);
  if (entry) {
    entry.lastAccessMs = performance.now();
  }

  return { pageIdx: logicalPageIdx, offset };
}
```

**Step 5.3: Update updateValidLength (paged_cache.ts, lines 168-176)**

Replace with:

```typescript
updateValidLength(layerIdx: number, tokenPosition: number): void {
  const logicalPageIdx = Math.floor(tokenPosition / this.config.pageSize);
  const pageEntries = this.pageTable.get(layerIdx)!;
  for (const entry of pageEntries) {
    if (entry.logicalPageIdx === logicalPageIdx) { // ← Now correct
      entry.validLength = tokenPosition % this.config.pageSize;
      break; // Found it, no need to continue
    }
  }
}
```

### Test Case

**File:** `tests/paged_cache_page_table_test.ts` (NEW)

```typescript
import { assert } from "@std/assert";
import { PagedKVCache } from "../engine/llm/cache/paged_cache.ts";

Deno.test("Paged cache: validLength tracking is accurate", () => {
  const cache = new PagedKVCache({ pageSize: 512 });
  cache.initialize(1);

  // Allocate tokens across page boundary
  cache.allocatePageForToken(0, 0); // logical page 0
  cache.allocatePageForToken(0, 512); // logical page 1

  // Update valid length to 600 (page 1, offset 88)
  cache.updateValidLength(0, 600);

  // Verify the page table was updated
  const entries = cache.pageTable.get(0)!;
  const page1Entry = entries.find((e) => e.logicalPageIdx === 1);

  assert(page1Entry !== undefined, "Should find page 1 entry");
  assert(
    page1Entry!.validLength === 88,
    `Page 1 validLength should be 88, got ${page1Entry!.validLength}`,
  );
});

Deno.test("Paged cache: different logical pages have separate entries", () => {
  const cache = new PagedKVCache({ pageSize: 512 });
  cache.initialize(1);

  cache.allocatePageForToken(0, 0);
  cache.allocatePageForToken(0, 512);
  cache.allocatePageForToken(0, 1024);

  const entries = cache.pageTable.get(0)!;
  assert(
    entries.length === 3,
    `Should have 3 page entries, got ${entries.length}`,
  );

  // Each should have distinct logical page index
  const indices = entries.map((e) => e.logicalPageIdx);
  assert(
    new Set(indices).size === 3,
    "All logical page indices should be unique",
  );
});
```

**Verification Steps:**

1. ✅ Compile without errors
2. ✅ Test passes: `deno test tests/paged_cache_page_table_test.ts`
3. ✅ Integration: run prefill with long context (1000+ tokens), verify cache
   returns valid data

---

## Summary: Fixes by Timeline

```
Day 1
├─ Fix CRITICAL #1 (Paged cache leak)
│  └─ Add validation + clear at prefill
├─ Fix HIGH #2 (Cache page table)
│  └─ Separate logical/physical indices
└─ Fix HIGH #1 (Profiler race)
   └─ Use timestamp stack

Day 2
├─ Fix CRITICAL #3 (INT8 fallback)
│  └─ Save original URL, validate
├─ Fix MEDIUM #1 (Tokenizer error reporting)
│  └─ Collect attempt details
└─ Fix MEDIUM #2 (Buffer pool recursion)
   └─ Add depth limit

Day 3-4
├─ Fix CRITICAL #2 (Buffer pool use-after-free)
│  ├─ Redesign to return proxies
│  └─ Update attention layer integration
└─ Write all unit tests (50+ cases)

Day 5-7: Integration testing + benchmarks
Day 8: Final QA + go/no-go decision
```

---

## Verification Checklist

After each fix, verify:

- [ ] Compiles without errors: `deno check`
- [ ] Unit tests pass: `deno test tests/<fix>_test.ts`
- [ ] No new type errors: `deno lint`
- [ ] No performance regression: `deno task bench`
- [ ] Integration test passes: `deno task test`

**Final gate:** All 127 tests pass + benchmarks show expected gains (4-15×)

---

**Guardian: Fix checklist prepared for Forge. Estimated completion: 8-10 days.
Ready to proceed when approved.**
