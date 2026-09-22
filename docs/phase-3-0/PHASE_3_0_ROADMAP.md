# Phase 3.0 Performance Quick Wins — Implementation Roadmap

**Last Updated:** 2026-09-22\
**Target:** 4× latency reduction (p90 jitter) + 50% memory footprint\
**Status:** Ready for Forge implementation

---

## Executive Summary

Five targeted optimizations across 10 files, zero breaking changes:

| Optimization           | Files | TTFT Impact | p90 Jitter | Memory         | Priority |
| ---------------------- | ----- | ----------- | ---------- | -------------- | -------- |
| **Buffer Pool**        | 2     | —           | **-80%**   | —              | P0       |
| **Tokenizer Parallel** | 1     | **-40%**    | —          | —              | P1       |
| **Paged Cache**        | 3     | —           | —          | **-75%** (8K+) | P1       |
| **INT8 Quant**         | 3     | —           | —          | **-50%**       | P2       |
| **Profiling**          | 2     | —           | —          | —              | P2       |

---

## 1. Buffer Pool Activation (engine/llm/layers/lfm/lfm_attention.ts)

**Goal:** Eliminate WebGPU allocation stalls per decode token\
**Impact:** p90 jitter ±2-3ms → ±0.2-0.5ms

### Changes

- **Line 1-10:** Add `import { globalBufferPool }`
- **Line 89, 95, 100:** Pre-allocate QKV buffers from pool before projections
- **Line 123:** Release buffers after attention computation

### Code Pattern

```typescript
// BEFORE: Fresh allocation per token (stalls)
let q = runLinear(qProj, x.ref).reshape([...]);

// AFTER: Reuse from pool
const qBufferId = globalBufferPool.allocate(qSize);
let q = runLinear(qProj, x.ref).reshape([...]);
// ... computation ...
globalBufferPool.release(qBufferId);
```

### Testing

```bash
# Measure p90 latency over 100+ decode steps
// Enable profiling
engine.enableProfiling();
// Generate 100+ tokens
// Check: engine.getMetrics().p90JitterMs < 0.5
```

---

## 2. Tokenizer Parallelization (engine/runtime/runtime.ts)

**Goal:** Fetch all tokenizer URL candidates in parallel\
**Impact:** TTFT 2-4s → 1.2-1.8s (200-400ms saved)

### Changes

- **Lines 85-147:** Replace `for (const url of candidates)` with
  `Promise.allSettled()`
- Add 5-second timeout per URL to prevent hangs
- Return first successful tokenizer; log which URL won

### Code Pattern

```typescript
// BEFORE: Sequential (timeout = sum of all URLs)
for (const url of candidates) {
  try {
    const resp = await fetch(url);
    if (resp.ok) {
      // Parse and return
    }
  } catch {
    continue; // Next candidate
  }
}

// AFTER: Parallel (timeout = max of any URL)
const fetchPromises = candidates.map(async (url) => {
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
    // Parse and return
  } catch {
    return null;
  }
});
const results = await Promise.allSettled(fetchPromises);
// Find first successful result
```

### Testing

```bash
# Measure TTFT from init() to first token
const start = performance.now();
await engine.init();
const resp = await engine.generate("Hi");
const ttft = performance.now() - start;
// Expect: ttft < 2000ms (vs 2-4s baseline)
```

---

## 3. Paged Cache Integration (engine/llm/state/lfm_state.ts + engine/llm/lfm.ts)

**Goal:** On-demand page allocation for 8K+ token sequences\
**Impact:** Memory 32MB (fixed 8K) → 8MB (dynamic pages)

### Changes

- **lfm_state.ts lines 1-8:** Import `PagedKVCache`
- **Lines 9-13:** Add `pagedCache?: PagedKVCache` and `usePagedCache?: boolean`
  to `LfmState` type
- **Lines 15-45:** Modify `createLfmState()` to optionally initialize paged
  cache
- **After line 66:** Add `disposeLfmPagedCache()` function
- **lfm.ts line 10:** Import `disposeLfmPagedCache`
- **lfm.ts lines 145-146:** Call `pagedCache.updateValidLength()` at end of
  `runLfmStep()`

### Code Pattern

```typescript
// Create state with paged cache for long context
const state = createLfmState({
  capacity: 16384, // 16K tokens
  usePagedCache: true,
  pagedCacheConfig: {
    pageSize: 512,
    maxPages: 32,
  },
});

// In runLfmStep(), update page tracking
if (state.usePagedCache && state.pagedCache) {
  state.pagedCache.updateValidLength(state.position, state.position + 1);
}
```

### Testing

```bash
# Generate 8K+ tokens and check memory
const state = createLfmState({
  capacity: 8192,
  usePagedCache: true,
});
const startMem = performance.memory?.usedJSHeapSize;
// Generate 8K tokens...
const endMem = performance.memory?.usedJSHeapSize;
// Expect: endMem - startMem ≈ 8MB (vs 32MB fixed allocation)
```

---

## 4. INT8 Quantization Hook (engine/runtime/registry.ts + engine/runtime/runtime.ts)

**Goal:** Load INT8 weights; fallback to FP32 if unavailable\
**Impact:** Model size 350MB (FP32) → 175MB (INT8)

### Changes

- **registry.ts line 12:** Import `globalQuantizationCache`
- **Lines 71-76:** Add INT8 variant detection in `resolveModel()` (check for
  `_q8.safetensors`)
- **types.ts:** Add `quantizationEnabled?: boolean` and
  `quantizationBits?: number` to `ModelDefinition`
- **runtime.ts line 19:** Import `globalQuantizationCache`
- **Lines 153-170:** Modify `loadWeights()` to attempt INT8 URL with FP32
  fallback

### Code Pattern

```typescript
// In resolveModel()
if (model.weightsUrl.includes(".safetensors")) {
  const q8Url = model.weightsUrl.replace(".safetensors", "_q8.safetensors");
  resolved.weightsUrl = q8Url; // Will fallback if 404
  resolved.quantizationEnabled = true;
}

// In loadWeights()
let weightsUrl = def.weightsUrl;
if (def.quantizationEnabled) {
  const resp = await fetch(weightsUrl, { signal: AbortSignal.timeout(10000) })
    .catch(() => null);
  if (!resp?.ok) {
    // Fallback to FP32
    weightsUrl = def.weightsUrl.replace("_q8.safetensors", ".safetensors");
  }
}
```

### Testing

```bash
# Verify quantized weights load and produce correct logits
const modelQua = new ChatEngine("lfm2.5-350m");
await modelQua.init(); // Should load _q8.safetensors
const resp1 = await modelQua.generate("Test");

const modelFp32 = new ChatEngine("lfm2.5-350m", {
  weightOverrides: { "lfm2.5-350m": "path/to/fp32/model.safetensors" },
});
await modelFp32.init();
const resp2 = await modelFp32.generate("Test");

// Expect: logit similarity > 0.99 (within 2% accuracy loss)
```

---

## 5. Profiling & Monitoring (engine/chat/chat_engine.ts + engine/llm/profiling/webgpu_profiler.ts)

**Goal:** Track TTFT, tok/s, latency percentiles; detect regressions\
**Impact:** Enable automated performance dashboarding

### Changes

- **chat_engine.ts lines 28-34:** Import `globalProfiler`
- **Lines 41-51:** Add fields: `profilingEnabled`, `metrics`
- **After line 85:** Add methods: `enableProfiling()`, `getMetrics()`
- **Wrap prefill/decode loops:** `globalProfiler.start/end("prefill")`,
  `globalProfiler.start/end("decode_step")`
- **webgpu_profiler.ts after line 156:** Add `metricsJSON()` and
  `detectRegression()` methods

### Code Pattern

```typescript
// Enable profiling
engine.enableProfiling();

// Metrics available after generation
const metrics = engine.getMetrics();
// {
//   ttftMs: 1234,
//   prefillTokPerSec: 512,
//   decodeTokPerSec: 35,
//   p50JitterMs: 32,
//   p90JitterMs: 48,
// }

// Auto-regression detection
const stats = globalProfiler.getAllStats();
globalProfiler.detectRegression(stats, 50, 5); // Alert if p90 > 55ms
```

### Metrics Dashboard (for external APM)

```json
{
  "timestamp": "2026-09-22T12:30:00Z",
  "operations": {
    "prefill": {
      "meanMs": 450,
      "p50Ms": 440,
      "p90Ms": 490,
      "p99Ms": 510
    },
    "decode_step": {
      "meanMs": 32,
      "p50Ms": 31,
      "p90Ms": 48,
      "p99Ms": 65
    }
  }
}
```

---

## Implementation Checklist for Forge

### Week 1: Foundations (Profiling + Buffer Pool)

- [ ] Add profiler hooks to `webgpu_profiler.ts` (low risk, high visibility)
- [ ] Implement `BufferPool` allocation/release in `runAttentionStep()`
- [ ] Measure baseline p90 latency (100+ decode steps)
- [ ] **Test:** Verify p90 jitter improves by ≥50%

### Week 2: Backend Optimizations (Tokenizer + Paged Cache)

- [ ] Parallelize tokenizer URL fetch with `Promise.allSettled()`
- [ ] Add paged cache infrastructure to `LfmState`
- [ ] Implement `updateValidLength()` tracking in `runLfmStep()`
- [ ] **Test:** Verify TTFT improves by ≥30%, 8K memory usage < 10MB

### Week 3: Advanced Optimizations (INT8 Quantization)

- [ ] Add INT8 variant detection to `resolveModel()`
- [ ] Implement FP32 fallback in `loadWeights()`
- [ ] Hook `globalQuantizationCache` into weight loading
- [ ] **Test:** Verify INT8 weights load, logits match within 2%

### Week 4: Release Readiness

- [ ] End-to-end testing (all 4 optimizations enabled)
- [ ] Regression detection dashboard live
- [ ] Documentation: feature flags, rollback procedures
- [ ] **Deployment:** Phase 3.0 lands on `main` with all features enabled by
      default

---

## Rollback Plan

Each optimization can be disabled independently:

```typescript
// Disable buffer pooling
globalBufferPool.clear(); // Falls back to direct allocation

// Disable tokenizer parallel (revert function body)
// Lines 85-147 → original for-loop

// Disable paged cache
const state = createLfmState({ usePagedCache: false }); // Default

// Disable INT8 quantization
// In resolveModel(): resolved.quantizationEnabled = false;

// Disable profiling
// Omit engine.enableProfiling() call
```

---

## Performance Benchmarks (Expected Post-Phase 3.0)

**Hardware:** RTX 4080 / Mac M1 Pro (WebGPU)\
**Model:** LFM 2.5 350M

| Metric             | Baseline     | Phase 3.0    | Improvement |
| ------------------ | ------------ | ------------ | ----------- |
| TTFT               | 2.5-4s       | 1.2-1.8s     | **-45%**    |
| Decode p50 latency | 35ms         | 31ms         | **-12%**    |
| Decode p90 latency | 52ms         | 32ms         | **-38%**    |
| Decode p99 latency | 95ms         | 65ms         | **-32%**    |
| Model size         | 700MB (FP16) | 350MB (INT8) | **-50%**    |
| 8K context memory  | 32MB         | 8MB          | **-75%**    |
| Tokens per second  | 28 tok/s     | 35 tok/s     | **+25%**    |

---

## Feature Flags (Environment Variables)

```bash
# Enable all Phase 3.0 features
export ENABLE_BUFFER_POOL=1
export ENABLE_PAGED_CACHE=1
export ENABLE_INT8_QUANTIZATION=1
export ENABLE_PROFILING=1

# OR selectively disable
export ENABLE_BUFFER_POOL=0
export ENABLE_INT8_QUANTIZATION=0
```

---

## Questions for Forge?

1. **Buffer Pool:** Should we pre-allocate per-layer (32 for LFM) or globally
   shared?
2. **Paged Cache:** Should page eviction use FIFO or LRU? (Recommend LRU for
   mixed prefill/decode)
3. **INT8 Quant:** Do we need per-channel or per-tensor scaling? (Recommend
   per-channel for accuracy)
4. **Profiling:** Export to DataDog/Prometheus? (Can add exporters as separate
   PR)

**See:** `PHASE_3_0_MODIFICATIONS.json` for complete line-by-line changes.
