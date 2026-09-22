## Codebase Performance Baseline

### Benchmarking Infrastructure

**Metrics tracked:**
- **TTFT (Time to First Token)**: Wall time from request start to first generated token (covers encode + prefill + first sample)
- **Prefill throughput**: Input-context tokens/second during full-prompt processing
- **Decode throughput**: Generated tokens/second during autoregressive `step()` loop
- **Per-token latency distribution**: p50, p90, min, max, mean (for variance analysis)
- **Encode throughput**: Raw tokenization tokens/second (separate from model inference)

**Test scope:**
- `benchmarkEncode()`: tokenizer-only (no model init)
- `benchmarkPrefillFn()`: session-level prefill (1 warmup, 3 iterations)
- `benchmarkDecodeSteps()`: per-token latency over a token sweep (default 8 steps)
- `benchmarkChatEngine()`: end-to-end from prompt to generation (TTFT + full reply)

**Prompt sizes tested:**
- **Short**: ~100 chars (~30 tokens) — smoke test baseline
- **Medium**: ~2K chars (~500 tokens) — realistic chat turn
- **Long**: ~8K chars (~2K tokens) — prefill stress (see known issues)

**File structure:**
- `engine/bench/types.ts` — metric type definitions (LatencyStats, GenerationMetrics, etc.)
- `engine/bench/measure.ts` — timing + statistics helpers (tokensPerSecond, summarizeLatencies)
- `engine/bench/harness.ts` — benchmark harness (6 entry points from encode to full generation)
- `engine/bench/fixtures.ts` — standard prompts + console formatting
- `tests/bench_test.ts` — live model benchmark (TTFT + throughput sweep, skips without network/GPU)

---

### Inference Flow (prefill → step → dispose)

```
1. ENCODE PHASE
   ├─ tokenizer.encode(text)
   └─ Returns: number[] of token IDs

2. SESSION CREATION
   └─ runtime.createSession()
      └─ Allocates: GemmaState or LfmState with KV cache

3. PREFILL PASS
   └─ session.prefill(np.array(tokenIds))
      ├─ Runs: full prompt through all transformer layers
      ├─ Outputs: logits for last token position
      ├─ Updates: state.position = promptLength
      ├─ Allocates: KV cache [capacity, num_kv_heads, head_dim]
      └─ Memory: cache is contiguous, allocated upfront (512-token blocks)

4. DECODE LOOP (per token)
   └─ session.step(tokenId)
      ├─ Runs: single-token through all layers (cache-aided attention)
      ├─ Outputs: logits for next position
      ├─ Updates: state.position++, appends to KV cache
      └─ Latency: typically 10-50ms per token (p50), 20-100ms (p90)

5. CLEANUP
   └─ session.dispose()
      ├─ Disposes: state via tree.dispose(state)
      ├─ Frees: KV cache tensors
      └─ Risk: UseAfterFreeError on large caches (see known issues)
```

**Memory patterns:**
- **KV cache size**: `[capacity, numKeyValueHeads, headDim]` per layer
  - Gemma: 32 layers × [512, 8, 64] × 2 (key+value) = ~67 MB per session (float32)
  - LFM: Mixed conv+attention; attention caches ~40 MB
- **Capacity growth**: Rounded to 512-token blocks; `ensureStateCapacity()` pads on-demand
- **Cache disposal**: Gemma disposes at prefill start; LFM has commented-out disposal (lines 114-115, 126)

---

### Bottlenecks Identified

| Bottleneck | Impact | Root Cause | Location |
|------------|--------|-----------|----------|
| **LFM cache disposal disabled** | Memory fragmentation, potential leaks | Commented-out `.dispose()` calls in step/prefill | `engine/llm/lfm.ts:114-115, 126` |
| **Long prompt refcount bug** | Crashes on 8K+ char prompts | jax-js UseAfterFreeError on 3072×8×64 float32 cache | jax-js internals; blocked by refcount issue |
| **KV cache allocation overhead** | Wasted memory on short prompts | Fixed 512-token block size; no dynamic downsizing | `engine/llm/state/*.ts` + `engine/llm/cache/*.ts` |
| **Per-token latency variance** | p90 2-3× p50 latency | GPU scheduling + lazy execution overhead | WebGPU kernel dispatch timing |
| **No weight offloading** | Memory ceiling at VRAM size | Offload stubs exist but unimplemented | `engine/runtime/offload/ram.ts`, `nvme.ts` |
| **Tokenizer fetch latency** | Blocks load on first run | Multi-URL fallback with N sequential fetches | `engine/runtime/runtime.ts:95-141` |
| **Single-session model lifecycle** | No re-prefill support | Each ChatEngine session is create→prefill→step→dispose | `engine/chat/chat_engine.ts:157-182` |

---

### Backends

**WebGPU (default)**
- Device init: `defaultDevice("webgpu")` → initializes GPU context
- Execution model: JIT compilation via @jax-js/jax (likely tiling + kernel fusion)
- Optimizations: Lazy execution with `.data()` forcing evaluation; GPU-resident weights
- Limitations: Requires WebGPU-capable runtime; no explicit memory pooling observed

**WASM (fallback)**
- Device init: `defaultDevice("wasm")` → falls back to CPU compute
- Execution model: SIMD-optimized CPU kernels (generic JAX codegen)
- Optimizations: Auto-vectorization; no GPU memory overhead
- Trade-off: ~10-50× slower than WebGPU on typical hardware

**Both backends share:**
- Same prefill/step/dispose API surface
- Same KV cache representation (np.Array)
- JAX's reference counting for automatic cleanup (tree.dispose() is explicit wrapper)

---

### Known Issues

1. **UseAfterFreeError on long prompts** (Critical)
   - Trigger: 8K+ char prompt (~2K tokens) → 3072-dim KV cache
   - Error: `UseAfterFreeError` during `session.dispose()`
   - Root cause: jax-js reference counting bug with large multidimensional arrays
   - Workaround: Benchmark limits to short + medium prompts only
   - File: `tests/bench_test.ts:11-14` (documented in module docstring)

2. **LFM cache disposal disabled** (Medium)
   - Location: `engine/llm/lfm.ts:114-115, 126`
   - Lines: Attention cache disposal commented out in step() and prefill()
   - Impact: Old cache tensors not freed → memory accumulation over many steps
   - Note: Gemma properly disposes at line 42-43

3. **No weight offloading** (Medium)
   - Files: `engine/runtime/offload/ram.ts`, `nvme.ts`
   - Status: Stubs only; `RamOffload` and `NvmeOffload` classes exist but unused
   - Impact: Large models may OOM; no CPU←→GPU paging

4. **Fixed cache block size** (Low)
   - Block size: 512 tokens
   - Issue: 30-token prompt allocates 512 slots → 94% wasted memory
   - Mitigation: Acceptable for typical chat (500+ tokens), suboptimal for batch inference

---

### Architecture Notes for Optimization

**Quick wins:**
1. **Fix LFM cache disposal** (10-line change)
   - Uncomment lines 114-115 and 126 in `engine/llm/lfm.ts`
   - Matches Gemma pattern for consistency
   - Frees per-step memory → enables longer generations

2. **Dynamic cache allocation**
   - Replace fixed 512-token blocks with exponential growth (128 → 256 → 512)
   - Reduces short-prompt waste by ~80%
   - Changes: `roundCacheCapacity()` + state init

3. **Tokenizer fetch retry** (resilience, not perf)
   - Parallelize URL candidates instead of sequential fallback
   - Adds `Promise.all()` with `Promise.race()` cancellation

**Medium-term improvements:**
4. **Weight offloading** (RAM/NVMe)
   - Activate `RamOffload` / `NvmeOffload` for models > available VRAM
   - Implement paging: LRU layer eviction on-demand
   - File: `engine/runtime/offload/*.ts` (framework exists, wiring needed)

5. **Multi-prefill sessions**
   - Support re-prefill on same session (e.g., for multi-turn re-context)
   - Adds state reset before prefill; reduces allocation churn

6. **Batch inference**
   - Stack multiple sessions → vectorized prefill + decode
   - Requires: state merging (outer product of caches)

**Deep optimizations:**
7. **Fix jax-js UseAfterFreeError**
   - File issue upstream or work around with smaller cache dimensions
   - Enables long-prompt benchmark sweep (currently blocked)

8. **Per-backend tuning**
   - WebGPU: Profile kernel dispatch; experiment with grid-strided loops
   - WASM: SIMD scheduling; consider wasm-opt post-processing

---

### Current Benchmark Baselines (from bench_test.ts)

**Expected results** (lfm2.5-350m, short prompt):
- Encode: ~1000+ tok/s (tokenizer-only)
- Prefill (32 tokens): ~50-200 tok/s (backend-dependent)
- Decode (8 steps): ~20-50 tok/s (p50 ~15-30ms/tok)
- TTFT: ~500-1000ms (covers all init overhead)

**Why variance?**
- First inference warm-up (JIT compilation not included in reported times)
- GPU scheduling jitter (p90 typically 2-3× p50)
- Browser/Deno GC pauses during `.data()` fetch
- Network-dependent: tokenizer/weight download on first run

---

### Files Priority

**Read first:**
- `engine/bench/types.ts` — understand metric definitions
- `tests/bench_test.ts` — see known issues + current test coverage
- `engine/runtime/runtime.ts` — device/tokenizer/weight loading
- `engine/llm/lfm.ts` — check commented-out disposal calls
- `engine/chat/chat_engine.ts` — end-to-end flow

**Deep dives:**
- `engine/llm/state/*.ts` — KV cache allocation strategies
- `engine/llm/cache/*.ts` — block sizing + capacity growth
- `engine/runtime/offload/*.ts` — unused weight offloading framework
- `deno.json` — task definitions for bench/test runs
