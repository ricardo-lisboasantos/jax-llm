# JAX-LLM Performance Optimization Initiative

**Session:** 2025-09-21 | **Scope:** All quick wins + medium + long-term
optimizations | **Goal:** 5-8x speedup + distributed support

---

## Discovery Phase Results

### Atlas: Actual Codebase State

**Current Bottlenecks:**

| Issue                                                      | Severity    | Effort         | ROI                                                  |
| ---------------------------------------------------------- | ----------- | -------------- | ---------------------------------------------------- |
| LFM cache disposal commented out (llm/lfm.ts:114-115, 126) | 🔴 Critical | 10min          | Fixes memory leak, frees per-step                    |
| UseAfterFreeError on long prompts (jax-js lib bug)         | 🔴 Critical | N/A (upstream) | Blocks 8K+ tests; need workaround                    |
| Fixed 512-token cache blocks                               | 🟡 Medium   | 4h             | Exponential growth saves 80% memory on short prompts |
| Weight offloading framework unused                         | 🟡 Medium   | 2d             | RAM/NVMe tier exists but not wired                   |
| GPU latency variance (p90 = 2-3× p50)                      | 🟡 Medium   | 1d             | Kernel dispatch jitter; needs profiling              |

**Existing Infrastructure:**

- ✅ Benchmark harness: TTFT, prefill/decode tok/s, latency percentiles
- ✅ KV cache state system: per-layer, pre-allocated blocks
- ✅ Offload modules (RAM/NVMe): stubs ready, need integration
- ✅ Multiple backends: WebGPU (default), WASM (fallback)
- ✅ Training framework: Optax-based (TrainingRunner exists)

**Inference Flow:**

```
encode() → createSession() → prefill(all_tokens) → loop: step(token) → dispose()
```

---

### Pioneer: SOTA Techniques (2024-2025)

**Recommended Stack (Phased):**

**Phase 1 — Quick Wins (1-3 days):**

1. Fix LFM disposal calls → memory leak fix
2. Dynamic cache sizing (exponential: 128→256→512) → short-prompt memory savings
3. Parallelize tokenizer fetch → network resilience

**Phase 2 — Medium (1-2 weeks):** 4. INT8 post-training quantization
(bitsandbytes-style) → 50% memory, 2x throughput 5. KV cache paging (continuous
batching pattern) → supports longer prompts 6. WebGPU kernel profiling +
persistent buffer pools → 20-30% latency reduction 7. WASM SIMD backend
activation → 3-5x CPU speedup

**Phase 3 — Long-term (3+ weeks):** 8. Distributed inference (tensor
parallelism) → 4-8x on 4 GPUs 9. LoRA/QLoRA fine-tuning adapters → 10x faster
training 10. MoE expert load-balancing (Maple model) → sparse activation wins
11. Speculative decoding (draft models) → 2-3x decode speedup

**JAX + Browser/Deno Constraints:**

- jax-js refcount bugs block long prompts (upstream issue)
- WebWorker limits (no shared memory threading)
- OPFS (persistent storage) for weight caching in browsers
- No native GPU compute API (stuck with WebGPU)
- WASM single-threaded (SIMD helps but still CPU-bound)

---

## Architecture Decisions Needed

1. **KV Cache Layout:** Stay with current per-layer block design, or move to
   paged/continuous pattern?
   - Trade: Simplicity vs flexibility for long sequences

2. **Quantization Strategy:** INT8 post-training (no retraining) vs QLoRA
   (requires fine-tuning)?
   - Trade: Speed to ship vs final quality

3. **Offload Priority:** RAM tier first (swap to NVMe), or cloud offload
   (separate inference server)?
   - Trade: Local inference simplicity vs scalability

4. **Backend Roadmap:** Stick with JAX-JS, or hedge with ONNX Runtime Web export
   path?
   - Trade: JAX training continuity vs broader ecosystem

---

## Exit Criteria

- ✅ All quick wins deployed + no regressions
- ✅ Benchmark harness reports 2-3x speedup on prefill
- ✅ Long prompts unblocked (workaround or upstream fix)
- ✅ Memory usage halved on short prompts
- ✅ Distributed inference drafted (optional for MVP)

---

## Files to Modify (Preliminary)

**Quick wins:**

- `engine/llm/lfm.ts` (uncomment disposal)
- `engine/llm/cache/*.ts` (dynamic sizing)
- `engine/runtime/runtime.ts` (tokenizer parallelization)

**Medium term:**

- `engine/llm/state/*.ts` (KV paging)
- `engine/runtime/safetensors.ts` (quantization loader)
- `engine/bench/*.ts` (new metrics)

**Long term:**

- `engine/runtime/offload/*.ts` (activate RAM/NVMe)
- `engine/runtime/training.ts` (LoRA wrapper)
- New files: `engine/runtime/distributed.ts`, `engine/runtime/speculative.ts`

---

## Next: Architect Task Breakdown

Waiting for Architect to convert this into concrete, sequential, parallelizable
tasks with dependencies.
