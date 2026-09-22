# JAX-LLM Performance Optimization: Task Breakdown

**Project:** JAX-LLM Performance Optimization\
**Session:** perf-optimization-2025\
**Created:** 2025-09-21\
**Duration:** ~3-4 weeks wall-clock (24 person-days total effort)\
**Goal:** 5-8x speedup + distributed support

---

## Executive Summary

This plan breaks down the JAX-LLM performance optimization initiative into **15
atomic tasks** across **3 phases**:

1. **Quick Wins (Day 1)** — 3 parallelizable fixes, 0 dependencies
2. **Medium (Weeks 2-3)** — 6 core optimizations with 1 integration gate
3. **Long-Term (Weeks 4+)** — 5 advanced features + validation

**Critical Path:** Quick Wins → M-03 + M-04 → M-06 (gate) → L-01 → L-02 →
Validation

**Parallelization:** Most tasks can run in parallel within their phase, reducing
24 person-days to ~3-4 weeks wall-clock with team of 3-4 engineers.

---

## Phase 1: Quick Wins (Day 1 — Parallelizable)

**Objective:** Ship immediate, high-ROI fixes with zero inter-dependencies. No
blocking. All 3 tasks can start Monday morning.

### QW-01: Fix LFM Cache Disposal Memory Leak

- **Status:** Ready to start
- **Effort:** 0.2 days (10 min)
- **Depends On:** None
- **Parallel With:** QW-02, QW-03
- **File:** `engine/llm/lfm.ts` (lines 114-115, 126)
- **Action:** Uncomment `cache.dispose()` calls
- **Acceptance:** Memory footprint stable over 100 inference steps (no per-token
  growth)
- **Verification:**
  `deno task bench -- --model=lfm --steps=100 --metrics=memory-per-step`
- **Expected Outcome:** 0% latency impact, memory leak fixed
- **Notes:** Highest ROI per minute. MUST complete before QW-02 benchmark to get
  true memory baseline.

### QW-02: Implement Dynamic KV Cache Sizing

- **Status:** Ready to start
- **Effort:** 0.5 days (4 hours)
- **Depends On:** None
- **Parallel With:** QW-01, QW-03
- **Files:** `engine/llm/cache/cache_state.ts`,
  `engine/llm/cache/block_allocator.ts`
- **Action:** Replace fixed 512-token blocks with exponential growth
  (128→256→512→...)
- **Acceptance:** Cache adapts to prompt length; 60-80% memory savings on short
  prompts (<512 tokens)
- **Verification:**
  `deno task bench -- --model=gemma --prompts=short,medium,long --metrics=memory-peak,ttft`
- **Expected Outcome:** Short prompts <200MB (was ~400MB); latency within 5% of
  QW-01 baseline
- **Notes:** Logic-only, no GPU complications. Test immediately after QW-01 for
  true memory delta.

### QW-03: Parallelize Tokenizer Fetch

- **Status:** Ready to start
- **Effort:** 0.3 days
- **Depends On:** None
- **Parallel With:** QW-01, QW-02
- **File:** `engine/runtime/runtime.ts`
- **Action:** Replace sequential tokenizer loading with `Promise.all` for
  multi-model workloads
- **Acceptance:** No race conditions; multi-model session init <100ms
- **Verification:**
  `deno task test:unit && deno task bench -- --models=lfm,gemma,maple --metric=session-init-latency`
- **Expected Outcome:** Multi-model startup ~100ms (was 250ms sequential);
  single-model no regression
- **Notes:** Low-risk async change. Independent of QW-01/02.

**Phase 1 Timeline:** All 3 tasks **in parallel on Day 1**. Completion check on
Tuesday morning.

---

## Phase 2: Medium-Term (Weeks 2-3 — Mostly Parallel)

**Objective:** Core optimizations for 2-3x speedup (quantization, profiling,
paging, WASM). Most tasks parallelizable; 1 integration gate at end.

### M-01: Profile WebGPU Kernel Dispatch & Latency Variance

- **Status:** Ready after QW completion
- **Effort:** 1.0 days
- **Depends On:** None
- **Parallel With:** M-02, M-03, M-05
- **Files:** `engine/runtime/webgpu_backend.ts`, `engine/bench/harness.ts`
- **Action:** Add WebGPU timing instrumentation; measure per-kernel latency
  distribution (p50/p90/p99)
- **Acceptance:** Identify top 3 slowest kernels; root-cause GPU jitter (p90/p99
  variance source)
- **Verification:**
  `deno task bench -- --model=gemma --metric=kernel-profile --output-json=profile.json`
- **Expected Outcome:** Profile report showing per-kernel latency, jitter <2x
  p50 identified
- **Notes:** Diagnostic task. Results inform M-02 optimization. Assign to
  **Medic**.

### M-02: Implement Persistent WebGPU Buffer Pools

- **Status:** Ready after QW completion
- **Effort:** 1.0 days
- **Depends On:** None
- **Parallel With:** M-01, M-03, M-05
- **Files:** `engine/runtime/webgpu_buffer_pool.ts`,
  `engine/runtime/webgpu_backend.ts`
- **Action:** Create reusable buffer pool with pre-allocation & LRU eviction;
  avoid repeated alloc/dealloc per inference step
- **Acceptance:** p90 latency reduced to <1.5x p50 (from 2-3x); buffer pool hit
  rate >90%
- **Verification:**
  `deno task bench -- --model=gemma --steps=100 --metric=decode-latency-p90 --compare=before:m-02`
- **Expected Outcome:** 20-30% latency reduction; stable p90 across all models
- **Notes:** Complements M-01 profiling. Can parallelize M-01. Assign to
  **Forge**.

### M-03: Create INT8 Quantization Loader

- **Status:** Ready after QW completion
- **Effort:** 1.0 days
- **Depends On:** None
- **Parallel With:** M-01, M-02, M-05
- **Files:** `engine/runtime/quantization_loader.ts`,
  `engine/runtime/safetensors.ts`
- **Action:** Implement INT8 weight loader (post-training quantization, no
  retraining); support scales/zeros format
- **Acceptance:** All models load correctly quantized; no accuracy drop >0.5%;
  memory 50% reduction
- **Verification:**
  `deno task test:unit -- quantization && deno task bench -- --model=gemma --quantize=int8 --metric=throughput,accuracy`
- **Expected Outcome:** Throughput 1.8-2.2x baseline; memory 50% reduction;
  perplexity delta <0.5%
- **Notes:** Orthogonal to M-02/M-01. Pre-quantized weights assumed to exist.
  Assign to **Forge**.

### M-04: Implement KV Cache Paging (Continuous Batching)

- **Status:** Ready after QW-02 (knowledge of dynamic cache design)
- **Effort:** 1.5 days
- **Depends On:** None (but benefits from QW-02 completion)
- **Parallel With:** M-01, M-02, M-03
- **Files:** `engine/llm/state/kv_paging.ts`,
  `engine/llm/cache/paging_allocator.ts`, `engine/llm/gemma.ts`,
  `engine/llm/lfm.ts`
- **Action:** Replace per-layer KV blocks with paged continuous buffer; support
  on-demand allocation
- **Acceptance:** 8K+ prompts unblocked (workaround for UseAfterFreeError);
  memory utilization >85%
- **Verification:**
  `deno task test:unit -- kv-paging && deno task bench -- --model=gemma --prompt-length=2048,4096,8192 --metric=throughput`
- **Expected Outcome:** 8K+ prompts work without crash; memory overhead <5% vs
  dynamic sizing; throughput stable
- **Notes:** Major architectural change. Integration test required with M-03.
  Assign to **Forge**.

### M-05: Activate WASM SIMD Backend

- **Status:** Ready after QW completion
- **Effort:** 1.0 days
- **Depends On:** None
- **Parallel With:** M-01, M-02, M-03, M-04
- **Files:** `engine/runtime/wasm_backend.ts`, `engine/runtime/runtime.ts`
- **Action:** Enable WASM SIMD128 instructions in Deno; switch backend routing
  to prefer SIMD for prefill
- **Acceptance:** WASM prefill 3-5x faster; GPU backend unchanged; no
  regressions
- **Verification:**
  `deno task bench -- --model=gemma --backend=wasm --metric=prefill-tok-per-s`
- **Expected Outcome:** CPU fallback now viable for inference; prefill 3-5x
  speedup on WASM
- **Notes:** Platform-specific. Requires Deno WASM SIMD support (check version).
  Assign to **Forge**.

### **M-06: Integration Gate — Quantization + Paging Cross-Model Benchmark** ⚠️

- **Status:** Ready after M-03 + M-04 complete
- **Effort:** 0.5 days
- **Depends On:** **M-03, M-04** (blocking)
- **Parallel With:** None (sequential gate)
- **Files:** `engine/bench/harness.ts`
- **Action:** Run comprehensive benchmark across all models with quantization +
  paging enabled; verify no unexpected interactions
- **Acceptance:** All combinations stable; memory + latency within expected
  cumulative improvements
- **Verification:**
  `deno task bench -- --all-models --quantize --paging --prompts=short,medium,long --output-json=combined_results.json`
- **Expected Outcome:** Benchmark matrix complete; no race conditions; safe to
  integrate into long-term features
- **Notes:** **GATEKEEPER.** Must pass before releasing M-03 + M-04. Assign to
  **Guardian**.

**Phase 2 Timeline:**

- **Week 2 (Mon-Fri):** M-01, M-02, M-03, M-04, M-05 all start Monday in
  parallel
- **End of Week 2 (Fri):** All except M-06 should be feature-complete
- **Week 3 (Mon):** M-06 validation runs Monday-Tuesday; if pass, clear for
  long-term; if fail, fix + retest

---

## Phase 3: Long-Term (Weeks 4+ — Sequential with Parallels)

**Objective:** Advanced features (distributed inference, fine-tuning, MoE,
speculative decoding). More dependencies; some parallelization.

### **L-01: Implement UseAfterFreeError Workaround** 🔴 Critical Path

- **Status:** Ready after M-06 gate passes
- **Effort:** 0.5 days
- **Depends On:** None
- **Parallel With:** L-02, L-03
- **File:** `engine/runtime/jaxjs_wrapper.ts`
- **Action:** Add reference counting workaround or buffer copying in jax-js
  layer to prevent UseAfterFreeError on 8K+ token sequences
- **Acceptance:** 8K+ prompts complete without crash; memory overhead <10%
- **Verification:**
  `deno task bench -- --model=gemma --prompt-length=8192,16384 --verify-no-crash`
- **Expected Outcome:** Long-prompt inference unblocked; UseAfterFree eliminated
- **Notes:** Upstream issue workaround. BLOCKS L-02. High priority. Assign to
  **Forge**.

### **L-02: Implement Distributed Tensor Parallelism (Base)** 🔴 Critical Path

- **Status:** Ready after L-01 complete
- **Effort:** 2.0 days
- **Depends On:** **L-01** (must complete first)
- **Parallel With:** L-03
- **Files:** `engine/runtime/distributed.ts`, `engine/runtime/communication.ts`,
  `engine/llm/gemma.ts`, `engine/llm/lfm.ts`
- **Action:** Distribute model weights across 2-4 GPUs via tensor parallelism;
  implement allreduce/allgather
- **Acceptance:** Throughput scales 1.8-3.5x on 2-4 GPUs; no communication
  bottleneck >20% overhead
- **Verification:**
  `deno task test:unit -- distributed && deno task bench -- --devices=2,4 --metric=throughput-scaling`
- **Expected Outcome:** 4-8x speedup potential on multi-GPU; inference latency
  20-30% increase per GPU but throughput multiplies
- **Notes:** High complexity. REQUIRES L-01. Requires multi-GPU testing
  environment. Assign to **Forge**.

### L-03: Implement LoRA/QLoRA Fine-Tuning Adapters

- **Status:** Ready after M-06 gate passes
- **Effort:** 1.0 days
- **Depends On:** M-03 (quantization loader)
- **Parallel With:** L-02
- **Files:** `engine/runtime/lora_adapter.ts`, `engine/runtime/training.ts`
- **Action:** Wrap quantized models with LoRA low-rank adapters; enable 10x
  faster fine-tuning on INT8 weights
- **Acceptance:** Fine-tuning 10x faster; memory overhead <15%; accuracy within
  0.5% of full tuning
- **Verification:**
  `deno task test:unit -- lora && deno task bench -- --model=gemma --lora=true --metric=training-throughput`
- **Expected Outcome:** Fine-tuning becomes practical; 10x speedup enables
  large-scale adaptation
- **Notes:** Orthogonal to L-02. Can parallelize with L-02 after M-03 complete.
  Assign to **Forge**.

### L-04: Implement MoE Expert Load-Balancing (Maple Model)

- **Status:** Ready after L-01 complete
- **Effort:** 1.0 days
- **Depends On:** L-01 (error fixes)
- **Parallel With:** L-03
- **Files:** `engine/runtime/moe_router.ts`, `engine/llm/maple.ts`
- **Action:** Activate expert load-balancing for Maple MoE model; route tokens
  to least-loaded experts
- **Acceptance:** Expert load within 10% of ideal; sparse activation 2-3x
  speedup vs dense
- **Verification:**
  `deno task bench -- --model=maple --metric=expert-load-distribution,throughput`
- **Expected Outcome:** Maple inference 2-3x faster via sparse activation
- **Notes:** Model-specific. Depends on L-01. Independent of L-02/L-03. Assign
  to **Forge**.

### **L-05: Implement Speculative Decoding with Draft Models** 🔴 Critical Path

- **Status:** Ready after L-02 complete
- **Effort:** 1.0 days
- **Depends On:** **L-02** (distributed backend helps with dual-model loading)
- **Parallel With:** None (sequential gate)
- **Files:** `engine/runtime/speculative_decoding.ts`,
  `engine/runtime/runtime.ts`
- **Action:** Use smaller draft model to generate k tokens, verify with main
  model; 2-3x decode speedup
- **Acceptance:** Decode 2-3x faster; accuracy unchanged; no false token accepts
- **Verification:**
  `deno task bench -- --model=gemma --speculative=true --metric=decode-latency`
- **Expected Outcome:** Decode becomes bottleneck elimination; 2-3x final
  speedup
- **Notes:** Depends on L-02 (concurrent inference). Requires draft model
  weights. Assign to **Forge**.

### **Validation-01: End-to-End Performance Validation Suite** 🎯

- **Status:** Ready after all long-term tasks complete
- **Effort:** 0.5 days
- **Depends On:** L-02, L-03, L-04, L-05
- **Parallel With:** None (final gatekeeper)
- **Files:** `engine/bench/harness.ts`, `engine/bench/validation.ts`
- **Action:** Comprehensive benchmark: all models, all optimizations enabled, no
  regressions
- **Acceptance:** Overall speedup 5-8x; all features composable; ready for
  production
- **Verification:**
  `deno task bench -- --all-features --all-models --output-report=final_validation.md`
- **Expected Outcome:** Release-ready performance report; 5-8x speedup validated
- **Notes:** Final gatekeeper. Ships the release. Assign to **Guardian**.

**Phase 3 Timeline:**

- **Week 4 (Mon-Tue):** L-01 develops (0.5d); ready Wed
- **Week 4 (Wed+):** L-02 starts Wednesday after L-01; L-03 starts independently
  Wednesday; L-04 starts after L-01
- **Week 4 (Fri):** L-02 ~80% done; L-03 done; L-04 done; L-05 queued
- **Week 5 (Mon-Tue):** L-02 finishes; L-05 starts immediately after; parallel
  with Validation
- **Week 5 (Wed-Fri):** Validation runs comprehensive suite; results ready Fri

---

## Dependency Graph

```
Quick Wins (Day 1)
├─ QW-01 ✅ (10 min)
├─ QW-02 ✅ (4h)
└─ QW-03 ✅ (3h)
    ↓ (all clear by Day 1 end)
    
Medium Phase (Week 2-3)
├─ M-01 (Profiling) ✅ 1d
├─ M-02 (Buffer Pools) ✅ 1d
├─ M-03 (INT8 Quantization) ✅ 1d
├─ M-04 (KV Paging) ✅ 1.5d
├─ M-05 (WASM SIMD) ✅ 1d
│  ↓ (all complete by end Week 2)
└─ M-06 🔴 GATE (M-03 + M-04 required) ✅ 0.5d
    ↓ (gate must pass)
    
Long-Term Phase (Week 4+)
├─ L-01 (UseAfterFree Workaround) 🔴 ✅ 0.5d
│  ├─ L-02 (Distributed) → MUST WAIT FOR L-01 ✅ 2d
│  └─ L-04 (MoE Load-Balancing) ✅ 1d
├─ L-03 (LoRA Adapters, depends M-03) ✅ 1d
└─ L-05 (Speculative Decode, depends L-02) 🔴 ✅ 1d
    ↓ (all complete by Week 5)
    
Final Validation 🎯
└─ Validation-01 (depends L-02, L-03, L-04, L-05) ✅ 0.5d
```

---

## Parallel Execution Strategy

### Week 1: Quick Wins (Day 1)

- **Team A:** QW-01 (LFM disposal)
- **Team B:** QW-02 (dynamic cache)
- **Team C:** QW-03 (tokenizer fetch)
- **Completion:** Tuesday EOD, all clear for Medium phase

### Week 2: Medium Phase (Day 3-7)

- **Team A:** M-01 (Profiling) + M-02 (Buffer Pools)
- **Team B:** M-03 (Quantization) + M-04 (KV Paging)
- **Team C:** M-05 (WASM SIMD)
- **Medic:** M-01 profiling depth work
- **Guardian:** Write M-06 benchmark harness in parallel
- **Completion:** Friday EOD, M-06 validation scheduled Mon morning

### Week 3: Medium Gate + Long-Term Start

- **Mon-Tue:** M-06 gate validation; if pass → clear for long-term
- **Wed-Fri:**
  - L-01 (UseAfterFree) — Team A
  - L-03 (LoRA) — Team B (parallel, different codebase)
  - L-04 (MoE) — Team C
  - L-02 prep — Forge (design after L-01 complete)

### Week 4-5: Long-Term Features

- **Week 4:** L-02 (Distributed, blocking) + L-03/L-04 (parallel)
- **Week 5:** L-05 (Speculative, waits for L-02) + Validation
- **Completion:** Fri EOD, release-ready

---

## Task Dependencies Matrix

| Task       | Phase  | Effort | Depends On             | Can Parallel     | Assign To |
| ---------- | ------ | ------ | ---------------------- | ---------------- | --------- |
| QW-01      | Quick  | 0.2d   | —                      | QW-02, QW-03     | Forge     |
| QW-02      | Quick  | 0.5d   | —                      | QW-01, QW-03     | Forge     |
| QW-03      | Quick  | 0.3d   | —                      | QW-01, QW-02     | Forge     |
| M-01       | Medium | 1.0d   | —                      | M-02, M-03, M-05 | Medic     |
| M-02       | Medium | 1.0d   | —                      | M-01, M-03, M-05 | Forge     |
| M-03       | Medium | 1.0d   | —                      | M-02, M-04, M-05 | Forge     |
| M-04       | Medium | 1.5d   | —                      | M-02, M-03, M-05 | Forge     |
| M-05       | Medium | 1.0d   | —                      | M-01, M-02, M-03 | Forge     |
| **M-06**   | Medium | 0.5d   | **M-03, M-04**         | —                | Guardian  |
| **L-01**   | Long   | 0.5d   | —                      | L-02, L-04       | Forge     |
| **L-02**   | Long   | 2.0d   | **L-01**               | L-03             | Forge     |
| L-03       | Long   | 1.0d   | M-03                   | L-02             | Forge     |
| L-04       | Long   | 1.0d   | L-01                   | L-03             | Forge     |
| **L-05**   | Long   | 1.0d   | **L-02**               | —                | Forge     |
| Validation | Long   | 0.5d   | L-02, L-03, L-04, L-05 | —                | Guardian  |

---

## Risk Matrix & Mitigations

| Risk                                  | Severity    | Mitigation                                                                                          |
| ------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------- |
| **UseAfterFreeError upstream bug**    | 🔴 Critical | L-01 workaround ready; test early (M-04); escalate to jax-js if workaround fails                    |
| **Multi-GPU coordination complexity** | 🟡 High     | L-02 design review pre-implementation; prototype allreduce first; test on 2 GPUs before 4           |
| **Quantization accuracy loss**        | 🟡 High     | M-03 must validate on full test suite; perplexity delta threshold <0.5%                             |
| **Buffer pool memory fragmentation**  | 🟡 Medium   | M-02 LRU eviction policy tested; monitor peak memory during benchmark                               |
| **WASM SIMD platform variance**       | 🟡 Medium   | M-05 validation on multiple Deno versions; fallback to non-SIMD if unavailable                      |
| **Long prompt OOM on paging**         | 🟡 Medium   | M-04 memory utilization test on extreme lengths (16K tokens); adjust allocation algorithm if needed |
| **Speculative decode false accepts**  | 🟡 Medium   | L-05 verification logic unit-tested extensively; semantic correctness critical                      |

---

## Success Criteria

### Quick Wins

- ✅ Memory leak fixed (QW-01)
- ✅ Dynamic cache proven stable (QW-02)
- ✅ Tokenizer fetch parallelized (QW-03)
- **Checkpoint:** 0 regressions, baseline established for medium phase

### Medium Phase

- ✅ Profiling shows jitter source (M-01)
- ✅ Buffer pools reduce p90 latency by 20-30% (M-02)
- ✅ INT8 quantization 50% memory reduction, 2x throughput (M-03)
- ✅ Long prompts (8K+) unblocked via paging (M-04)
- ✅ WASM SIMD 3-5x CPU prefill speedup (M-05)
- ✅ M-06 gate passes: all combinations stable
- **Checkpoint:** 2-3x speedup on prefill, memory halved on short prompts

### Long-Term Phase

- ✅ UseAfterFreeError workaround proven (L-01)
- ✅ Tensor parallelism 4-8x on multi-GPU (L-02)
- ✅ LoRA adapters 10x fine-tuning speedup (L-03)
- ✅ MoE load-balancing 2-3x sparse speedup (L-04)
- ✅ Speculative decoding 2-3x decode speedup (L-05)
- ✅ Validation suite: 5-8x overall speedup, no regressions
- **Checkpoint:** Release-ready; ship production

---

## Files to Modify (Complete List)

### Quick Wins

- `engine/llm/lfm.ts` (uncomment disposal)
- `engine/llm/cache/cache_state.ts` (dynamic sizing)
- `engine/llm/cache/block_allocator.ts` (allocation logic)
- `engine/runtime/runtime.ts` (tokenizer parallelization)

### Medium Phase

- `engine/runtime/webgpu_backend.ts` (profiling, buffer pools)
- `engine/runtime/webgpu_buffer_pool.ts` (NEW)
- `engine/runtime/quantization_loader.ts` (NEW)
- `engine/runtime/safetensors.ts` (quantization support)
- `engine/llm/state/kv_paging.ts` (NEW)
- `engine/llm/cache/paging_allocator.ts` (NEW)
- `engine/llm/gemma.ts` (KV paging integration)
- `engine/llm/lfm.ts` (KV paging integration)
- `engine/runtime/wasm_backend.ts` (SIMD activation)
- `engine/bench/harness.ts` (new metrics, M-06 gate)

### Long-Term Phase

- `engine/runtime/jaxjs_wrapper.ts` (UseAfterFree workaround)
- `engine/runtime/distributed.ts` (NEW)
- `engine/runtime/communication.ts` (NEW)
- `engine/runtime/lora_adapter.ts` (NEW)
- `engine/runtime/training.ts` (LoRA integration)
- `engine/runtime/moe_router.ts` (NEW)
- `engine/llm/maple.ts` (MoE integration)
- `engine/runtime/speculative_decoding.ts` (NEW)
- `engine/bench/validation.ts` (validation suite)

**Total new files:** 9\
**Total modified files:** 22

---

## Agent Assignments

| Agent        | Role                | Tasks                                                       |
| ------------ | ------------------- | ----------------------------------------------------------- |
| **Forge**    | Implementation      | QW-01/02/03, M-02/03/04/05, L-01/02/03/04/05                |
| **Guardian** | QA/Integration      | M-06, Validation-01                                         |
| **Medic**    | Profiling/Debugging | M-01 (deep profiling work)                                  |
| **Pixel**    | Dashboards/Viz      | Benchmark visualization (optional)                          |
| **Scribe**   | Documentation       | README updates, performance report (included in Validation) |

**Recommended Team:** 3 Forge (parallel), 1 Guardian (quality gates), 1 Medic
(optional, profiling depth)

---

## Timeline & Milestones

```
Week 1 (Day 1-2)
  Mon   QW-01, QW-02, QW-03 all start (parallel)
  Tue   All QW complete ✅
  
Week 2 (Day 3-7)
  Mon   M-01/02/03/04/05 all start (parallel)
  Fri   M-01-05 feature-complete; M-06 scheduled
  
Week 3 (Day 8-12)
  Mon   M-06 validation runs; gate decision
  Wed   L-01, L-03, L-04 start (after gate passes)
  Fri   L-01/03/04 complete; L-02 design finalized
  
Week 4 (Day 13-19)
  Wed   L-02 starts; L-05 queued
  Fri   L-02 80% complete
  
Week 5 (Day 20-24)
  Mon   L-02 finishes; L-05 starts
  Wed   L-05 completes; Validation starts
  Fri   Validation complete ✅ Release-ready
  
TOTAL: ~3.5 weeks wall-clock
```

---

## Next Steps

1. **Verify context & dependencies** — Review this plan with Chief & team
2. **Assign engineers** — Allocate Forge × 3, Guardian × 1, Medic × 1
3. **Create tracking** — Use task.json for issue tracking (GitHub/Linear)
4. **Week 1 Kickoff** — QW tasks start Monday EOD with acceptance tests ready
5. **Weekly Sync** — Report progress, adjust parallelization if bottlenecks
   emerge

---

**Plan Owner:** Architect\
**Status:** Ready for Chief approval\
**Next:** Chief dispatches tasks to Forge, Guardian, Medic
