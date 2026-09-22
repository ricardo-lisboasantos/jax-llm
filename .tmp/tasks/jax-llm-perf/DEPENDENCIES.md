# Detailed Dependency Graph & Sequencing Guide

**Version:** 1.0\
**Date:** 2025-09-21\
**Purpose:** Architect-to-Engineer handoff; shows exact blocking dependencies
and parallel opportunities

---

## Dependency Network (Full Detail)

### Phase 1: Quick Wins (Day 1)

```
                    ┌─────────────┐
                    │   QW-01     │ (LFM disposal)
                    │ 0.2d        │
                    │  Forge      │
                    └─────────────┘
                           ↓ (no blocking)
                           ↓
    ┌──────────────────────┼──────────────────────┐
    ↓                      ↓                      ↓
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   QW-02     │     │   QW-03     │     │   QW-02     │ (runs parallel)
│ Dynamic     │     │ Tokenizer   │     │ Cache       │
│ Cache       │     │ Parallelize │     │ Sizing      │
│ 0.5d Forge  │     │ 0.3d Forge  │     │ (see left)  │
└─────────────┘     └─────────────┘     └─────────────┘
    ↓                      ↓                      ↓
    └──────────────────────┼──────────────────────┘
                           ↓ (END: Day 1, all pass ✅)
                    
                  ┌─────────────────┐
                  │  Checkpoint:    │
                  │  Baseline Est.  │
                  │  Memory, Lat.   │
                  │  All QW pass ✅ │
                  └─────────────────┘
```

**Sequencing Rule:** Start all 3 Monday 9am. No ordering. Parallel OK = 1d
wall-clock.

**Handoff to Medium:** All QW must PASS before M-01-05 start.

---

### Phase 2: Medium (Week 2-3)

```
QW COMPLETE ✅
    ↓
    ├─────────────────────────────────────────────────────────┐
    │                                                         │
    v                                                         │
┌──────────────┐                                              │
│   M-01       │ (Profiling, independent)                    │
│ WebGPU Prof  │                                              │
│ 1d, Medic    │                                              │
└──────────────┘                                              │
    │                                                         │
    ├─→ informs optimization (M-02)                          │
    │                                                         │
    └─→ but NOT blocking (M-02 can start in parallel)       │
            ↓                                                 │
            ├─────────────────────────────────┐             │
            │                                 │             │
    ┌──────────────┐            ┌──────────────┐            │
    │   M-02       │  parallel  │   M-03       │            │
    │ Buffer Pool  │            │ INT8 Quant   │            │
    │ 1d, Forge    │            │ 1d, Forge    │            │
    └──────────────┘            └──────────────┘            │
            │                                 │             │
            │                                 │             │
    ┌──────────────┐            ┌──────────────┐            │
    │   M-04       │            │   M-05       │            │
    │ KV Paging    │  parallel  │ WASM SIMD    │            │
    │ 1.5d, Forge  │            │ 1d, Forge    │            │
    └──────────────┘            └──────────────┘            │
            │                                 │             │
            └────────────────┬─────────────────┘             │
                             │                              │
                    ┌────────v────────┐                     │
                    │   [EOW2: All     │                     │
                    │    M-01 to M-05  │                     │
                    │    feature done] │                     │
                    └────────┬────────┘                      │
                             │                              │
                             │ (critical dependency)        │
                             │                              │
                    ┌────────v────────┐                     │
                    │   **M-06 GATE** 🔴 │                  │
                    │ M-03 + M-04      │                    │
                    │ required         │                    │
                    │ 0.5d, Guardian   │                    │
                    └────────┬────────┘                      │
                             │                              │
                    ┌────────v────────┐                    │
                    │   M-06 Result   │                    │
                    │  PASS → OK ✅   │                    │
                    │  FAIL → Fix+Re  │                    │
                    └────────┬────────┘                    │
                             │                            │
        [If FAIL, loop back to M-03+M-04 fixes]          │
        [If PASS, clear for Long-Term]                    │
                             │                            │
                             └────────────────────────────┘
                                      ↓
                            [Week 3, ready for L-01]
```

**Sequencing Rules:**

1. QW must complete before ANY Medium task starts
2. M-01, M-02, M-03, M-04, M-05 can all start **simultaneously** (no inter-deps)
3. M-06 **MUST WAIT** for M-03 + M-04 to feature-complete
4. M-06 is **GATEKEEPER** — if it fails, iterate on M-03 + M-04
5. Only after M-06 **PASSES** can long-term start

**Parallelization:** 5 concurrent developers (Forge×3, Medic×1, Guardian prep
M-06)\
**Wall-clock:** Week 2 Mon start → Week 3 Wed M-06 passes (can do earlier if
M-03+M-04 finish Fri of Week 2)

---

### Phase 3: Long-Term (Weeks 4-5)

```
M-06 PASSES ✅
    │
    ├─ L-01: UseAfterFree Workaround (independent start)
    │       0.5d, Forge
    │       ├─→ Entry: UseAfterFreeError on 8K+ tokens
    │       ├─→ Exit: 8K+ prompts work
    │       │
    │       ├─→ BLOCKS L-02 ⚠️ (CRITICAL PATH)
    │       │
    │       └─→ Parallel: L-03 can start (different codebase)
    │
    ├─ L-03: LoRA/QLoRA Adapters (depends M-03)
    │       1d, Forge
    │       ├─→ Entry: Quantization loader (M-03) complete
    │       ├─→ Exit: Fine-tuning 10x faster
    │       │
    │       ├─→ Can start Week 3 if M-03 done
    │       │
    │       └─→ Parallel: L-02 (once L-01 done)
    │
    └─ L-04: MoE Load-Balancing (depends L-01)
           1d, Forge
           ├─→ Entry: UseAfterFree fixed (L-01)
           ├─→ Exit: Maple 2-3x faster
           │
           └─→ Parallel: L-02 (once L-01 done)


AFTER L-01 COMPLETE (week 4 Wed)
    │
    └─ L-02: Distributed Inference (BLOCKS L-05)
           2d, Forge
           ├─→ Entry: UseAfterFree fixed (L-01)
           ├─→ Exit: Tensor parallelism on 2-4 GPUs
           │
           ├─→ CRITICAL PATH: Blocks L-05
           │
           └─→ Design review + allreduce prototype ASAP
                (can parallelize with L-01 design work)


AFTER L-02 COMPLETE (week 5 Mon)
    │
    └─ L-05: Speculative Decoding (final feature)
           1d, Forge
           ├─→ Entry: L-02 complete
           ├─→ Exit: Decode 2-3x faster
           │
           └─→ Concurrent: Validation can start
                (runs during L-05 implementation)


FINAL GATE: Validation-01
           0.5d, Guardian
           ├─→ Entry: L-02, L-03, L-04, L-05 all done
           ├─→ Exit: 5-8x speedup verified, release-ready
           │
           └─→ Result: ✅ SHIP or 🔴 FIX+RETEST
```

**Sequencing Rules:**

1. **L-01 is GATEKEEPER for L-02 + L-04**
   - L-01 must complete before L-02 starts
   - L-03 can run in parallel (different codebase)
   - L-04 needs L-01 but can overlap implementation with L-01 design

2. **L-02 is GATEKEEPER for L-05**
   - L-05 waits for L-02 to complete
   - Can prototype speculative logic in parallel, but not integrate until L-02
     stable

3. **Validation-01 needs all long-term done**
   - Can start once L-02 completes (if L-03/L-04 taking time)
   - Iterates if any regressions found

**Critical Path Timeline:**

- Week 3 Wed: L-01 starts (0.5d) → ready Thu morning
- Week 4 Wed: L-02 starts (2d) → ready Fri or Mon Week 5
- Week 5 Mon-Tue: L-05 starts (1d) → ready Wed
- Week 5 Wed-Fri: Validation runs → ship Fri

**Parallelization Potential:**

- L-01 + L-03 + L-04 concurrent (3 Forge streams)
- After L-01 done: L-02 takes priority (2d intensive)
- After L-02 done: L-05 (1d) + Validation (0.5d) concurrent
- Need: Forge×3 full-time + Guardian×1 from Wed Week 3 onward

---

## Sequencing Decision Matrix

### Question 1: Can Task X start in parallel with Task Y?

Use this matrix. ✅ = yes, 🔴 = no (blocking), ⚠️ = optional/informational

| From → To   | M-01 | M-02 | M-03 | M-04 | M-05 | M-06 | L-01 | L-02 | L-03 | L-04 | L-05 | Val |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- |
| QW complete | ✅   | ✅   | ✅   | ✅   | ✅   | 🔴   | 🔴   | 🔴   | 🔴   | 🔴   | 🔴   | 🔴  |
| M-01 done   | —    | ✅   | ✅   | ✅   | ✅   | ⚠️   | —    | —    | —    | —    | —    | —   |
| M-02 done   | —    | —    | ✅   | ✅   | ✅   | ⚠️   | —    | —    | —    | —    | —    | —   |
| M-03 done   | —    | —    | —    | ✅   | ✅   | ⚠️   | —    | —    | ✅   | —    | —    | —   |
| M-04 done   | —    | —    | —    | —    | ✅   | ⚠️   | —    | —    | —    | —    | —    | —   |
| M-05 done   | —    | —    | —    | —    | —    | ⚠️   | —    | —    | —    | —    | —    | —   |
| M-06 pass   | —    | —    | —    | —    | —    | —    | ✅   | 🔴   | ✅   | ✅   | 🔴   | 🔴  |
| L-01 done   | —    | —    | —    | —    | —    | —    | —    | ✅   | —    | ✅   | 🔴   | 🔴  |
| L-02 done   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | ✅   | ⚠️  |
| L-03 done   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | ⚠️  |
| L-04 done   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | ⚠️  |
| L-05 done   | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | —    | ✅  |

**Legend:**

- ✅ = Can start immediately (no blocking)
- 🔴 = MUST WAIT (blocking dependency)
- ⚠️ = Optional parallel (informational, not blocking)
- — = N/A (same task or already complete)

### Question 2: What's the earliest I can ship each task?

| Task              | Earliest Start | Estimated End | Can Parallelize With       |
| ----------------- | -------------- | ------------- | -------------------------- |
| QW-01             | Week 1 Mon     | Week 1 Tue    | QW-02, QW-03               |
| QW-02             | Week 1 Mon     | Week 1 Tue    | QW-01, QW-03               |
| QW-03             | Week 1 Mon     | Week 1 Tue    | QW-01, QW-02               |
| M-01              | Week 2 Mon     | Week 2 Fri    | M-02, M-03, M-04, M-05     |
| M-02              | Week 2 Mon     | Week 2 Fri    | M-01, M-03, M-04, M-05     |
| M-03              | Week 2 Mon     | Week 2 Fri    | M-01, M-02, M-04, M-05     |
| M-04              | Week 2 Mon     | Week 3 Tue    | M-01, M-02, M-03, M-05     |
| M-05              | Week 2 Mon     | Week 2 Fri    | M-01, M-02, M-03, M-04     |
| **M-06** 🔴       | Week 3 Mon     | Week 3 Wed    | **M-03 & M-04 must pass**  |
| **L-01** 🔴       | Week 3 Wed     | Week 3 Thu    | L-03, L-04 (not L-02)      |
| **L-02** 🔴       | Week 4 Wed     | Week 5 Mon    | L-03, L-04 (not L-05)      |
| L-03              | Week 3 Wed     | Week 3 Thu    | L-02, L-04 (after M-03)    |
| L-04              | Week 4 Wed     | Week 4 Thu    | L-02, L-03 (after L-01)    |
| **L-05** 🔴       | Week 5 Mon     | Week 5 Wed    | Validation (after L-02)    |
| **Validation** 🔴 | Week 5 Wed     | Week 5 Fri    | **All long-term required** |

---

## Recommended Scheduling (Forge × 3, Guardian × 1)

### Week 1 (Mon-Tue): Quick Wins Sprint

```
Mon 9am:  QW-01 (Forge-A), QW-02 (Forge-B), QW-03 (Forge-C) all start
Tue 2pm:  All QW complete & pass ✅
Tue 5pm:  Checkpoint review: baseline memory/latency established
```

### Week 2 (Mon-Fri): Medium Feature Sprint

```
Mon 9am:  M-01 (Medic start profiling)
          M-02 (Forge-A: buffer pools)
          M-03 (Forge-B: quantization)
          M-04 (Forge-C: KV paging)
          M-05 (Guardian prep: WASM SIMD setup)

Wed:      Forge-C can assist M-04 if overrun; rotate WASM SIMD to Forge-A/B
          Medic delivers M-01 profiling results to Forge-A

Fri 5pm:  M-01 to M-05 feature-complete (code review scheduled)
          Guardian: M-06 harness ready to run
```

### Week 3 (Mon-Fri): Medium Gate + Long-Term Ramp

```
Mon 9am:  M-06 validation runs (Guardian lead, Forge-A monitoring)
Tue-Wed:  M-06 decision point
          - If PASS: Long-term green light
          - If FAIL: Forge-A/B loop back to fix M-03/M-04; rethink Fri

Thu 9am:  Assuming M-06 PASS:
          L-01 (Forge-A: UseAfterFree workaround)
          L-03 (Forge-B: LoRA adapters, if M-03 stable)
          L-04 (Forge-C: MoE load-balancing design/start after L-01 done)

Fri 5pm:  L-01 complete ✅
          L-03 (60-80% done)
          L-04 (design complete, ready for implementation Mon)
```

### Week 4 (Mon-Fri): Long-Term Main Sprint

```
Mon 9am:  L-02 starts (Forge-A dedicated, 2-day task)
          L-03 finishes (Forge-B)
          L-04 implementation (Forge-C, from L-01 insights)

Wed-Thu:  L-02 implementation peak
          Forge-B/C assist if bottlenecked

Fri 5pm:  L-02 80-90% done (test + debug continues)
```

### Week 5 (Mon-Fri): Long-Term Completion + Validation

```
Mon 9am:  L-02 finishes & passes unit tests
          L-05 starts (Forge-A or B, whoever freed up)
          Guardian: Validation suite prep

Tue 5pm:  L-05 complete

Wed 9am:  Validation-01 runs (all features enabled)

Wed-Fri:  If Validation PASS ✅ → Ready to ship
          If Validation issues → Debug & iterate

Fri 5pm:  Release-ready or rework in progress
```

---

## Communication Checkpoints

| Date                | Attendees                      | Decision                            | Inputs                       |
| ------------------- | ------------------------------ | ----------------------------------- | ---------------------------- |
| **Tue Week 1** 5pm  | Chief, Architect               | QW checkpoint                       | QW test results              |
| **Fri Week 2** 5pm  | Forge×3, Guardian, Architect   | M-01-05 status                      | Code reviews, benchmark data |
| **Tue Week 3** 10am | Guardian, Forge-A/B, Architect | **M-06 gate decision**              | M-06 test results 🔴         |
| **Wed Week 3** 9am  | All engineers, Chief           | **Long-term green light or rework** | M-06 outcome                 |
| **Fri Week 3** 5pm  | Forge-A/B/C, Architect         | L-01/L-03/L-04 status               | Unit test results            |
| **Fri Week 4** 5pm  | Forge-A/B/C, Architect         | L-02 status, L-05 readiness         | L-02 benchmark data          |
| **Tue Week 5** 5pm  | Guardian, Forge, Architect     | L-05 complete                       | Final e2e test results       |
| **Fri Week 5** 5pm  | Chief, All engineers           | **SHIP DECISION**                   | Validation-01 report 🎯      |

---

## Troubleshooting: If Behind Schedule

### If QW not complete by Tue Week 1

- **Action:** Extend to Wed; don't start Medium until all QW pass
- **Impact:** Medium shifts to Thu start; loses 1 day

### If M-03 or M-04 unstable (M-06 fails)

- **Action:** Identify regression; Forge-A/B loop back; retest Wed
- **Impact:** Long-term starts Thu Week 3 instead of Wed (–1 day)
- **Mitigation:** Pre-validation: run M-03+M-04 together mid-Week 2

### If L-01 (UseAfterFree) has upstream blockers

- **Action:** Escalate to jax-js maintainers; prototype workaround in parallel
- **Impact:** L-02 can't start; risks shipping without distributed feature
- **Mitigation:** Start L-02 design early; have fallback (single-GPU
  optimizations sufficient?)

### If L-02 overruns (multi-GPU complexity)

- **Action:** Parallelize L-02 design+implementation; call out bottleneck early
  (Week 4 Mon)
- **Impact:** Delays L-05, shifts Validation to following week
- **Mitigation:** Weekly standups; escalate design review to Chief if stuck

### If Validation-01 finds regressions

- **Action:** Identify which feature regressed; isolate + fix specific task
- **Impact:** +1-2 days for fix + retest; may slip release to following Mon
- **Mitigation:** Aggressive unit testing in M-06 + integration testing in L
  tasks

---

## Success Criteria Verification

### Quick Wins Checkpoint (Tue Week 1)

```bash
✅ deno task bench --all-quick-wins
   ├─ QW-01: Memory stable over 100 steps (delta < 1%)
   ├─ QW-02: Short prompts <200MB memory (vs ~400MB baseline)
   └─ QW-03: Multi-model init <100ms
```

### Medium Gate Checkpoint (Tue-Wed Week 3)

```bash
✅ deno task bench --all-models --quantize --paging --prompts=short,medium,long
   ├─ Throughput: 1.8-2.2x baseline (M-03)
   ├─ Memory: 50% reduction (M-03) + paging overhead <5% (M-04)
   ├─ p90 latency: <1.5x p50 (M-02)
   ├─ Prefill (WASM): 3-5x faster (M-05)
   └─ NO race conditions or unexpected interactions
```

### Long-Term Completion (Fri Week 5)

```bash
✅ deno task bench --all-features --all-models --output-report=final_validation.md
   ├─ UseAfterFree: No errors on 8K+ prompts (L-01)
   ├─ Distributed: 1.8-3.5x scaling on 2-4 GPUs (L-02)
   ├─ LoRA: 10x fine-tuning speedup (L-03)
   ├─ MoE: 2-3x sparse speedup on Maple (L-04)
   ├─ Speculative: 2-3x decode speedup (L-05)
   └─ OVERALL: 5-8x speedup vs baseline, ship ✅
```

---

**Document Owner:** Architect\
**Last Updated:** 2025-09-21\
**Status:** Ready for team implementation\
**Next:** Chief dispatches to Forge/Guardian/Medic
