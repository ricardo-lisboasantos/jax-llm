/**
 * @module chat/sampler
 *
 * Logit sampling with temperature, top-k, top-p (nucleus), and
 * repetition penalty. Extracted from the monolithic ChatEngine to
 * keep the high-level chat layer focused and testable.
 */

import type { SamplingDefaults } from "../runtime/types.ts";

export type SamplingOptions = {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  previousTokens: number[];
};

/**
 * Sample a single token ID from a logits vector.
 *
 * @param logits  Float32Array of raw model logits over the vocabulary.
 * @param opts    Sampling parameters.
 * @returns The selected token ID.
 */
export function sampleLogits(
  logits: Float32Array,
  opts: SamplingOptions,
): number {
  const V = logits.length;
  const penalty = opts.repetitionPenalty;
  const usePenalty = penalty !== 1;
  // Build the previous-token set lazily; only needed when a penalty applies.
  const prevSet = usePenalty && opts.previousTokens.length > 0
    ? new Set<number>(opts.previousTokens)
    : null;

  // --- Greedy fast path: temperature <= 0 ---
  // Argmax scan over all logits, applying repetition penalty inline to the
  // few previous-token indices. No top-K collection, no object allocations.
  if (opts.temperature <= 0) {
    let bestId = -1;
    let bestLogit = -Infinity;
    for (let id = 0; id < V; id++) {
      let logit = logits[id];
      if (Number.isNaN(logit)) continue;
      if (prevSet !== null && prevSet.has(id)) {
        logit = logit < 0 ? logit * penalty : logit / penalty;
      }
      if (logit > bestLogit) {
        bestLogit = logit;
        bestId = id;
      }
    }
    if (bestId === -1) {
      throw new Error("Model returned all-NaN logits.");
    }
    return bestId;
  }

  // --- Stochastic path: top-K via parallel typed arrays ---
  const k = Math.max(1, Math.min(opts.topK, V));
  const topLogits = new Float32Array(k);
  const topIds = new Uint32Array(k);
  let count = 0; // number of valid entries currently held

  // Pass 1: collect top-K by raw logit (no repetition penalty yet). Maintains
  // descending order via binary search for the insertion point, then a shift.
  for (let id = 0; id < V; id++) {
    const logit = logits[id];
    if (Number.isNaN(logit)) continue;
    // Not full yet, or beats the current tail => insert.
    if (count < k || logit > topLogits[count - 1]) {
      // Binary search for the first index whose logit is <= `logit`.
      let lo = 0;
      let hi = count < k ? count : k - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (topLogits[mid] > logit) lo = mid + 1;
        else hi = mid;
      }
      const insertAt = lo;
      // Shift the tail right by one to open a slot at insertAt.
      const shiftEnd = count < k ? count : k - 1;
      for (let j = shiftEnd; j > insertAt; j--) {
        topLogits[j] = topLogits[j - 1];
        topIds[j] = topIds[j - 1];
      }
      topLogits[insertAt] = logit;
      topIds[insertAt] = id;
      if (count < k) count++;
    }
  }

  if (count === 0) {
    throw new Error("Model returned all-NaN logits.");
  }

  // Pass 2 (rep penalty pre-pass): apply penalty only to the small set of
  // collected candidates that appear in previousTokens, then re-sort the
  // small k-element array. Far cheaper than scanning all V logits.
  if (prevSet !== null) {
    for (let i = 0; i < count; i++) {
      if (prevSet.has(topIds[i])) {
        const l = topLogits[i];
        topLogits[i] = l < 0 ? l * penalty : l / penalty;
      }
    }
    // Re-sort descending: a single pass of insertion sort suffices since at
    // most `|prevSet ∩ topK|` elements moved, and k is small (~40-64).
    for (let i = 1; i < count; i++) {
      const l = topLogits[i];
      const d = topIds[i];
      let j = i - 1;
      while (j >= 0 && topLogits[j] < l) {
        topLogits[j + 1] = topLogits[j];
        topIds[j + 1] = topIds[j];
        j--;
      }
      topLogits[j + 1] = l;
      topIds[j + 1] = d;
    }
  }

  // Pass 3: softmax over top-k with temperature.
  const maxLogit = topLogits[0];
  if (!Number.isFinite(maxLogit)) return topIds[0];

  const probs = new Float32Array(count);
  let total = 0;
  for (let i = 0; i < count; i++) {
    const p = Math.exp((topLogits[i] - maxLogit) / opts.temperature);
    probs[i] = p;
    total += p;
  }
  if (!Number.isFinite(total) || total <= 0) return topIds[0];

  // Pass 4: top-p (nucleus) filtering.
  let keptTotal = 0;
  let kept = 0;
  for (; kept < count; kept++) {
    keptTotal += probs[kept];
    if (keptTotal / total >= opts.topP) {
      kept++;
      break;
    }
  }
  if (kept === 0) kept = 1;

  // Pass 5: sample from the filtered distribution.
  let r = Math.random() * keptTotal;
  for (let i = 0; i < kept; i++) {
    r -= probs[i];
    if (r <= 0) return topIds[i];
  }
  return topIds[kept - 1];
}

/** Merge per-model defaults with caller overrides. */
export function resolveSamplingDefaults(
  base: SamplingDefaults,
  overrides?: Partial<SamplingDefaults>,
): SamplingDefaults {
  return {
    temperature: overrides?.temperature ?? base.temperature,
    topK: overrides?.topK ?? base.topK,
    topP: overrides?.topP ?? base.topP,
    repetitionPenalty: overrides?.repetitionPenalty ?? base.repetitionPenalty,
  };
}
