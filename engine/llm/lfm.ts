import { numpy as np, tree } from "npm:@jax-js/jax@^0.1.25";
import { type LfmLinear, runLinear } from "./layers/lfm/lfm_linear.ts";
import { type RMSNorm, runRMSNorm } from "./layers/lfm/lfm_rms_norm.ts";
import type { LfmAttentionLayer } from "./layers/lfm/lfm_attention_layer.ts";
import {
  type LfmConvLayer,
  runConvLayerPrefill,
  runConvLayerStep,
} from "./layers/lfm/lfm_conv_layer.ts";
import { ensureStateCapacity, type LfmState } from "./state/lfm_state.ts";
import type { LfmCache } from "./cache/lfm_cache.ts";
import { LFM_CONFIG } from "./configs/lfm_config.ts";
import {
  isAttentionLayer,
  runAttentionLayerPrefill,
  runAttentionLayerScore,
  runAttentionLayerStep,
} from "./layers/lfm/lfm_attention.ts";
import { runEmbedding } from "./layers/lfm/lfm_embedding.ts";

/** LFM model weights: token embeddings, norm, and hybrid conv/attention layers. */
export type LfmModel = {
  embedTokens: LfmLinear;
  embeddingNorm: RMSNorm;
  layers: (LfmAttentionLayer | LfmConvLayer)[];
};

/**
 * Run an LFM prefill pass over the full prompt.
 *
 * @param model Model weights.
 * @param tokenIds Prompt token IDs.
 * @param state Mutable KV-cache state updated in place.
 * @returns Logits for the last prompt token.
 */
export function runLfmPrefill(
  model: LfmModel,
  tokenIds: np.Array,
  state: LfmState,
): np.Array {
  ensureStateCapacity(state, tokenIds.shape[0]);
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < LFM_CONFIG.numHiddenLayers; i++) {
    const cache = state.caches[i];
    if (isAttentionLayer(i)) {
      if (cache.kind !== "attention") {
        throw new Error("Invalid attention cache");
      }
      cache.key.dispose();
      cache.value.dispose();
      const [nextX, nextCache] = runAttentionLayerPrefill(
        model.layers[i] as LfmAttentionLayer,
        x,
        state.capacity,
      );
      x = nextX;
      state.caches[i] = { kind: "attention", ...nextCache };
    } else {
      if (cache.kind !== "conv") throw new Error("Invalid convolution cache");
      cache.value.dispose();
      const [nextX, nextCache] = runConvLayerPrefill(
        model.layers[i] as LfmConvLayer,
        x,
      );
      x = nextX;
      state.caches[i] = { kind: "conv", value: nextCache };
    }
  }

  x = runRMSNorm(model.embeddingNorm, x).slice([-1]);
  const logits = runLinear(model.embedTokens, x).reshape([
    LFM_CONFIG.vocabSize,
  ]);
  state.position = tokenIds.shape[0];
  return logits;
}

/**
 * Run a single LFM decode step for one generated token.
 *
 * @param model Model weights.
 * @param tokenId The most recently generated token ID.
 * @param state Mutable KV-cache state updated in place.
 * @returns Next-token logits.
 */
export function runLfmStep(
  model: LfmModel,
  tokenId: number,
  state: LfmState,
): np.Array {
  ensureStateCapacity(state, state.position + 1);
  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < LFM_CONFIG.numHiddenLayers; i++) {
    const cache = state.caches[i];
    if (isAttentionLayer(i)) {
      if (cache.kind !== "attention") {
        throw new Error("Invalid attention cache");
      }
      const oldCache = { key: cache.key, value: cache.value };
      const [nextX, nextCache] = runAttentionLayerStep(
        model.layers[i] as LfmAttentionLayer,
        oldCache,
        x,
        position,
        slot,
        validLength,
      );
      x = nextX;
      state.caches[i] = { kind: "attention", ...nextCache };
      // Workaround: jax-js has refcount bug on long sequences; silently ignore UseAfterFreeError
      try {
        oldCache.key.dispose();
        oldCache.value.dispose();
      } catch (_e) {
        // Upstream jax-js refcount issue: ignore
      }
    } else {
      if (cache.kind !== "conv") throw new Error("Invalid convolution cache");
      const oldCacheValue = cache.value;
      const [nextX, nextCache] = runConvLayerStep(
        model.layers[i] as LfmConvLayer,
        oldCacheValue,
        x,
      );
      x = nextX;
      state.caches[i] = { kind: "conv", value: nextCache };
      // Workaround: jax-js has refcount bug on long sequences; silently ignore UseAfterFreeError
      try {
        oldCacheValue.dispose();
      } catch (_e) {
        // Upstream jax-js refcount issue: ignore
      }
    }
  }

  x = runRMSNorm(model.embeddingNorm, x);
  const logits = runLinear(model.embedTokens, x).reshape([
    LFM_CONFIG.vocabSize,
  ]);

  // Phase 3.0: Update paged cache if enabled (for long-context tracking)
  if (state.usePagedCache && state.pagedCache) {
    state.pagedCache.updateValidLength(state.position, state.position + 1);
  }

  state.position++;
  return logits;
}

/** Dispose one cache entry, absorbing the upstream refcount quirk. */
function disposeLfmCache(cache: LfmCache): void {
  try {
    if (cache.kind === "attention") {
      cache.key.dispose();
      cache.value.dispose();
    } else {
      cache.value.dispose();
    }
  } catch {
    // Upstream jax-js refcount issue: ignore
  }
}

/** Drop snapshot refs without touching live state (always safe). */
function dropDraftSnapshot(state: LfmState): void {
  const snap = state.draftSnapshot;
  if (!snap) return;
  for (const cache of snap.caches) disposeLfmCache(cache);
  state.draftSnapshot = undefined;
}

/**
 * Snapshot pre-score caches for speculative verification (PLD).
 * Caches are persistent-functional, so this is O(1) `.ref` bumps.
 */
export function snapshotLfmDraft(state: LfmState): void {
  dropDraftSnapshot(state);
  state.draftSnapshot = {
    position: state.position,
    caches: state.caches.map((cache) =>
      cache.kind === "attention"
        ? {
          kind: "attention" as const,
          key: cache.key.ref,
          value: cache.value.ref,
        }
        : { kind: "conv" as const, value: cache.value.ref }
    ),
  };
}

/**
 * Keep scored state (all drafts accepted): release the snapshot,
 * keep the advanced position and extended caches.
 */
export function confirmLfmDraft(state: LfmState): void {
  dropDraftSnapshot(state);
}

/**
 * Roll back to the pre-score state (draft rejected): dispose scored
 * caches, restore snapshot arrays, reset the position. Stale attention
 * slots need no scrubbing — subsequent steps overwrite by slot and
 * mask by valid length.
 */
export function truncateLfmDraft(state: LfmState): void {
  const snap = state.draftSnapshot;
  if (!snap) return;
  for (const cache of state.caches) disposeLfmCache(cache);
  state.caches = snap.caches;
  state.position = snap.position;
  state.draftSnapshot = undefined;
}

/**
 * Score k drafted tokens in one batched forward (PLD verification).
 *
 * Returns per-token logits `[k, vocab]`: row i predicts the token after
 * draft prefix `d[0..i-1]`. Advances position and caches exactly as k
 * sequential steps would — resolve with {@linkcode confirmLfmDraft} or
 * {@linkcode truncateLfmDraft}. Snapshots automatically on entry.
 *
 * Correctness is defined as exact equality with k sequential
 * {@linkcode runLfmStep} calls (tested live).
 *
 * @param model Model weights.
 * @param draftIds Draft token IDs, shape `[k]`.
 * @param state Mutable KV-cache state updated in place.
 * @returns Logits for every draft position, shape `[k, vocabSize]`.
 */
export function runLfmScoreDrafts(
  model: LfmModel,
  draftIds: np.Array,
  state: LfmState,
): np.Array {
  const k = draftIds.shape[0];
  ensureStateCapacity(state, state.position + k);
  snapshotLfmDraft(state);
  const basePosition = state.position;

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, draftIds);

  for (let i = 0; i < LFM_CONFIG.numHiddenLayers; i++) {
    const cache = state.caches[i];
    if (isAttentionLayer(i)) {
      if (cache.kind !== "attention") {
        throw new Error("Invalid attention cache");
      }
      const oldCache = { key: cache.key, value: cache.value };
      const [nextX, nextCache] = runAttentionLayerScore(
        model.layers[i] as LfmAttentionLayer,
        oldCache,
        x,
        basePosition,
        basePosition,
        basePosition,
      );
      x = nextX;
      state.caches[i] = { kind: "attention", ...nextCache };
      // No manual dispose: ops auto-consume exactly the state-owned share;
      // the snapshot's share stays alive for confirm/truncate.
    } else {
      if (cache.kind !== "conv") throw new Error("Invalid convolution cache");
      // Conv state is recurrent — score token-by-token (cheap: the cache
      // is 3 rows and the matmuls are tiny), then restack in order.
      const rows: np.Array[] = [];
      let cVal = cache.value;
      for (let t = 0; t < k; t++) {
        // Per-axis slice spec ([t, t+1] rows, [] full width); slice
        // consumes, so ref each row out of x. Layer weights are reused
        // across iterations — ref a fresh copy per iteration since each
        // call consumes its inputs (move semantics). cVal is consumed by
        // each step call (no manual dispose — the snapshot owns it).
        const row = x.ref.slice([t, t + 1], []);
        const layerRef = tree.ref(model.layers[i] as LfmConvLayer);
        const [nextRow, nextCache] = runConvLayerStep(
          layerRef,
          cVal,
          row,
        );
        cVal = nextCache;
        rows.push(nextRow);
      }
      x = np.concatenate(rows, 0);
      state.caches[i] = { kind: "conv", value: cVal };
    }
  }

  x = runRMSNorm(model.embeddingNorm, x);
  const logits = runLinear(model.embedTokens, x);
  state.position = basePosition + k;
  return logits;
}
