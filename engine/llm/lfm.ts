import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type LfmLinear, runLinear } from "./layers/lfm/lfm_linear.ts";
import { type RMSNorm, runRMSNorm } from "./layers/lfm/lfm_rms_norm.ts";
import type { LfmAttentionLayer } from "./layers/lfm/lfm_attention_layer.ts";
import {
  type LfmConvLayer,
  runConvLayerPrefill,
  runConvLayerStep,
} from "./layers/lfm/lfm_conv_layer.ts";
import { ensureStateCapacity, type LfmState } from "./state/lfm_state.ts";
import { LFM_CONFIG } from "./configs/lfm_config.ts";
import {
  isAttentionLayer,
  runAttentionLayerPrefill,
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
      // oldCache.key.dispose();
      // oldCache.value.dispose();
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
      // oldCacheValue.dispose();
    }
  }

  x = runRMSNorm(model.embeddingNorm, x);
  const logits = runLinear(model.embedTokens, x).reshape([
    LFM_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}
