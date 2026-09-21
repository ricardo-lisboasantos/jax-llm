import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { GEMMA_CONFIG } from "./configs/gemma_config.ts";
import { type GemmaLinear, runLinear } from "./layers/gemma/gemma_linear.ts";
import {
  type GemmaDecoderLayer,
  runGemmaDecoderLayerPrefill,
  runGemmaDecoderLayerStep,
} from "./layers/gemma/gemma_decoder_layer.ts";
import {
  ensureGemmaStateCapacity,
  type GemmaState,
} from "./state/gemma_state.ts";
import { runEmbedding } from "./layers/gemma/gemma_embedding.ts";
import { layerRopeTheta } from "./layers/gemma/gemma_rope_theta.ts";
import { type RMSNorm, runRMSNorm } from "./layers/gemma/gemma_rms_norm.ts";

/** Gemma model weights: token embeddings, decoder layers, and final norm. */
export type GemmaModel = {
  embedTokens: GemmaLinear;
  layers: GemmaDecoderLayer[];
  norm: RMSNorm;
};

/**
 * Run a Gemma prefill pass over the full prompt.
 *
 * @param model Model weights.
 * @param tokenIds Prompt token IDs.
 * @param state Mutable KV-cache state updated in place.
 * @returns Logits for the last prompt token.
 */
export function runGemmaPrefill(
  model: GemmaModel,
  tokenIds: np.Array,
  state: GemmaState,
): np.Array {
  ensureGemmaStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < GEMMA_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runGemmaDecoderLayerPrefill(
      model.layers[i],
      x,
      layerRopeTheta(i),
      state.capacity,
    );
  }

  x = runRMSNorm({ weight: model.norm.weight.ref }, x);
  x = x.slice([-1]);
  const embedTokens: GemmaLinear = { weight: model.embedTokens.weight.ref };
  if (model.embedTokens.bias !== undefined) {
    embedTokens.bias = model.embedTokens.bias.ref;
  }
  const logits = runLinear(embedTokens, x).reshape([
    GEMMA_CONFIG.vocabSize,
  ]);
  state.position = tokenIds.shape[0];
  return logits;
}

/**
 * Run a single Gemma decode step for one generated token.
 *
 * @param model Model weights.
 * @param tokenId The most recently generated token ID.
 * @param state Mutable KV-cache state updated in place.
 * @returns Next-token logits.
 */
export function runGemmaStep(
  model: GemmaModel,
  tokenId: number,
  state: GemmaState,
): np.Array {
  ensureGemmaStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < GEMMA_CONFIG.numHiddenLayers; i++) {
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runGemmaDecoderLayerStep(
      model.layers[i],
      oldCache,
      x,
      position,
      slot,
      validLength,
      layerRopeTheta(i),
    );
  }

  x = runRMSNorm({ weight: model.norm.weight.ref }, x);
  const embedTokens: GemmaLinear = { weight: model.embedTokens.weight.ref };
  if (model.embedTokens.bias !== undefined) {
    embedTokens.bias = model.embedTokens.bias.ref;
  }
  const logits = runLinear(embedTokens, x).reshape([
    GEMMA_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}
