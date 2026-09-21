/**
 * @module llm/bonsai
 *
 * Bonsai architecture — a Llama-style decoder transformer used by
 * deepgrove/Bonsai, a 500M-parameter ternary-weight language model.
 * The architecture is standard Llama: BonsaiRMSNorm, SwiGLU MLP, RoPE, and
 * grouped-query attention (GQA) with no biases on any projection.
 *
 * The ternary quantization (weights ∈ {-1, 0, +1} with per-channel scales)
 * is handled at the weight-loading level; this module implements the
 * dequantized forward pass identically to a standard Llama model.
 *
 * Reference: https://huggingface.co/deepgrove/Bonsai
 */

import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { BONSAI_CONFIG } from "./configs/bonsai_config.ts";
import {
  type BonsaiState,
  ensureBonsaiStateCapacity,
} from "./state/bonsai_state.ts";
import { type BonsaiLinear, runLinear } from "./layers/bonsai/bonsai_linear.ts";
import {
  type BonsaiRMSNorm,
  runRMSNorm,
} from "./layers/bonsai/bonsai_rms_norm.ts";
import {
  type BonsaiDecoderLayer,
  runBonsaiDecoderLayerPrefill,
  runBonsaiDecoderLayerStep,
} from "./layers/bonsai/bonsai_decoder.ts";
import { runEmbedding } from "./layers/bonsai/bonsai_embedding.ts";

export type BonsaiModel = {
  embedTokens: BonsaiLinear;
  layers: BonsaiDecoderLayer[];
  norm: BonsaiRMSNorm;
  lmHead: BonsaiLinear;
};

export function runBonsaiPrefill(
  model: BonsaiModel,
  tokenIds: np.Array,
  state: BonsaiState,
): np.Array {
  ensureBonsaiStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < BONSAI_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runBonsaiDecoderLayerPrefill(
      model.layers[i],
      x,
      state.capacity,
    );
  }

  x = runRMSNorm(model.norm, x);
  x = x.slice([-1]);
  const logits = runLinear(model.lmHead, x).reshape([BONSAI_CONFIG.vocabSize]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runBonsaiStep(
  model: BonsaiModel,
  tokenId: number,
  state: BonsaiState,
): np.Array {
  ensureBonsaiStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < BONSAI_CONFIG.numHiddenLayers; i++) {
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runBonsaiDecoderLayerStep(
      model.layers[i],
      oldCache,
      x,
      position,
      slot,
      validLength,
    );
  }

  x = runRMSNorm(model.norm, x);
  const logits = runLinear(model.lmHead, x).reshape([BONSAI_CONFIG.vocabSize]);
  state.position++;
  return logits;
}
