/**
 * @module llm/qwen
 *
 * Qwen2 architecture — a Llama-style decoder transformer with RMSNorm,
 * SwiGLU MLP, rotary position embeddings (RoPE), and grouped-query
 * attention (GQA).  Unlike Llama, Qwen2 adds biases to the Q/K/V
 * projections (but not the output projection) and ties the embedding
 * and LM head weights.
 *
 * Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B
 */

import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { QWEN_CONFIG } from "./configs/qwen_config.ts";
import { type QwenLinear, runLinear } from "./layers/qwen/qwen_linear.ts";
import {
  type QwenDecoderLayer,
  runQwenDecoderLayerPrefill,
  runQwenDecoderLayerStep,
} from "./layers/qwen/qwen_decoder.ts";
import { type QwenRMSNorm, runRMSNorm } from "./layers/qwen/qwen_rms_norm.ts";
import { ensureQwenStateCapacity, type QwenState } from "./state/qwen_state.ts";
import { runEmbedding } from "./layers/qwen/qwen_embeddings.ts";

export type QwenModel = {
  embedTokens: QwenLinear;
  layers: QwenDecoderLayer[];
  norm: QwenRMSNorm;
};

export function runQwenPrefill(
  model: QwenModel,
  tokenIds: np.Array,
  state: QwenState,
): np.Array {
  ensureQwenStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < QWEN_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runQwenDecoderLayerPrefill(
      model.layers[i],
      x,
      state.capacity,
    );
  }

  x = runRMSNorm(model.norm, x);
  x = x.slice([-1]);
  // Qwen2 ties embeddings and LM head.
  const logits = runLinear(model.embedTokens, x).reshape([
    QWEN_CONFIG.vocabSize,
  ]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runQwenStep(
  model: QwenModel,
  tokenId: number,
  state: QwenState,
): np.Array {
  ensureQwenStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < QWEN_CONFIG.numHiddenLayers; i++) {
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runQwenDecoderLayerStep(
      model.layers[i],
      oldCache,
      x,
      position,
      slot,
      validLength,
    );
  }

  x = runRMSNorm(model.norm, x);
  // Qwen2 ties embeddings and LM head.
  const logits = runLinear(model.embedTokens, x).reshape([
    QWEN_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}
