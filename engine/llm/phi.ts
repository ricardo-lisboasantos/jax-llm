/**
 * @module llm/phi
 *
 * Phi-2 architecture — a decoder-only transformer from Microsoft that
 * differs from the Llama family in several ways:
 *
 *  - **LayerNorm** (with bias) instead of RMSNorm.
 *  - **GeLU** activation (not SwiGLU) — the MLP is a simple 2-layer
 *    `fc2(gelu(fc1(x)))` network.
 *  - **Partial rotary** position embeddings: only `partialRotaryFactor`
 *    (40%) of each head dimension receives RoPE; the rest passes through.
 *  - **No GQA** — all attention heads share the same number of K/V heads.
 *  - The output projection is named `dense` (not `o_proj`).
 *
 * Reference: https://huggingface.co/microsoft/phi-2
 */

import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type PhiLinear, runLinear } from "./layers/phi/phi_linear.ts";
import {
  type PhiDecoderLayer,
  runPhiDecoderLayerPrefill,
  runPhiDecoderLayerStep,
} from "./layers/phi/phi_decoder.ts";
import type { LayerNorm } from "./gpt.ts";
import { ensurePhiStateCapacity, type PhiState } from "./state/phi_state.ts";
import { runEmbedding } from "./layers/phi/phi_embeddings.ts";
import { PHI_CONFIG } from "./configs/phi_config.ts";
import { runLayerNorm } from "./layers/phi/phi_rms_norm.ts";

export type PhiModel = {
  embedTokens: PhiLinear;
  layers: PhiDecoderLayer[];
  finalLayernorm: LayerNorm;
  lmHead: PhiLinear;
};

export function runPhiPrefill(
  model: PhiModel,
  tokenIds: np.Array,
  state: PhiState,
): np.Array {
  ensurePhiStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < PHI_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runPhiDecoderLayerPrefill(
      model.layers[i],
      x,
      state.capacity,
    );
  }

  x = runLayerNorm(model.finalLayernorm, x);
  x = x.slice([-1]);
  const logits = runLinear(model.lmHead, x).reshape([PHI_CONFIG.vocabSize]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runPhiStep(
  model: PhiModel,
  tokenId: number,
  state: PhiState,
): np.Array {
  ensurePhiStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < PHI_CONFIG.numHiddenLayers; i++) {
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runPhiDecoderLayerStep(
      model.layers[i],
      oldCache,
      x,
      position,
      slot,
      validLength,
    );
  }

  x = runLayerNorm(model.finalLayernorm, x);
  const logits = runLinear(model.lmHead, x).reshape([PHI_CONFIG.vocabSize]);
  state.position++;
  return logits;
}
