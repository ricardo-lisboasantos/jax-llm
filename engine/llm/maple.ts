/**
 * @module llm/maple
 *
 * Maple architecture — a Mixture-of-Experts (MoE) decoder transformer
 * from DeepGrove.  Key architectural features:
 *
 *  - **MoE MLP**: 256 experts with top-8 routing per token.  Each expert
 *    is a SwiGLU MLP with `moe_intermediate_size = 512`.
 *  - **QK-norm**: RMSNorm applied to query and key projections (like Gemma).
 *  - **Partial rotary**: 50% of each head dimension receives RoPE.
 *  - **Sliding + full attention**: layers alternate between sliding window
 *    (512 tokens, with partial RoPE) and full attention (no positional
 *    encoding — NoPE on global attention layers).
 *  - Standard RMSNorm, GQA, SwiGLU expert MLPs.
 *
 * Note: This is a 20B-A1B model.  The full checkpoint is too large for
 * in-browser inference, but this module implements the complete forward
 * pass for correctness and future use.
 *
 * Reference: https://huggingface.co/deepgrove/maple-preview
 */

import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type MapleLinear, runLinear } from "./layers/maple/maple_linear.ts";
import { type RMSNorm, runRMSNorm } from "./layers/maple/maple_rms_norm.ts";
import type { MapleDecoderLayer } from "./layers/maple/maple_decoder.ts";
import {
  ensureMapleStateCapacity,
  type MapleState,
} from "./state/maple_state.ts";
import { runEmbedding } from "./layers/maple/maple_embedding.ts";
import { MAPLE_CONFIG } from "./configs/maple_config.ts";
import {
  isSlidingLayer,
  runMapleAttentionPrefill,
  runMapleAttentionStep,
} from "./layers/maple/maple_attention.ts";
import { runMapleMoENorm, runMoE } from "./layers/maple/maple_moe.ts";

export type MapleModel = {
  wordEmbeddings: MapleLinear;
  layers: MapleDecoderLayer[];
  norm: RMSNorm;
  lmHead: MapleLinear;
};

export function runMaplePrefill(
  model: MapleModel,
  tokenIds: np.Array,
  state: MapleState,
): np.Array {
  ensureMapleStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.wordEmbeddings.weight.ref }, tokenIds);

  for (let i = 0; i < MAPLE_CONFIG.numHiddenLayers; i++) {
    const layer = model.layers[i];
    const sliding = isSlidingLayer(i);
    const useRoPE = sliding || !MAPLE_CONFIG.nopeOnGlobalAttention;

    // Attention (JIT-compiled).
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runMapleAttentionPrefill(
      layer,
      x,
      useRoPE,
      state.capacity,
    );

    // MoE (non-JIT, loops over experts).
    const residual = x.ref;
    const h = runMapleMoENorm(layer, x);
    const moeOut = runMoE(layer.mlp, h);
    x = residual.add(moeOut);
  }

  x = runRMSNorm(model.norm, x);
  x = x.slice([-1]);
  const logits = runLinear(model.lmHead, x).reshape([MAPLE_CONFIG.vocabSize]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runMapleStep(
  model: MapleModel,
  tokenId: number,
  state: MapleState,
): np.Array {
  ensureMapleStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.wordEmbeddings.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < MAPLE_CONFIG.numHiddenLayers; i++) {
    const layer = model.layers[i];
    const sliding = isSlidingLayer(i);
    const useRoPE = sliding || !MAPLE_CONFIG.nopeOnGlobalAttention;

    // Attention (JIT-compiled).
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runMapleAttentionStep(
      layer,
      oldCache,
      x,
      position,
      slot,
      validLength,
      useRoPE,
    );

    // MoE (non-JIT, loops over experts).
    const residual = x.ref;
    const h = runMapleMoENorm(layer, x);
    const moeOut = runMoE(layer.mlp, h);
    x = residual.add(moeOut);
  }

  x = runRMSNorm(model.norm, x);
  const logits = runLinear(model.lmHead, x).reshape([MAPLE_CONFIG.vocabSize]);
  state.position++;
  return logits;
}
