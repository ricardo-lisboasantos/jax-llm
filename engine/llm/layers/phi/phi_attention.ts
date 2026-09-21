import { nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { PHI_CONFIG } from "../../configs/phi_config.ts";
import { type PhiLinear, runLinear } from "./phi_linear.ts";
import { applyPartialRoPE } from "./phi_rope.ts";
import type { PhiKVCache } from "../../cache/phi_cache.ts";

export const ATTENTION_SCALE = 1 / Math.sqrt(PHI_CONFIG.headDim);

export type PhiAttention = {
  qProj: PhiLinear;
  kProj: PhiLinear;
  vProj: PhiLinear;
  dense: PhiLinear; // output projection (named "dense" in Phi-2)
};

export function runAttentionPrefill(
  { qProj, kProj, vProj, dense }: PhiAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  const q = runLinear(qProj, x.ref).reshape([
    T,
    PHI_CONFIG.numAttentionHeads,
    PHI_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    PHI_CONFIG.numKeyValueHeads,
    PHI_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    PHI_CONFIG.numKeyValueHeads,
    PHI_CONFIG.headDim,
  ]);

  const [qRot, kRot] = applyPartialRoPE(q, k, 0);
  const attn = nn.dotProductAttention(qRot, kRot.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    dense,
    attn.reshape([T, PHI_CONFIG.numAttentionHeads * PHI_CONFIG.headDim]),
  );
  return { output, key: kRot, value: v };
}

export function runAttentionStep(
  { qProj, kProj, vProj, dense }: PhiAttention,
  cache: PhiKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: PhiKVCache } {
  const T = 1;
  const q = runLinear(qProj, x.ref).reshape([
    T,
    PHI_CONFIG.numAttentionHeads,
    PHI_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    PHI_CONFIG.numKeyValueHeads,
    PHI_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    PHI_CONFIG.numKeyValueHeads,
    PHI_CONFIG.headDim,
  ]);

  const [qRot, kRot] = applyPartialRoPE(q, k, position);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(
    slotMask.ref,
    np.tile(kRot, [capacity, 1, 1]),
    cache.key,
  );
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);

  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(qRot, key.ref, value.ref, {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    dense,
    attn.reshape([T, PHI_CONFIG.numAttentionHeads * PHI_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}
