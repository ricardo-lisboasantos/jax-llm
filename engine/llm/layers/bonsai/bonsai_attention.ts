import { nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type BonsaiLinear, runLinear } from "./bonsai_linear.ts";
import { BONSAI_CONFIG } from "../../configs/bonsai_config.ts";
import { applyRoPE } from "./bonsi_ops.ts";
import type { BonsaiKVCache } from "../../state/bonsai_state.ts";

export const ATTENTION_SCALE = 1 / Math.sqrt(BONSAI_CONFIG.headDim);

export type BonsaiAttention = {
  qProj: BonsaiLinear;
  kProj: BonsaiLinear;
  vProj: BonsaiLinear;
  oProj: BonsaiLinear;
};

export function runAttentionPrefill(
  { qProj, kProj, vProj, oProj }: BonsaiAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  const q = runLinear(qProj, x.ref).reshape([
    T,
    BONSAI_CONFIG.numAttentionHeads,
    BONSAI_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    BONSAI_CONFIG.numKeyValueHeads,
    BONSAI_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    BONSAI_CONFIG.numKeyValueHeads,
    BONSAI_CONFIG.headDim,
  ]);

  const [qRot, kRot] = applyRoPE(q, k, 0);
  const attn = nn.dotProductAttention(qRot, kRot.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, BONSAI_CONFIG.numAttentionHeads * BONSAI_CONFIG.headDim]),
  );
  return { output, key: kRot, value: v };
}

export function runAttentionStep(
  { qProj, kProj, vProj, oProj }: BonsaiAttention,
  cache: BonsaiKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: BonsaiKVCache } {
  const T = 1;
  const q = runLinear(qProj, x.ref).reshape([
    T,
    BONSAI_CONFIG.numAttentionHeads,
    BONSAI_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    BONSAI_CONFIG.numKeyValueHeads,
    BONSAI_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    BONSAI_CONFIG.numKeyValueHeads,
    BONSAI_CONFIG.headDim,
  ]);

  const [qRot, kRot] = applyRoPE(q, k, position);

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
    oProj,
    attn.reshape([T, BONSAI_CONFIG.numAttentionHeads * BONSAI_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}
