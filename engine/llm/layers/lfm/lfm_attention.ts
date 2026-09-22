import { jit, nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import type { LfmAttentionArrays } from "../../cache/lfm_attention_arrays.ts";
import { LFM_CONFIG } from "../../configs/lfm_config.ts";
import { type LfmLinear, runLinear } from "./lfm_linear.ts";

import type { LfmAttentionLayer } from "./lfm_attention_layer.ts";
import { runMLP } from "./lfm_mlp.ts";
import { rotateHalf } from "../common/ops.ts";
import { type RMSNorm, runRMSNorm } from "./lfm_rms_norm.ts";

export const ATTENTION_SCALE = 1 / Math.sqrt(LFM_CONFIG.headDim);

export type LfmAttention = {
  qProj: LfmLinear;
  kProj: LfmLinear;
  vProj: LfmLinear;
  outProj: LfmLinear;
  qLayernorm: RMSNorm;
  kLayernorm: RMSNorm;
};

function runAttentionPrefill(
  { qProj, kProj, vProj, outProj, qLayernorm, kLayernorm }: LfmAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    LFM_CONFIG.numAttentionHeads,
    LFM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  q = runRMSNorm(qLayernorm, q);
  k = runRMSNorm(kLayernorm, k);
  [q, k] = applyRoPE(q, k, 0);
  const attn = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    outProj,
    attn.reshape([T, LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

function applyRoPE(
  q: np.Array,
  k: np.Array,
  offset: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;
  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(LFM_CONFIG.ropeTheta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq);

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);
  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

function runAttentionStep(
  { qProj, kProj, vProj, outProj, qLayernorm, kLayernorm }: LfmAttention,
  cache: LfmAttentionArrays,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: LfmAttentionArrays } {
  const T = 1;

  let q = runLinear(qProj, x.ref).reshape([
    T,
    LFM_CONFIG.numAttentionHeads,
    LFM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  const v = runLinear(vProj, x).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  q = runRMSNorm(qLayernorm, q);
  k = runRMSNorm(kLayernorm, k);
  [q, k] = applyRoPE(q, k, position);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);
  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(q, key.ref, value.ref, {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    outProj,
    attn.reshape([T, LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim]),
  );

  return { output, cache: { key, value } };
}

function padAttentionCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): LfmAttentionArrays {
  const T = key.shape[0];
  if (T > capacity) {
    throw new Error(`Prompt length ${T} exceeds cache capacity ${capacity}`);
  }
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

export const runAttentionLayerPrefill = jit(
  function runAttentionLayerPrefill(
    { operatorNorm, ffnNorm, feedForward, selfAttn }: LfmAttentionLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, LfmAttentionArrays] {
    const residual = x.ref;
    x = runRMSNorm(operatorNorm, x);
    const { output, key, value } = runAttentionPrefill(selfAttn, x);
    x = residual.add(output);

    const residual2 = x.ref;
    x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
    return [residual2.add(x), padAttentionCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

export const runAttentionLayerStep = jit(function runAttentionLayerStep(
  { operatorNorm, ffnNorm, feedForward, selfAttn }: LfmAttentionLayer,
  cache: LfmAttentionArrays,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): [np.Array, LfmAttentionArrays] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache: updatedCache } = runAttentionStep(
    selfAttn,
    cache,
    x,
    position,
    slot,
    validLength,
  );
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), updatedCache];
});

export function isAttentionLayer(index: number): boolean {
  return LFM_CONFIG.layerTypes[index] === "full_attention";
}
