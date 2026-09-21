import { jit, nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type MapleLinear, runLinear } from "./maple_linear.ts";
import { type RMSNorm, runRMSNorm } from "./maple_rms_norm.ts";
import { MAPLE_CONFIG } from "../../configs/maple_config.ts";
import { applyPartialRoPE } from "./maple_rope.ts";
import { type MapleKVCache, padCache } from "../../cache/maple_cache.ts";
import type { MapleDecoderLayer } from "./maple_decoder.ts";

export const ATTENTION_SCALE = 1 / Math.sqrt(MAPLE_CONFIG.headDim);

export type MapleAttention = {
  qProj: MapleLinear;
  kProj: MapleLinear;
  vProj: MapleLinear;
  oProj: MapleLinear;
  qNorm: RMSNorm;
  kNorm: RMSNorm;
};

export function isSlidingLayer(index: number): boolean {
  return MAPLE_CONFIG.layerTypes[index] === "sliding_attention";
}

export function runAttentionPrefill(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: MapleAttention,
  x: np.Array,
  useRoPE: boolean,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    MAPLE_CONFIG.numAttentionHeads,
    MAPLE_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    MAPLE_CONFIG.numKeyValueHeads,
    MAPLE_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    MAPLE_CONFIG.numKeyValueHeads,
    MAPLE_CONFIG.headDim,
  ]);

  // QK-norm (RMSNorm on query and key, like Gemma).
  q = runRMSNorm(qNorm, q);
  k = runRMSNorm(kNorm, k);

  // Partial RoPE only on sliding attention layers; full attention uses NoPE.
  if (useRoPE) {
    [q, k] = applyPartialRoPE(q, k, 0);
  }

  const attn = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, MAPLE_CONFIG.numAttentionHeads * MAPLE_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

export function runAttentionStep(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: MapleAttention,
  cache: MapleKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
  useRoPE: boolean,
): { output: np.Array; cache: MapleKVCache } {
  const T = 1;
  let q = runLinear(qProj, x.ref).reshape([
    T,
    MAPLE_CONFIG.numAttentionHeads,
    MAPLE_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    MAPLE_CONFIG.numKeyValueHeads,
    MAPLE_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    MAPLE_CONFIG.numKeyValueHeads,
    MAPLE_CONFIG.headDim,
  ]);

  q = runRMSNorm(qNorm, q);
  k = runRMSNorm(kNorm, k);

  if (useRoPE) {
    [q, k] = applyPartialRoPE(q, k, position);
  }
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
    oProj,
    attn.reshape([T, MAPLE_CONFIG.numAttentionHeads * MAPLE_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}

export const runMapleAttentionPrefill = jit(
  function runMapleAttentionPrefill(
    { inputLayernorm, selfAttn }: MapleDecoderLayer,
    x: np.Array,
    useRoPE: boolean,
    capacity: number,
  ): [np.Array, MapleKVCache] {
    const residual = x.ref;
    const h = runRMSNorm(inputLayernorm, x);
    const { output: attnOut, key, value } = runAttentionPrefill(
      selfAttn,
      h,
      useRoPE,
    );
    return [residual.add(attnOut), padCache(key, value, capacity)];
  },
  { staticArgnums: [2, 3] },
);

export const runMapleAttentionStep = jit(
  function runMapleAttentionStep(
    { inputLayernorm, selfAttn }: MapleDecoderLayer,
    cache: MapleKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
    useRoPE: boolean,
  ): [np.Array, MapleKVCache] {
    const residual = x.ref;
    const h = runRMSNorm(inputLayernorm, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      selfAttn,
      cache,
      h,
      position,
      slot,
      validLength,
      useRoPE,
    );
    return [residual.add(attnOut), updatedCache];
  },
  { staticArgnums: [6, 7] },
);
