import { nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type GemmaLinear, runLinear } from "./gemma_linear.ts";
import { type RMSNorm, runRMSNorm } from "./gemma_rms_norm.ts";
import { rotateHalf } from "../common/ops.ts";
import { GEMMA_CONFIG } from "../../configs/gemma_config.ts";
import type { GemmaKVCache } from "../../cache/gemma_cache.ts";

const ATTENTION_SCALE = 1 / Math.sqrt(GEMMA_CONFIG.queryPreAttnScalar);

export type GemmaAttention = {
  qProj: GemmaLinear;
  kProj: GemmaLinear;
  vProj: GemmaLinear;
  oProj: GemmaLinear;
  qNorm: RMSNorm;
  kNorm: RMSNorm;
};

export function applyGemmaRoPE(
  q: np.Array, // [T, num_heads, head_dim]
  k: np.Array, // [T, num_key_value_heads, head_dim]
  offset: number,
  theta: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(theta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq); // [T, head_dim / 2]

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

export function runAttentionPrefill(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: GemmaAttention,
  x: np.Array,
  ropeTheta: number,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numAttentionHeads,
    GEMMA_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);

  q = runRMSNorm(qNorm, q);
  k = runRMSNorm(kNorm, k);
  [q, k] = applyGemmaRoPE(q, k, 0, ropeTheta);

  const attn = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, GEMMA_CONFIG.numAttentionHeads * GEMMA_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

export function runAttentionStep(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: GemmaAttention,
  cache: GemmaKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
  ropeTheta: number,
): { output: np.Array; cache: GemmaKVCache } {
  const T = 1;
  let q = runLinear(qProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numAttentionHeads,
    GEMMA_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);

  q = runRMSNorm(qNorm, q);
  k = runRMSNorm(kNorm, k);
  [q, k] = applyGemmaRoPE(q, k, position, ropeTheta);

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
    attn.reshape([T, GEMMA_CONFIG.numAttentionHeads * GEMMA_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}
