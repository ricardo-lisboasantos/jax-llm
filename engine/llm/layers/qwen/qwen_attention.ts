import { nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { QWEN_CONFIG } from "../../configs/qwen_config.ts";
import { type QwenLinear, runLinear } from "./qwen_linear.ts";
import type { QwenKVCache } from "../../cache/qwen_cache.ts";
import { applyRoPE } from "./qwen_rope.ts";

export const ATTENTION_SCALE = 1 / Math.sqrt(QWEN_CONFIG.headDim);

export type QwenAttention = {
  qProj: QwenLinear;
  kProj: QwenLinear;
  vProj: QwenLinear;
  oProj: QwenLinear;
};

export function runAttentionPrefill(
  { qProj, kProj, vProj, oProj }: QwenAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  const q = runLinear(qProj, x.ref).reshape([
    T,
    QWEN_CONFIG.numAttentionHeads,
    QWEN_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    QWEN_CONFIG.numKeyValueHeads,
    QWEN_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    QWEN_CONFIG.numKeyValueHeads,
    QWEN_CONFIG.headDim,
  ]);

  const [qRot, kRot] = applyRoPE(q, k, 0);
  const attn = nn.dotProductAttention(qRot, kRot.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, QWEN_CONFIG.numAttentionHeads * QWEN_CONFIG.headDim]),
  );
  return { output, key: kRot, value: v };
}

export function runAttentionStep(
  { qProj, kProj, vProj, oProj }: QwenAttention,
  cache: QwenKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: QwenKVCache } {
  const T = 1;
  const q = runLinear(qProj, x.ref).reshape([
    T,
    QWEN_CONFIG.numAttentionHeads,
    QWEN_CONFIG.headDim,
  ]);
  const k = runLinear(kProj, x.ref).reshape([
    T,
    QWEN_CONFIG.numKeyValueHeads,
    QWEN_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    QWEN_CONFIG.numKeyValueHeads,
    QWEN_CONFIG.headDim,
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
    attn.reshape([T, QWEN_CONFIG.numAttentionHeads * QWEN_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}
