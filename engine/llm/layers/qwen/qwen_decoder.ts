import { jit, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type QwenRMSNorm, runRMSNorm } from "./qwen_rms_norm.ts";
import {
  type QwenAttention,
  runAttentionPrefill,
  runAttentionStep,
} from "./qwen_attention.ts";
import { type QwenMLP, runMLP } from "./qwen_mlp.ts";
import { padCache, type QwenKVCache } from "../../cache/qwen_cache.ts";

export type QwenDecoderLayer = {
  inputLayernorm: QwenRMSNorm;
  postAttentionLayernorm: QwenRMSNorm;
  selfAttn: QwenAttention;
  mlp: QwenMLP;
};

export const runQwenDecoderLayerPrefill = jit(
  function runQwenDecoderLayerPrefill(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }: QwenDecoderLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, QwenKVCache] {
    const residual = x.ref;
    x = runRMSNorm(inputLayernorm, x);
    const { output: attnOut, key, value } = runAttentionPrefill(selfAttn, x);
    x = residual.add(attnOut);

    const residual2 = x.ref;
    x = runRMSNorm(postAttentionLayernorm, x);
    x = runMLP(mlp, x);
    return [residual2.add(x), padCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

export const runQwenDecoderLayerStep = jit(
  function runQwenDecoderLayerStep(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }: QwenDecoderLayer,
    cache: QwenKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
  ): [np.Array, QwenKVCache] {
    const residual = x.ref;
    x = runRMSNorm(inputLayernorm, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      selfAttn,
      cache,
      x,
      position,
      slot,
      validLength,
    );
    x = residual.add(attnOut);

    const residual2 = x.ref;
    x = runRMSNorm(postAttentionLayernorm, x);
    x = runMLP(mlp, x);
    return [residual2.add(x), updatedCache];
  },
  { staticArgnums: [6] },
);
