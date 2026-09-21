import { jit, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type BonsaiKVCache, padCache } from "../../cache/bonsai_cache.ts";
import {
  type BonsaiAttention,
  runAttentionPrefill,
  runAttentionStep,
} from "./bonsai_attention.ts";
import { type BonsaiMLP, runMLP } from "./bonsai_mlp.ts";
import { type BonsaiRMSNorm, runRMSNorm } from "./bonsai_rms_norm.ts";

export type BonsaiDecoderLayer = {
  inputLayernorm: BonsaiRMSNorm;
  postAttentionLayernorm: BonsaiRMSNorm;
  selfAttn: BonsaiAttention;
  mlp: BonsaiMLP;
};

export const runBonsaiDecoderLayerPrefill = jit(
  function runBonsaiDecoderLayerPrefill(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }:
      BonsaiDecoderLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, BonsaiKVCache] {
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

export const runBonsaiDecoderLayerStep = jit(
  function runBonsaiDecoderLayerStep(
    { inputLayernorm, postAttentionLayernorm, selfAttn, mlp }:
      BonsaiDecoderLayer,
    cache: BonsaiKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
  ): [np.Array, BonsaiKVCache] {
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
