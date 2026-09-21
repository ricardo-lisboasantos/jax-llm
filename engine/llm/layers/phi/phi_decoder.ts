import { jit, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type PhiRMSNorm, runLayerNorm } from "./phi_rms_norm.ts";
import {
  type PhiAttention,
  runAttentionPrefill,
  runAttentionStep,
} from "./phi_attention.ts";
import { type PhiMLP, runMLP } from "./phi_mlp.ts";
import { padCache, type PhiKVCache } from "../../cache/phi_cache.ts";

export type PhiDecoderLayer = {
  inputLayernorm: PhiRMSNorm;
  selfAttn: PhiAttention;
  mlp: PhiMLP;
};

export const runPhiDecoderLayerPrefill = jit(
  function runPhiDecoderLayerPrefill(
    { inputLayernorm, selfAttn, mlp }: PhiDecoderLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, PhiKVCache] {
    // Pre-LN attention (Phi-2 has no post-attention norm — the residual
    // is added directly after attention, then the MLP norm is applied).
    const residual = x.ref;
    const h = runLayerNorm(inputLayernorm, x);
    const { output: attnOut, key, value } = runAttentionPrefill(selfAttn, h);
    x = residual.add(attnOut);

    // No separate post-attention layernorm — MLP follows directly.
    const residual2 = x.ref;
    const mlpOut = runMLP(mlp, x);
    return [residual2.add(mlpOut), padCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

export const runPhiDecoderLayerStep = jit(
  function runPhiDecoderLayerStep(
    { inputLayernorm, selfAttn, mlp }: PhiDecoderLayer,
    cache: PhiKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
  ): [np.Array, PhiKVCache] {
    const residual = x.ref;
    const h = runLayerNorm(inputLayernorm, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      selfAttn,
      cache,
      h,
      position,
      slot,
      validLength,
    );
    x = residual.add(attnOut);

    const residual2 = x.ref;
    const mlpOut = runMLP(mlp, x);
    return [residual2.add(mlpOut), updatedCache];
  },
  { staticArgnums: [6] },
);
