import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type GemmaKVCache, padCache } from "../../cache/gemma_cache.ts";
import {
  type GemmaAttention,
  runAttentionPrefill,
  runAttentionStep,
} from "./gemma_attention.ts";
import type { GemmaMLP } from "./gemma_mlp.ts";
import { type RMSNorm, runRMSNorm } from "./gemma_rms_norm.ts";
import { runMLP } from "./gemma_mlp.ts";

export type GemmaDecoderLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  preFeedforwardLayernorm: RMSNorm;
  postFeedforwardLayernorm: RMSNorm;
  selfAttn: GemmaAttention;
  mlp: GemmaMLP;
};

function refRMSNorm({ weight }: RMSNorm): RMSNorm {
  return { weight: weight.ref };
}

function refLinear({ weight, bias }: { weight: np.Array; bias?: np.Array }) {
  const out: { weight: np.Array; bias?: np.Array } = { weight: weight.ref };
  if (bias !== undefined) out.bias = bias.ref;
  return out;
}

function refAttention(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: GemmaAttention,
): GemmaAttention {
  return {
    qProj: refLinear(qProj),
    kProj: refLinear(kProj),
    vProj: refLinear(vProj),
    oProj: refLinear(oProj),
    qNorm: refRMSNorm(qNorm),
    kNorm: refRMSNorm(kNorm),
  };
}

function refMLP({ gateProj, upProj, downProj }: GemmaMLP): GemmaMLP {
  return {
    gateProj: refLinear(gateProj),
    upProj: refLinear(upProj),
    downProj: refLinear(downProj),
  };
}

function refLayer({
  inputLayernorm,
  postAttentionLayernorm,
  preFeedforwardLayernorm,
  postFeedforwardLayernorm,
  selfAttn,
  mlp,
}: GemmaDecoderLayer): GemmaDecoderLayer {
  return {
    inputLayernorm: refRMSNorm(inputLayernorm),
    postAttentionLayernorm: refRMSNorm(postAttentionLayernorm),
    preFeedforwardLayernorm: refRMSNorm(preFeedforwardLayernorm),
    postFeedforwardLayernorm: refRMSNorm(postFeedforwardLayernorm),
    selfAttn: refAttention(selfAttn),
    mlp: refMLP(mlp),
  };
}

export function runGemmaDecoderLayerPrefill(
  layer: GemmaDecoderLayer,
  x: np.Array,
  ropeTheta: number,
  capacity: number,
): [np.Array, GemmaKVCache] {
  const {
    inputLayernorm,
    postAttentionLayernorm,
    preFeedforwardLayernorm,
    postFeedforwardLayernorm,
    selfAttn,
    mlp,
  } = refLayer(layer);

  const residual = x.ref;
  x = runRMSNorm(inputLayernorm, x);
  const {
    output: attnOut,
    key,
    value,
  } = runAttentionPrefill(selfAttn, x, ropeTheta);
  x = runRMSNorm(postAttentionLayernorm, attnOut);
  x = residual.add(x);

  const residual2 = x.ref;
  x = runRMSNorm(preFeedforwardLayernorm, x);
  x = runMLP(mlp, x);
  x = runRMSNorm(postFeedforwardLayernorm, x);
  x = residual2.add(x);

  return [x, padCache(key, value, capacity)];
}

export function runGemmaDecoderLayerStep(
  layer: GemmaDecoderLayer,
  cache: GemmaKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
  ropeTheta: number,
): [np.Array, GemmaKVCache] {
  const {
    inputLayernorm,
    postAttentionLayernorm,
    preFeedforwardLayernorm,
    postFeedforwardLayernorm,
    selfAttn,
    mlp,
  } = refLayer(layer);

  const residual = x.ref;
  x = runRMSNorm(inputLayernorm, x);
  const { output: attnOut, cache: updatedCache } = runAttentionStep(
    selfAttn,
    cache,
    x,
    position,
    slot,
    validLength,
    ropeTheta,
  );
  x = runRMSNorm(postAttentionLayernorm, attnOut);
  x = residual.add(x);

  const residual2 = x.ref;
  x = runRMSNorm(preFeedforwardLayernorm, x);
  x = runMLP(mlp, x);
  x = runRMSNorm(postFeedforwardLayernorm, x);
  x = residual2.add(x);

  return [x, updatedCache];
}
