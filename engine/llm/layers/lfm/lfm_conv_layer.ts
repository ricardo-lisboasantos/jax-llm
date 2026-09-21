import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { LFM_CONFIG } from "../../configs/lfm_config.ts";
import type { LfmLayerBase } from "./lfm_layer_base.ts";
import type { LfmShortConv } from "./lfm_short_conv.ts";
import { runLinear } from "./lfm_linear.ts";

import { runMLP } from "./lfm_mlp.ts";
import { runRMSNorm } from "./lfm_rms_norm.ts";

export type LfmConvLayer = LfmLayerBase & {
  conv: LfmShortConv;
};

export function runConvPrefill(
  { conv, inProj, outProj }: LfmShortConv,
  x: np.Array,
): { output: np.Array; cache: np.Array } {
  const T = x.shape[0];
  const [b, c, gate] = np.split(runLinear(inProj, x), 3, -1);
  const bx = b.mul(gate);
  const cache = T >= LFM_CONFIG.convCacheLength
    ? bx.ref.slice([T - LFM_CONFIG.convCacheLength], [])
    : np.pad(bx.ref, {
      0: [LFM_CONFIG.convCacheLength - T, 0],
    });

  const padded = np.pad(bx, {
    0: [LFM_CONFIG.convCacheLength - 1, 0],
  });
  const windows = np.stack(
    [
      padded.ref.slice([0, T], []),
      padded.ref.slice([1, T + 1], []),
      padded.slice([2, T + 2], []),
    ],
    -1,
  );
  const convOut = windows
    .mul(
      conv.weight.reshape([LFM_CONFIG.hiddenSize, LFM_CONFIG.convCacheLength]),
    )
    .sum(-1);
  return { output: runLinear(outProj, c.mul(convOut)), cache };
}

function runConvStep(
  { conv, inProj, outProj }: LfmShortConv,
  cache: np.Array,
  x: np.Array,
): { output: np.Array; cache: np.Array } {
  const [b, c, gate] = np.split(runLinear(inProj, x), 3, -1);
  const bx = b.mul(gate);
  const updatedCache = np.concatenate([cache.slice([1], []), bx], 0);
  const convOut = updatedCache.ref
    .mul(
      conv.weight
        .reshape([LFM_CONFIG.hiddenSize, LFM_CONFIG.convCacheLength])
        .transpose(),
    )
    .sum(0)
    .reshape([1, LFM_CONFIG.hiddenSize]);
  return {
    output: runLinear(outProj, c.mul(convOut)),
    cache: updatedCache,
  };
}

export const runConvLayerPrefill = jit(function runConvLayerPrefill(
  { operatorNorm, ffnNorm, feedForward, conv }: LfmConvLayer,
  x: np.Array,
): [np.Array, np.Array] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache } = runConvPrefill(conv, x);
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), cache];
});

export const runConvLayerStep = jit(function runConvLayerStep(
  { operatorNorm, ffnNorm, feedForward, conv }: LfmConvLayer,
  cache: np.Array,
  x: np.Array,
): [np.Array, np.Array] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache: updatedCache } = runConvStep(conv, cache, x);
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), updatedCache];
});
