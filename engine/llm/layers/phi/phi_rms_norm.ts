import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { PHI_CONFIG } from "../../configs/phi_config.ts";

export type PhiRMSNorm = {
  weight: np.Array;
  bias: np.Array;
};

export const runLayerNorm = jit(
  function runLayerNorm(
    { weight, bias }: PhiRMSNorm,
    x: np.Array,
    eps: number = PHI_CONFIG.layerNormEps,
  ): np.Array {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const mean = x.ref.mean(-1, { keepdims: true });
    const diff = x.ref.sub(mean);
    const variance = diff.mul(diff).mean(-1, { keepdims: true });
    x = x.sub(mean).div(np.sqrt(variance.add(eps)));
    return x.mul(weight.astype(np.float32)).add(bias.astype(np.float32)).astype(
      dtype,
    );
  },
  { staticArgnums: [2] },
);
