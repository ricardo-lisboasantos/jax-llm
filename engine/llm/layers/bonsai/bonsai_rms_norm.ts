import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { BONSAI_CONFIG } from "../../configs/bonsai_config.ts";

export type BonsaiRMSNorm = {
  weight: np.Array;
};

export const runRMSNorm = jit(
  function runRMSNorm(
    { weight }: BonsaiRMSNorm,
    x: np.Array,
    eps: number = BONSAI_CONFIG.rmsNormEps,
  ): np.Array {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
    x = x.div(np.sqrt(rms.add(eps)));
    return x.mul(weight.astype(np.float32)).astype(dtype);
  },
  { staticArgnums: [2] },
);
