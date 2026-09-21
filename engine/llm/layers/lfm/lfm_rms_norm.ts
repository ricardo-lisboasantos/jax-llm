import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { LFM_CONFIG } from "../../configs/lfm_config.ts";

export type RMSNorm = {
  weight: np.Array;
};

export const runRMSNorm = jit(function runRMSNorm(
  { weight }: RMSNorm,
  x: np.Array,
): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  x = x.div(np.sqrt(rms.add(LFM_CONFIG.rmsNormEps)));
  return x.mul(weight.astype(np.float32)).astype(dtype);
});
