import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import type { LfmLinear } from "./lfm_linear.ts";

export const runEmbedding = jit(function runEmbedding(
  { weight }: LfmLinear,
  tokenIds: np.Array,
): np.Array {
  // Keep the residual stream in fp32. LFM is trained with bf16 activations,
  // whose range cannot be represented safely by fp16.
  return weight.slice(tokenIds).astype(np.float32);
});
