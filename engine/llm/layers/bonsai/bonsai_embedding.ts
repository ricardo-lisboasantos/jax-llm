import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import type { BonsaiLinear } from "./bonsai_linear.ts";

export const runEmbedding = jit(function runEmbedding(
  { weight }: BonsaiLinear,
  tokenIds: np.Array,
): np.Array {
  // Keep the residual stream in fp32; Bonsai is trained with bf16 activations.
  return weight.slice(tokenIds).astype(np.float32);
});
