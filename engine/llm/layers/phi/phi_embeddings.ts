import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";

export const runEmbedding = jit(function runEmbedding(
  { weight }: { weight: np.Array },
  tokenIds: np.Array,
): np.Array {
  return weight.slice(tokenIds).astype(np.float32);
});
