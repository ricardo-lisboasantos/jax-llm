import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import type { MapleLinear } from "./maple_linear.ts";

export const runEmbedding = jit(function runEmbedding(
  { weight }: MapleLinear,
  tokenIds: np.Array,
): np.Array {
  return weight.slice(tokenIds).astype(np.float32);
});
