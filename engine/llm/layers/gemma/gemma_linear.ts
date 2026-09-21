import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";

export type GemmaLinear = {
  weight: np.Array;
  bias?: np.Array;
};

export const runLinear = jit(function runLinear(
  { weight, bias }: GemmaLinear,
  x: np.Array,
): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
});
