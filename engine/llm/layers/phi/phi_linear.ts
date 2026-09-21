import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";

export type PhiLinear = {
  weight: np.Array;
  bias: np.Array;
};

export const runLinear = jit(function runLinear(
  { weight, bias }: PhiLinear,
  x: np.Array,
): np.Array {
  x = np.dot(x, weight.transpose());
  return x.add(bias);
});
