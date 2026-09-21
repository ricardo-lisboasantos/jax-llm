import { jit, numpy as np } from "npm:@jax-js/jax@^0.1.25";

export type MapleLinear = {
  weight: np.Array;
};

export const runLinear = jit(function runLinear(
  { weight }: MapleLinear,
  x: np.Array,
): np.Array {
  return np.dot(x, weight.transpose());
});
