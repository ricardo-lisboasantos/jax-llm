import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
export type LfmConvCache = {
  kind: "conv";
  value: np.Array;
};
