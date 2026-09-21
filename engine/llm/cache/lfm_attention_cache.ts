import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
export type LfmAttentionCache = {
  kind: "attention";
  key: np.Array;
  value: np.Array;
};
