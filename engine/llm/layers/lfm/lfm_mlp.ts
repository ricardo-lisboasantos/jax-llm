import { nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type LfmLinear, runLinear } from "./lfm_linear.ts";

export type LfmMLP = {
  w1: LfmLinear;
  w2: LfmLinear;
  w3: LfmLinear;
};

export function runMLP({ w1, w2, w3 }: LfmMLP, x: np.Array): np.Array {
  const gate = nn.silu(runLinear(w1, x.ref));
  const up = runLinear(w3, x);
  return runLinear(w2, gate.mul(up));
}
