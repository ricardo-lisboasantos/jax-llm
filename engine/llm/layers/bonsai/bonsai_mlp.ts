import { nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type BonsaiLinear, runLinear } from "./bonsai_linear.ts";

export type BonsaiMLP = {
  gateProj: BonsaiLinear;
  upProj: BonsaiLinear;
  downProj: BonsaiLinear;
};

export function runMLP(
  { gateProj, upProj, downProj }: BonsaiMLP,
  x: np.Array,
): np.Array {
  const gate = nn.silu(runLinear(gateProj, x.ref));
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
}
