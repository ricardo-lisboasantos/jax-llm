import { jit, nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type MapleLinear, runLinear } from "./maple_linear.ts";

export type ExpertMLP = {
  gateProj: MapleLinear;
  upProj: MapleLinear;
  downProj: MapleLinear;
};

export const runExpertMLP = jit(function runExpertMLP(
  { gateProj, upProj, downProj }: ExpertMLP,
  x: np.Array,
): np.Array {
  const gate = nn.silu(runLinear(gateProj, x.ref));
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
});
