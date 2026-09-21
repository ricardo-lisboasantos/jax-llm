import { nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type GemmaLinear, runLinear } from "./gemma_linear.ts";

export type GemmaMLP = {
  gateProj: GemmaLinear;
  upProj: GemmaLinear;
  downProj: GemmaLinear;
};

export function runMLP({ gateProj, upProj, downProj }: GemmaMLP, x: np.Array) {
  const gate = nn.gelu(runLinear(gateProj, x.ref), { approximate: true });
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
}
