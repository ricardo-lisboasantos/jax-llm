import { nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type QwenLinear, runLinear } from "./qwen_linear.ts";

export type QwenMLP = {
  gateProj: QwenLinear;
  upProj: QwenLinear;
  downProj: QwenLinear;
};

export function runMLP(
  { gateProj, upProj, downProj }: QwenMLP,
  x: np.Array,
): np.Array {
  const gate = nn.silu(runLinear(gateProj, x.ref));
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
}
