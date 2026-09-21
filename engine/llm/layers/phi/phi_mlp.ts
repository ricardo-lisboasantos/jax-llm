import { nn, type numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type PhiLinear, runLinear } from "./phi_linear.ts";

export type PhiMLP = {
  fc1: PhiLinear;
  fc2: PhiLinear;
};

export function runMLP({ fc1, fc2 }: PhiMLP, x: np.Array): np.Array {
  const h = nn.gelu(runLinear(fc1, x.ref), { approximate: true });
  return runLinear(fc2, h);
}
