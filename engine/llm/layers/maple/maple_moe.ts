import { jit, nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type MapleLinear, runLinear } from "./maple_linear.ts";
import { type ExpertMLP, runExpertMLP } from "./maple_mlp.ts";
import { MAPLE_CONFIG } from "../../configs/maple_config.ts";
import type { MapleDecoderLayer } from "./maple_decoder.ts";
import { runRMSNorm } from "./maple_rms_norm.ts";

export type MapleMoE = {
  gate: MapleLinear; // router: [hidden, num_experts]
  experts: ExpertMLP[];
};

/**
 * Compute the MoE output for all tokens.
 *
 * For each of the `numExperts` experts, computes its MLP output for all
 * tokens, then accumulates the weighted result.  The weight for each
 * (token, expert) pair is the normalized top-k routing probability if
 * that expert was selected, or 0 otherwise.
 *
 * This evaluates all experts for all tokens, which is O(E) expert
 * evaluations.  For inference with a small number of tokens this is
 * acceptable; for large prefill batches it is expensive.
 */
export function runMoE(
  { gate, experts }: MapleMoE,
  x: np.Array, // [T, hidden]
): np.Array {
  const [T, H] = x.shape;
  const E = MAPLE_CONFIG.numExperts;
  const k = MAPLE_CONFIG.numExpertsPerTok;

  // Router logits: [T, E]
  const logits = runLinear(gate, x);

  // Top-k indices (descending = highest values last in ascending sort).
  const sortedIndices = np.argsort(logits, -1); // [T, E] ascending
  const topKIndices = sortedIndices.slice([], [E - k]).astype(np.uint32); // [T, k]

  // Top-k logits and normalized weights (softmax over selected experts).
  const topKLogits = np.takeAlongAxis(logits, topKIndices.astype(np.int32), -1);
  const topKWeights = nn.softmax(topKLogits, -1); // [T, k]

  // Accumulate weighted expert outputs.
  let output = np.zeros([T, H], { dtype: x.dtype });

  for (let e = 0; e < E; e++) {
    // Expert output for all tokens: [T, hidden]
    const expertOut = runExpertMLP(experts[e], x.ref);

    // Weight mask: for each token, the routing weight for expert e (0 if not selected).
    const expertIdx = np.full([T, k], e, { dtype: np.uint32 });
    const matchMask = np.equal(topKIndices, expertIdx).astype(x.dtype);
    const expertWeights = matchMask.mul(topKWeights).sum(-1).reshape([T, 1]);

    output = output.add(expertOut.mul(expertWeights));
  }

  return output;
}

export const runMapleMoENorm = jit(function runMapleMoENorm(
  { postAttentionLayernorm }: MapleDecoderLayer,
  x: np.Array,
): np.Array {
  return runRMSNorm(postAttentionLayernorm, x);
});
