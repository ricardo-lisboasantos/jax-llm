import { jit, nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { type MapleLinear, runLinear } from "./maple_linear.ts";
import { type ExpertMLP, runExpertMLP } from "./maple_mlp.ts";
import type { PagedExpertStore } from "./paged_experts.ts";
import { MAPLE_CONFIG } from "../../configs/maple_config.ts";
import type { MapleDecoderLayer } from "./maple_decoder.ts";
import { runRMSNorm } from "./maple_rms_norm.ts";

export type MapleMoE = {
  gate: MapleLinear; // router: [hidden, num_experts]
  experts: ExpertMLP[];
  /**
   * Paged backing (B1.3). When present, `experts` stays empty and weights
   * hydrate on demand through the store. Absent = dense legacy path.
   */
  expertStore?: PagedExpertStore;
};

/**
 * Compute the MoE output for all tokens (top-k sparse).
 *
 * Only the union of selected experts across the batch is evaluated
 * (≤ T×k, typically 8 on decode) instead of all 256 — mathematically
 * identical, since non-selected experts contribute weight 0.
 * The host read is one tiny [T, k] int sync per layer; expert MLPs
 * (2048→512→2048 each) dominate by orders of magnitude.
 */
export async function runMoE(
  { gate, experts, expertStore }: MapleMoE,
  x: np.Array, // [T, hidden]
): Promise<np.Array> {
  const [T, H] = x.shape;
  const E = MAPLE_CONFIG.numExperts;
  const k = MAPLE_CONFIG.numExpertsPerTok;

  // Router logits: [T, E]
  const logits = runLinear(gate, x.ref);

  // Top-k indices (descending = highest values last in ascending sort).
  // `.ref` at each use: primitives consume move refs (repo convention).
  const sortedIndices = np.argsort(logits.ref, -1); // [T, E] ascending
  const topKIndices = sortedIndices.slice([], [E - k]).astype(np.uint32); // [T, k]

  // Top-k logits and normalized weights (softmax over selected experts).
  const topKLogits = np.takeAlongAxis(
    logits.ref,
    topKIndices.ref.astype(np.int32),
    -1,
  );
  const topKWeights = nn.softmax(topKLogits, -1); // [T, k]

  // Accumulate weighted expert outputs — only for selected experts.
  let output = np.zeros([T, H], { dtype: x.dtype });

  // Sparse expert set: host-side unique of the tiny [T, k] index matrix.
  // Non-selected experts have weight 0 for every token, so skipping them
  // is exact. Decode (T=1) evaluates 8 of 256 experts (32x fewer MLPs).
  const idxData = topKIndices.ref.dataSync() as ArrayLike<number>;
  const selected = [...new Set(Array.from(idxData))].filter(
    (e) => e >= 0 && e < E,
  );

  for (const e of selected) {
    // Paged path hydrates on demand; dense path indexes directly.
    const expert = expertStore ? await expertStore.get(e) : experts[e];
    // Expert output for all tokens: [T, hidden]
    const expertOut = runExpertMLP(expert, x.ref);

    // Weight mask: for each token, the routing weight for expert e (0 if not selected).
    const expertIdx = np.full([T, k], e, { dtype: np.uint32 });
    const matchMask = np.equal(topKIndices.ref, expertIdx).astype(x.dtype);
    const expertWeights = matchMask.mul(topKWeights.ref).sum(-1).reshape([
      T,
      1,
    ]);

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
