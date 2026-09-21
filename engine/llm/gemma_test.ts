/**
 * unit: Gemma model configuration and state helpers (no device needed).
 */
import { assert, assertEquals } from "@std/assert";
import { GEMMA_CONFIG } from "./configs/gemma_config.ts";
import { layerRopeTheta } from "./layers/gemma/gemma_rope_theta.ts";
import { runEmbedding } from "./layers/gemma/gemma_embedding.ts";
import { runRMSNorm } from "./layers/gemma/gemma_rms_norm.ts";
import { runLinear } from "./layers/gemma/gemma_linear.ts";
import { runMLP } from "./layers/gemma/gemma_mlp.ts";

Deno.test("unit: gemma config values", () => {
  assertEquals(GEMMA_CONFIG.vocabSize, 262_144);
  assertEquals(GEMMA_CONFIG.numHiddenLayers, 18);
  assertEquals(GEMMA_CONFIG.numAttentionHeads, 4);
  assertEquals(GEMMA_CONFIG.layerTypes.length, 18);
});

Deno.test("unit: gemma rope theta per layer type", () => {
  // Layer 0 is sliding_attention -> local base freq.
  assertEquals(layerRopeTheta(0), GEMMA_CONFIG.ropeLocalBaseFreq);
  // Layer 5 is full_attention -> global theta.
  assertEquals(layerRopeTheta(5), GEMMA_CONFIG.ropeTheta);
});

Deno.test("unit: gemma layer fns are defined", () => {
  assert(typeof runEmbedding === "function");
  assert(typeof runRMSNorm === "function");
  assert(typeof runLinear === "function");
  assert(typeof runMLP === "function");
});
