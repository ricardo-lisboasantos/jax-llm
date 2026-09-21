/**
 * unit: logit sampling (temperature, topK, topP, repetition penalty).
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import { resolveSamplingDefaults, sampleLogits } from "./sampler.ts";
import type { SamplingOptions } from "./sampler.ts";

const baseOpts: SamplingOptions = {
  temperature: 0,
  topK: 64,
  topP: 1,
  repetitionPenalty: 1,
  previousTokens: [],
};

Deno.test("unit: sampler greedy picks argmax", () => {
  const logits = new Float32Array([0.1, 2.5, 1.0, -3.0]);
  assertEquals(sampleLogits(logits, baseOpts), 1);
});

Deno.test("unit: sampler greedy skips NaN logits", () => {
  const logits = new Float32Array([NaN, 1.0, NaN, 0.5]);
  assertEquals(sampleLogits(logits, baseOpts), 1);
});

Deno.test("unit: sampler greedy throws on all-NaN", () => {
  const logits = new Float32Array([NaN, NaN]);
  assertThrows(() => sampleLogits(logits, baseOpts), Error, "all-NaN");
});

Deno.test("unit: sampler greedy applies repetition penalty", () => {
  // Token 1 is previous: positive logit divided by penalty loses to token 0.
  const logits = new Float32Array([1.0, 1.1]);
  const id = sampleLogits(logits, {
    ...baseOpts,
    repetitionPenalty: 2,
    previousTokens: [1],
  });
  assertEquals(id, 0);
});

Deno.test("unit: sampler stochastic respects topK=1 (deterministic)", () => {
  const logits = new Float32Array([0.1, 5.0, 2.0]);
  for (let i = 0; i < 10; i++) {
    const id = sampleLogits(logits, { ...baseOpts, temperature: 1, topK: 1 });
    assertEquals(id, 1);
  }
});

Deno.test("unit: sampler stochastic throws on all-NaN", () => {
  const logits = new Float32Array([NaN, NaN, NaN]);
  assertThrows(
    () => sampleLogits(logits, { ...baseOpts, temperature: 1 }),
    Error,
    "all-NaN",
  );
});

Deno.test("unit: sampler stochastic returns id within vocab", () => {
  const logits = new Float32Array([1, 2, 3, 4]);
  const id = sampleLogits(logits, {
    ...baseOpts,
    temperature: 0.8,
    topK: 4,
    topP: 0.95,
  });
  assert(id >= 0 && id < 4, `id ${id} out of range`);
});

Deno.test("unit: resolveSamplingDefaults merges overrides", () => {
  const base = { temperature: 0.8, topK: 64, topP: 0.95, repetitionPenalty: 1 };
  const merged = resolveSamplingDefaults(base, { temperature: 0.5 });
  assertEquals(merged, {
    temperature: 0.5,
    topK: 64,
    topP: 0.95,
    repetitionPenalty: 1,
  });
  // No overrides returns a copy of base.
  assertEquals(resolveSamplingDefaults(base), base);
});
