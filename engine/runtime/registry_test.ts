/**
 * unit: model registry & resolution.
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  DEFAULT_MODEL_ID,
  isAlias,
  isBuiltInModel,
  isHuggingFaceRepo,
  MODEL_IDS,
  resolveModel,
  tokenizerUrlCandidates,
} from "./registry.ts";

Deno.test("unit: registry lists built-in models", () => {
  assert(MODEL_IDS.length >= 7, `expected >=7 models, got ${MODEL_IDS.length}`);
  assert(MODEL_IDS.includes("lfm2.5-350m"));
  assert(MODEL_IDS.includes("gemma-3-270m"));
  assertEquals(DEFAULT_MODEL_ID, "lfm2.5-350m");
});

Deno.test("unit: isBuiltInModel / isAlias / isHuggingFaceRepo guards", () => {
  assert(isBuiltInModel("lfm2.5-350m"));
  assert(!isBuiltInModel("lfm"));
  assert(isAlias("lfm"));
  assert(isAlias("gemma"));
  assert(!isAlias("lfm2.5-350m"));
  assert(isHuggingFaceRepo("org/model-name"));
  assert(!isHuggingFaceRepo("gemma"));
  assert(!isHuggingFaceRepo("https://example.com/x"));
});

Deno.test("unit: resolveModel built-in id", () => {
  const def = resolveModel("gemma-3-270m");
  assertEquals(def.id, "gemma-3-270m");
  assert(def.weightsUrl.length > 0);
  assert(def.tokenizerUrl.length > 0);
});

Deno.test("unit: resolveModel short aliases", () => {
  assertEquals(resolveModel("lfm").id, "lfm2.5-350m");
  assertEquals(resolveModel("gemma").id, "gemma-3-270m");
  assertEquals(resolveModel("qwen").id, "qwen2.5-0.5b");
});

Deno.test("unit: resolveModel applies URL overrides", () => {
  const def = resolveModel("lfm2.5-350m", {
    weightOverrides: { "lfm2.5-350m": "https://mirror/weights" },
    tokenizerOverrides: { "lfm2.5-350m": "https://mirror/tokenizer" },
  });
  assertEquals(def.weightsUrl, "https://mirror/weights");
  assertEquals(def.tokenizerUrl, "https://mirror/tokenizer");
});

Deno.test("unit: resolveModel dynamic HuggingFace repo", () => {
  const def = resolveModel("my-org/my-model");
  assertEquals(def.id, "my-org/my-model");
  assert(def.weightsUrl.includes("huggingface.co/my-org/my-model"));
  assert(def.tokenizerUrl.includes("huggingface.co/my-org/my-model"));
  assertEquals(def.contextSize, 4096);
});

Deno.test("unit: resolveModel throws on unknown id", () => {
  assertThrows(() => resolveModel("no-such-model"), Error, "Unknown model");
});

Deno.test("unit: tokenizerUrlCandidates expands base", () => {
  const out = tokenizerUrlCandidates(
    "https://hf.co/org/m",
    "https://hf.co/org/m/resolve/main/tokenizer.json",
  );
  assertEquals(out.length, 4);
  assert(out.includes("https://hf.co/org/m/tokenizer.model"));
  assertEquals(tokenizerUrlCandidates(undefined, "https://x/y"), [
    "https://x/y",
  ]);
});
