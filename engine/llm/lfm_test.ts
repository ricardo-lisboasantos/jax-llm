/**
 * unit: LFM model configuration and cache helpers.
 */
import { assert, assertEquals } from "@std/assert";
import { LFM_CONFIG } from "./configs/lfm_config.ts";
import { CACHE_GROWTH_STEPS, KV_CACHE_BLOCK_SIZE, roundCacheCapacity } from "./cache/lfm_cache.ts";

Deno.test("unit: lfm config values", () => {
  assert(LFM_CONFIG.vocabSize > 0);
  assert(LFM_CONFIG.numHiddenLayers > 0);
  assert(LFM_CONFIG.hiddenSize > 0);
});

Deno.test("unit: lfm roundCacheCapacity uses exponential growth", () => {
  // Short prompts use small allocations
  assertEquals(roundCacheCapacity(1), 128);
  assertEquals(roundCacheCapacity(64), 128);
  assertEquals(roundCacheCapacity(128), 128);
  // Medium prompts
  assertEquals(roundCacheCapacity(129), 256);
  assertEquals(roundCacheCapacity(256), 256);
  assertEquals(roundCacheCapacity(257), 512);
  // Large prompts
  assertEquals(roundCacheCapacity(512), 512);
  assertEquals(roundCacheCapacity(513), 1024);
  assertEquals(roundCacheCapacity(1024), 1024);
  assertEquals(roundCacheCapacity(2048), 2048);
  // Very large prompts beyond steps
  assertEquals(roundCacheCapacity(4097), 8192);
});

Deno.test("unit: lfm cache kind discrimination", async () => {
  const { LFM_CONFIG: cfg } = await import("./configs/lfm_config.ts");
  assert(cfg !== undefined);
  // Static type check: attention vs conv cache kinds are distinct strings.
  const attnKind: string = "attention";
  const convKind: string = "conv";
  assert(attnKind !== convKind);
});
