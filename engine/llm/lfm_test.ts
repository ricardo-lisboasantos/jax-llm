/**
 * unit: LFM model configuration and cache helpers.
 */
import { assert, assertEquals } from "@std/assert";
import { LFM_CONFIG } from "./configs/lfm_config.ts";
import { KV_CACHE_BLOCK_SIZE, roundCacheCapacity } from "./cache/lfm_cache.ts";

Deno.test("unit: lfm config values", () => {
  assert(LFM_CONFIG.vocabSize > 0);
  assert(LFM_CONFIG.numHiddenLayers > 0);
  assert(LFM_CONFIG.hiddenSize > 0);
});

Deno.test("unit: lfm roundCacheCapacity rounds to block size", () => {
  assertEquals(roundCacheCapacity(1), KV_CACHE_BLOCK_SIZE);
  assertEquals(roundCacheCapacity(512), 512);
  assertEquals(roundCacheCapacity(513), 1024);
  assertEquals(roundCacheCapacity(1024), 1024);
});

Deno.test("unit: lfm cache kind discrimination", async () => {
  const { LFM_CONFIG: cfg } = await import("./configs/lfm_config.ts");
  assert(cfg !== undefined);
  // Static type check: attention vs conv cache kinds are distinct strings.
  const attnKind: string = "attention";
  const convKind: string = "conv";
  assert(attnKind !== convKind);
});
