import type { LfmAttentionCache } from "./lfm_attention_cache.ts";
import type { LfmConvCache } from "./lfm_conv_cache.ts";

export const KV_CACHE_BLOCK_SIZE = 512;

export type LfmCache = LfmAttentionCache | LfmConvCache;

export function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}
