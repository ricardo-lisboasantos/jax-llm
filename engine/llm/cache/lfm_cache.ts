import type { LfmAttentionCache } from "./lfm_attention_cache.ts";
import type { LfmConvCache } from "./lfm_conv_cache.ts";

/** Exponential growth steps for KV cache allocation. */
export const CACHE_GROWTH_STEPS = [128, 256, 512, 1024, 2048, 4096];
export const KV_CACHE_BLOCK_SIZE = 512; // Deprecated; use dynamic sizing

export type LfmCache = LfmAttentionCache | LfmConvCache;

/**
 * Allocate cache capacity with exponential growth.
 * Avoids over-allocation on short prompts while supporting long context.
 *
 * @param requiredCapacity Token count needed
 * @returns Next step size from exponential growth curve
 */
export function roundCacheCapacity(requiredCapacity: number): number {
  // Find the smallest step that fits requiredCapacity
  for (const step of CACHE_GROWTH_STEPS) {
    if (step >= requiredCapacity) return step;
  }
  // For capacity > 4096, continue doubling beyond the steps
  return Math.pow(2, Math.ceil(Math.log2(requiredCapacity)));
}
