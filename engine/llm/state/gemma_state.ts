import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import {
  type GemmaKVCache,
  KV_CACHE_BLOCK_SIZE,
  roundCacheCapacity,
} from "../cache/gemma_cache.ts";
import { GEMMA_CONFIG } from "../configs/gemma_config.ts";

export type GemmaState = {
  caches: GemmaKVCache[];
  position: number;
  capacity: number;
};

export function createGemmaState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): GemmaState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: GEMMA_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros(
        [capacity, GEMMA_CONFIG.numKeyValueHeads, GEMMA_CONFIG.headDim],
        { dtype },
      ),
      value: np.zeros(
        [capacity, GEMMA_CONFIG.numKeyValueHeads, GEMMA_CONFIG.headDim],
        { dtype },
      ),
    })),
  };
}

export function ensureGemmaStateCapacity(
  state: GemmaState,
  requiredCapacity: number,
) {
  if (state.capacity >= requiredCapacity) return;

  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    cache.key = np.pad(cache.key, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(cache.value, { 0: [0, newCapacity - oldCapacity] });
  }
  state.capacity = newCapacity;
}
