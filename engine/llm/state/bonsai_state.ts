import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { BONSAI_CONFIG } from "../configs/bonsai_config.ts";
import {
  type BonsaiKVCache,
  KV_CACHE_BLOCK_SIZE,
} from "../cache/bonsai_cache.ts";
export type { BonsaiKVCache };

export type BonsaiState = {
  caches: BonsaiKVCache[];
  position: number;
  capacity: number;
};

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createBonsaiState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): BonsaiState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: BONSAI_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros([
        capacity,
        BONSAI_CONFIG.numKeyValueHeads,
        BONSAI_CONFIG.headDim,
      ], { dtype }),
      value: np.zeros([
        capacity,
        BONSAI_CONFIG.numKeyValueHeads,
        BONSAI_CONFIG.headDim,
      ], { dtype }),
    })),
  };
}

export function ensureBonsaiStateCapacity(
  state: BonsaiState,
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
