import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { PHI_CONFIG } from "../configs/phi_config.ts";
import type { PhiKVCache } from "../cache/phi_cache.ts";

const KV_CACHE_BLOCK_SIZE = 512;

export type PhiState = {
  caches: PhiKVCache[];
  position: number;
  capacity: number;
};

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createPhiState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): PhiState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: PHI_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros(
        [capacity, PHI_CONFIG.numKeyValueHeads, PHI_CONFIG.headDim],
        { dtype },
      ),
      value: np.zeros([
        capacity,
        PHI_CONFIG.numKeyValueHeads,
        PHI_CONFIG.headDim,
      ], { dtype }),
    })),
  };
}

export function ensurePhiStateCapacity(
  state: PhiState,
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
