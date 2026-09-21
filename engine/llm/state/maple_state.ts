import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { MAPLE_CONFIG } from "../configs/maple_config.ts";
import {
  KV_CACHE_BLOCK_SIZE,
  type MapleKVCache,
} from "../cache/maple_cache.ts";

export type MapleState = {
  caches: MapleKVCache[];
  position: number;
  capacity: number;
};

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createMapleState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): MapleState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: MAPLE_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros([
        capacity,
        MAPLE_CONFIG.numKeyValueHeads,
        MAPLE_CONFIG.headDim,
      ], { dtype }),
      value: np.zeros([
        capacity,
        MAPLE_CONFIG.numKeyValueHeads,
        MAPLE_CONFIG.headDim,
      ], { dtype }),
    })),
  };
}

export function ensureMapleStateCapacity(
  state: MapleState,
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
