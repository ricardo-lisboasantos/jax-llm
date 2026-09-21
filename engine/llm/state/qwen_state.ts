import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { QWEN_CONFIG } from "../configs/qwen_config.ts";
import type { QwenKVCache } from "../cache/qwen_cache.ts";

const KV_CACHE_BLOCK_SIZE = 512;

export type QwenState = {
  caches: QwenKVCache[];
  position: number;
  capacity: number;
};

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createQwenState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): QwenState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: QWEN_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros([
        capacity,
        QWEN_CONFIG.numKeyValueHeads,
        QWEN_CONFIG.headDim,
      ], { dtype }),
      value: np.zeros([
        capacity,
        QWEN_CONFIG.numKeyValueHeads,
        QWEN_CONFIG.headDim,
      ], { dtype }),
    })),
  };
}

export function ensureQwenStateCapacity(
  state: QwenState,
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
