import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { GPT_CONFIG } from "../configs/gpt_config.ts";

const KV_CACHE_BLOCK_SIZE = 512;

export type GptKVCache = {
  key: np.Array;
  value: np.Array;
};

export type GptState = {
  caches: GptKVCache[];
  position: number;
  capacity: number;
};

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createGptState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): GptState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: GPT_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros([
        capacity,
        GPT_CONFIG.numAttentionHeads,
        GPT_CONFIG.headDim,
      ], { dtype }),
      value: np.zeros([
        capacity,
        GPT_CONFIG.numAttentionHeads,
        GPT_CONFIG.headDim,
      ], { dtype }),
    })),
  };
}

export function ensureGptStateCapacity(
  state: GptState,
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
