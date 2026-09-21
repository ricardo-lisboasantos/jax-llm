import { numpy as np } from "npm:@jax-js/jax@^0.1.25";

export const KV_CACHE_BLOCK_SIZE = 512;

export type MapleKVCache = {
  key: np.Array; // [capacity, num_key_value_heads, head_dim]
  value: np.Array; // [capacity, num_key_value_heads, head_dim]
};

export function padCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): MapleKVCache {
  const T = key.shape[0];
  if (T > capacity) {
    throw new Error(`Prompt length ${T} exceeds cache capacity ${capacity}`);
  }
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

export function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}
