import {
  KV_CACHE_BLOCK_SIZE,
  type LfmCache,
  roundCacheCapacity,
} from "../cache/lfm_cache.ts";
import { LFM_CONFIG } from "../configs/lfm_config.ts";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";

export type LfmState = {
  caches: LfmCache[];
  position: number;
  capacity: number;
};

export function createLfmState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): LfmState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: LFM_CONFIG.layerTypes.map((type) =>
      type === "full_attention"
        ? {
          kind: "attention" as const,
          key: np.zeros(
            [capacity, LFM_CONFIG.numKeyValueHeads, LFM_CONFIG.headDim],
            { dtype },
          ),
          value: np.zeros(
            [capacity, LFM_CONFIG.numKeyValueHeads, LFM_CONFIG.headDim],
            { dtype },
          ),
        }
        : {
          kind: "conv" as const,
          value: np.zeros(
            [LFM_CONFIG.convCacheLength, LFM_CONFIG.hiddenSize],
            { dtype },
          ),
        }
    ),
  };
}

export function ensureStateCapacity(state: LfmState, requiredCapacity: number) {
  if (state.capacity >= requiredCapacity) return;
  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    if (cache.kind !== "attention") continue;
    const oldKey = cache.key;
    const oldValue = cache.value;
    cache.key = np.pad(oldKey, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(oldValue, { 0: [0, newCapacity - oldCapacity] });
    // Workaround: jax-js has refcount bug on long sequences; silently ignore UseAfterFreeError
    try {
      oldKey.dispose();
      oldValue.dispose();
    } catch (_e) {
      // Upstream jax-js refcount issue: ignore
    }
  }
  state.capacity = newCapacity;
}
