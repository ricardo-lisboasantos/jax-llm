import {
  KV_CACHE_BLOCK_SIZE,
  type LfmCache,
  roundCacheCapacity,
} from "../cache/lfm_cache.ts";
import { PagedKVCache, type PagedKVCacheConfig } from "../cache/paged_cache.ts";
import { LFM_CONFIG } from "../configs/lfm_config.ts";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";

export type LfmState = {
  caches: LfmCache[];
  position: number;
  capacity: number;
  // Phase 3.0: Optional paged cache for long-context (8K+ tokens)
  pagedCache?: PagedKVCache;
  usePagedCache?: boolean;
  /**
   * Speculative-draft snapshot (PLD): `.ref`-bumped pre-score caches plus
   * the pre-score position. Caches are persistent-functional (replaced,
   * never mutated), so a snapshot is O(1) — no bytes are copied.
   * Set by `snapshotLfmDraft`, consumed by confirm/truncate.
   */
  draftSnapshot?: {
    position: number;
    caches: LfmCache[];
  };
};

export function createLfmState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
  usePagedCache = false,
  pagedCacheConfig = {},
}: {
  capacity?: number;
  dtype?: np.DType;
  usePagedCache?: boolean;
  pagedCacheConfig?: PagedKVCacheConfig;
} = {}): LfmState {
  capacity = roundCacheCapacity(capacity);

  // Phase 3.0: Initialize paged cache if requested (for sequences > 8K tokens)
  let pagedCache: PagedKVCache | undefined;
  if (usePagedCache && capacity > 8192) {
    pagedCache = new PagedKVCache({
      pageSize: 512,
      maxPages: Math.ceil(capacity / 512),
      headDim: LFM_CONFIG.headDim,
      numKvHeads: LFM_CONFIG.numKeyValueHeads,
      dtype: dtype === np.float16 ? "float16" : "float32",
      ...pagedCacheConfig,
    });
    pagedCache.initialize(LFM_CONFIG.numHiddenLayers);
  }

  return {
    capacity,
    position: 0,
    usePagedCache,
    pagedCache,
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

/**
 * Dispose paged cache if initialized (for session cleanup).
 * @param state LFM state with optional paged cache
 */
export function disposeLfmPagedCache(state: LfmState): void {
  if (state.pagedCache) {
    state.pagedCache.clear();
    state.pagedCache = undefined;
  }
}
