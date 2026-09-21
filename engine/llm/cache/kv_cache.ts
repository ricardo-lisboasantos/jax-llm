export type KVCacheEntry<K = unknown, V = unknown> = {
  key: K;
  value: V;
};

export interface KVCache<K = unknown, V = unknown> {
  get(key: K): V | undefined;
  set(key: K, value: V): void;
  has(key: K): boolean;
  clear(): void;
  readonly size: number;
}

/** Simple in-memory KV cache used by tests and non-JAX fallbacks. */
export class MemoryKVCache<K, V> implements KVCache<K, V> {
  private store = new Map<K, V>();

  get(key: K): V | undefined {
    return this.store.get(key);
  }

  set(key: K, value: V): void {
    this.store.set(key, value);
  }

  has(key: K): boolean {
    return this.store.has(key);
  }

  clear(): void {
    this.store.clear();
  }

  get size(): number {
    return this.store.size;
  }
}
