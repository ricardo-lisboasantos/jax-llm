/**
 * @module runtime/offload/ram
 *
 * RAM (CPU) weight offloading: keeps inactive tensors in main memory
 * and pages them to the accelerator on demand.
 */

/** Configuration for {@linkcode RamOffload}. */
export type RamOffloadConfig = {
  /** Max bytes to keep pinned in RAM for offloaded weights. */
  maxBytes?: number;
};

/** Simple in-memory offload store backed by a Map. */
export class RamOffload {
  private store = new Map<string, Uint8Array>();
  /** Max bytes this store budgets (enforced by the owning pager). */
  readonly maxBytes: number;

  /** Create a store with an optional RAM budget (default 1 GiB). */
  constructor(config: RamOffloadConfig = {}) {
    this.maxBytes = config.maxBytes ?? 1 << 30; // 1 GiB default
  }

  /** Current number of offloaded tensors. */
  get size(): number {
    return this.store.size;
  }

  /** Store raw bytes under a key (overwrites). */
  put(key: string, data: Uint8Array): void {
    this.store.set(key, data);
  }

  /** Fetch stored bytes, or `undefined` on miss. */
  get(key: string): Uint8Array | undefined {
    return this.store.get(key);
  }

  /** Drop one entry; returns whether it existed. */
  evict(key: string): boolean {
    return this.store.delete(key);
  }

  /** Drop all entries. */
  clear(): void {
    this.store.clear();
  }
}
