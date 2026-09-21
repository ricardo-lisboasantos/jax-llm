/**
 * @module runtime/offload/ram
 *
 * RAM (CPU) weight offloading: keeps inactive tensors in main memory
 * and pages them to the accelerator on demand.
 */

export type RamOffloadConfig = {
  /** Max bytes to keep pinned in RAM for offloaded weights. */
  maxBytes?: number;
};

/** Simple in-memory offload store backed by a Map. */
export class RamOffload {
  private store = new Map<string, Uint8Array>();
  readonly maxBytes: number;

  constructor(config: RamOffloadConfig = {}) {
    this.maxBytes = config.maxBytes ?? 1 << 30; // 1 GiB default
  }

  /** Current number of offloaded tensors. */
  get size(): number {
    return this.store.size;
  }

  put(key: string, data: Uint8Array): void {
    this.store.set(key, data);
  }

  get(key: string): Uint8Array | undefined {
    return this.store.get(key);
  }

  evict(key: string): boolean {
    return this.store.delete(key);
  }

  clear(): void {
    this.store.clear();
  }
}
