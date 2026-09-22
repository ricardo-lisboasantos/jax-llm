/**
 * @module llm/profiling/buffer_pool
 *
 * WebGPU buffer pool for reusable GPU memory allocation.
 * Reduces p90 latency by pre-allocating persistent buffers instead of
 * alloc/dealloc per inference step.
 *
 * Goals:
 *  - Pre-allocate a fixed set of buffers at initialization
 *  - Reuse across multiple inference steps
 *  - Track allocation/deallocation to prevent leaks
 *  - Reduce p90 latency variance from allocation jitter
 *
 * Usage:
 * ```typescript
 * const pool = new BufferPool({ initialSize: 10, maxSize: 100 });
 * const buf = await pool.allocate(1024); // Reuses or creates
 * // Use buffer...
 * pool.release(buf);
 * ```
 */

export interface PooledBuffer {
  /** Unique buffer ID. */
  id: string;
  /** Size in bytes. */
  size: number;
  /** Whether this buffer is currently allocated. */
  allocated: boolean;
  /** Timestamp when last released (for eviction policy). */
  releasedAtMs?: number;
  /** Reference count for nested allocation scenarios. */
  refCount: number;
}

export interface BufferPoolConfig {
  /** Number of buffers to pre-allocate at startup. */
  initialSize?: number;
  /** Maximum number of buffers in the pool. */
  maxSize?: number;
  /** Default buffer size in bytes. */
  defaultBufferSize?: number;
  /** Enable aggressive cleanup when pool reaches maxSize. */
  aggressiveCleanup?: boolean;
}

export interface PoolStats {
  totalBuffers: number;
  allocatedBuffers: number;
  freeBuffers: number;
  totalBytes: number;
  allocatedBytes: number;
  freeBytes: number;
}

/**
 * GPU buffer pool for reducing allocation/deallocation jitter.
 * Implements a simple free-list strategy with optional LRU eviction.
 */
export class BufferPool {
  private config: Required<BufferPoolConfig>;
  private buffers = new Map<string, PooledBuffer>();
  private freeList: string[] = [];
  private allocationOrder: string[] = [];
  private nextBufferId = 0;

  constructor(config: BufferPoolConfig = {}) {
    this.config = {
      initialSize: config.initialSize ?? 10,
      maxSize: config.maxSize ?? 100,
      defaultBufferSize: config.defaultBufferSize ?? 8192,
      aggressiveCleanup: config.aggressiveCleanup ?? false,
    };
  }

  /**
   * Pre-allocate buffers in the pool.
   * Should be called once during initialization.
   */
  prewarmPool(): void {
    for (let i = 0; i < this.config.initialSize; i++) {
      const id = `buf_${this.nextBufferId++}`;
      const buf: PooledBuffer = {
        id,
        size: this.config.defaultBufferSize,
        allocated: false,
        refCount: 0,
        releasedAtMs: performance.now(),
      };
      this.buffers.set(id, buf);
      this.freeList.push(id);
    }
  }

  /**
   * Allocate a buffer from the pool (reuses or creates new).
   * @param size Requested size in bytes
   * @returns Pooled buffer identifier
   */
  allocate(size: number): string {
    // Try to find a free buffer of sufficient size
    for (const bufferId of this.freeList) {
      const buf = this.buffers.get(bufferId);
      if (buf && buf.size >= size) {
        buf.allocated = true;
        buf.refCount++;
        this.freeList = this.freeList.filter((id) => id !== bufferId);
        return bufferId;
      }
    }

    // No suitable free buffer; create a new one (if under limit)
    if (this.buffers.size < this.config.maxSize) {
      return this.createBuffer(Math.max(size, this.config.defaultBufferSize));
    }

    // Pool is full; apply eviction policy if configured
    if (this.config.aggressiveCleanup) {
      const evicted = this.evictLRU();
      if (evicted > 0) {
        return this.allocate(size); // Retry after eviction
      }
    }

    throw new Error(
      `Buffer pool exhausted: ${this.buffers.size} buffers allocated, cannot allocate ${size} more bytes`,
    );
  }

  /**
   * Release a buffer back to the free list.
   * @param bufferId Buffer identifier from allocate()
   */
  release(bufferId: string): void {
    const buf = this.buffers.get(bufferId);
    if (!buf) {
      console.warn(`Release called on unknown buffer: ${bufferId}`);
      return;
    }
    if (!buf.allocated) {
      console.warn(`Release called on already-free buffer: ${bufferId}`);
      return;
    }
    buf.refCount = Math.max(0, buf.refCount - 1);
    if (buf.refCount === 0) {
      buf.allocated = false;
      buf.releasedAtMs = performance.now();
      this.freeList.push(bufferId);
    }
  }

  /**
   * Evict the least-recently-used buffer (FIFO within free list).
   * @returns Size of evicted buffer or 0 if none available
   */
  private evictLRU(): number {
    if (this.freeList.length === 0) {
      console.warn("No free buffers to evict");
      return 0;
    }
    const toEvict = this.freeList.shift()!;
    const buf = this.buffers.get(toEvict);
    if (buf) {
      const size = buf.size;
      this.buffers.delete(toEvict);
      return size;
    }
    return 0;
  }

  /**
   * Get current pool statistics.
   */
  getStats(): PoolStats {
    let allocatedCount = 0;
    let allocatedBytes = 0;
    let freeBytes = 0;
    for (const buf of this.buffers.values()) {
      if (buf.allocated) {
        allocatedCount++;
        allocatedBytes += buf.size;
      } else {
        freeBytes += buf.size;
      }
    }
    return {
      totalBuffers: this.buffers.size,
      allocatedBuffers: allocatedCount,
      freeBuffers: this.freeList.length,
      totalBytes: allocatedBytes + freeBytes,
      allocatedBytes,
      freeBytes,
    };
  }

  /**
   * Clear all buffers from the pool (for cleanup).
   */
  clear(): void {
    this.buffers.clear();
    this.freeList = [];
    this.allocationOrder = [];
  }

  /**
   * Create a new buffer and add to pool.
   * @param size Buffer size in bytes
   * @returns Buffer ID
   */
  private createBuffer(size: number): string {
    const id = `buf_${this.nextBufferId++}`;
    const buf: PooledBuffer = {
      id,
      size,
      allocated: true,
      refCount: 1,
    };
    this.buffers.set(id, buf);
    this.allocationOrder.push(id);
    return id;
  }

  /**
   * Get the actual number of buffers currently in the pool.
   */
  size(): number {
    return this.buffers.size;
  }

  /**
   * Check if a buffer ID is known to the pool.
   */
  has(bufferId: string): boolean {
    return this.buffers.has(bufferId);
  }

  /**
   * Get buffer info (for debugging).
   */
  getBufferInfo(bufferId: string): PooledBuffer | undefined {
    return this.buffers.get(bufferId);
  }
}

/**
 * Global singleton buffer pool instance.
 */
export const globalBufferPool = new BufferPool();
