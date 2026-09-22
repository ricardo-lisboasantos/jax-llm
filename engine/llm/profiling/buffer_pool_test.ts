/**
 * unit: WebGPU buffer pool tests
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import { BufferPool } from "./buffer_pool.ts";

Deno.test("unit: BufferPool creates initial buffers on prewarm", () => {
  const pool = new BufferPool({ initialSize: 5, defaultBufferSize: 1024 });
  pool.prewarmPool();

  const stats = pool.getStats();
  assertEquals(stats.totalBuffers, 5);
  assertEquals(stats.freeBuffers, 5);
  assertEquals(stats.allocatedBuffers, 0);
  assertEquals(stats.totalBytes, 5 * 1024);
});

Deno.test("unit: BufferPool allocate and release", () => {
  const pool = new BufferPool({ initialSize: 3, defaultBufferSize: 1024 });
  pool.prewarmPool();

  // Allocate one buffer
  const buf1 = pool.allocate(512);
  assert(pool.has(buf1));

  let stats = pool.getStats();
  assertEquals(stats.allocatedBuffers, 1);
  assertEquals(stats.freeBuffers, 2);

  // Release it
  pool.release(buf1);
  stats = pool.getStats();
  assertEquals(stats.allocatedBuffers, 0);
  assertEquals(stats.freeBuffers, 3);
});

Deno.test("unit: BufferPool reuses free buffers", () => {
  const pool = new BufferPool({ initialSize: 2, defaultBufferSize: 1024 });
  pool.prewarmPool();

  // Allocate first buffer
  const buf1 = pool.allocate(1024);
  const stats1 = pool.getStats();
  const poolSize1 = stats1.totalBuffers;

  // Release it
  pool.release(buf1);

  // Allocate again (should reuse the same buffer)
  const _buf2 = pool.allocate(512);
  const stats2 = pool.getStats();
  const poolSize2 = stats2.totalBuffers;

  // Pool size should not have grown
  assertEquals(poolSize1, poolSize2);
});

Deno.test("unit: BufferPool creates new buffer if needed", () => {
  const pool = new BufferPool({
    initialSize: 1,
    maxSize: 5,
    defaultBufferSize: 1024,
  });
  pool.prewarmPool();

  // Allocate multiple buffers (should create new ones as needed)
  const buf1 = pool.allocate(500);
  const buf2 = pool.allocate(500);
  const buf3 = pool.allocate(500);

  const stats = pool.getStats();
  assertEquals(stats.totalBuffers, 3);
  assertEquals(stats.allocatedBuffers, 3);

  pool.release(buf1);
  pool.release(buf2);
  pool.release(buf3);
});

Deno.test("unit: BufferPool throws when exceeding maxSize", () => {
  const pool = new BufferPool({
    initialSize: 1,
    maxSize: 2,
    defaultBufferSize: 1024,
  });
  pool.prewarmPool();

  // Allocate initial buffer + 1 more = pool is full
  const buf1 = pool.allocate(500);
  const buf2 = pool.allocate(500);

  // Third allocation should fail
  assertThrows(() => {
    pool.allocate(500);
  });

  pool.release(buf1);
  pool.release(buf2);
});

Deno.test("unit: BufferPool supports reference counting", () => {
  const pool = new BufferPool({ initialSize: 2, defaultBufferSize: 1024 });
  pool.prewarmPool();

  // Allocate first buffer
  const buf = pool.allocate(512);
  const info1 = pool.getBufferInfo(buf);
  assertEquals(info1?.refCount, 1);

  // Allocate a second buffer (pool has 2)
  const buf2 = pool.allocate(512);
  assert(buf !== buf2); // Should be different buffers

  const info2 = pool.getBufferInfo(buf2);
  assertEquals(info2?.refCount, 1);

  // Release first buffer
  pool.release(buf);
  const info3 = pool.getBufferInfo(buf);
  assertEquals(info3?.refCount, 0);
  assertEquals(info3?.allocated, false);

  // Release second buffer
  pool.release(buf2);
  const info4 = pool.getBufferInfo(buf2);
  assertEquals(info4?.refCount, 0);
  assertEquals(info4?.allocated, false);
});

Deno.test("unit: BufferPool.clear() empties pool", () => {
  const pool = new BufferPool({ initialSize: 3, defaultBufferSize: 1024 });
  pool.prewarmPool();

  assertEquals(pool.size(), 3);
  pool.clear();
  assertEquals(pool.size(), 0);

  const stats = pool.getStats();
  assertEquals(stats.totalBuffers, 0);
});

Deno.test("unit: BufferPool LRU eviction with aggressiveCleanup", () => {
  const pool = new BufferPool({
    initialSize: 2,
    maxSize: 2,
    defaultBufferSize: 1024,
    aggressiveCleanup: true,
  });
  pool.prewarmPool();

  // Fill the pool
  const buf1 = pool.allocate(500);
  const buf2 = pool.allocate(500);

  // Release them in order (FIFO)
  pool.release(buf1);
  pool.release(buf2);

  // Now with aggressive cleanup enabled, a new allocation should evict the oldest
  const buf3 = pool.allocate(500);
  assert(pool.has(buf3)); // New buffer should be allocated
  assertEquals(pool.size(), 2); // Pool size should stay at maxSize
});

Deno.test("unit: BufferPool gets stats accurately", () => {
  const pool = new BufferPool({ initialSize: 2, defaultBufferSize: 2048 });
  pool.prewarmPool();

  let stats = pool.getStats();
  assertEquals(stats.totalBytes, 4096);
  assertEquals(stats.allocatedBytes, 0); // Pre-warmed buffers are free, not allocated
  assertEquals(stats.freeBytes, 4096);

  const buf1 = pool.allocate(1024);
  const _buf2 = pool.allocate(1024);
  pool.release(buf1);

  stats = pool.getStats();
  assertEquals(stats.allocatedBuffers, 1);
  assertEquals(stats.freeBuffers, 1);
  assertEquals(stats.allocatedBytes, 2048);
  assertEquals(stats.freeBytes, 2048);
});
