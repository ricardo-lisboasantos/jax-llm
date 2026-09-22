/**
 * Reproducer for Issue #2: Buffer Pool Use-After-Free
 *
 * Demonstrates that buffer pool IDs are reused while JAX arrays still
 * reference old GPU buffers, causing silent data corruption.
 */

import { BufferPool } from "../engine/llm/profiling/buffer_pool.ts";

Deno.test("Buffer Pool UAF: IDs reused while buffers still live", () => {
  const pool = new BufferPool({
    initialSize: 10,
    maxSize: 20,
    defaultBufferSize: 8192,
  });
  pool.prewarmPool();

  // Simulate attention step that allocates then releases buffers
  const qBufId1 = pool.allocate(8192);
  const kBufId1 = pool.allocate(2048);
  const vBufId1 = pool.allocate(2048);

  console.log(`Iteration 1: Q=${qBufId1}, K=${kBufId1}, V=${vBufId1}`);

  // In real code, these IDs would be passed to JAX, but then ignored!
  // JAX creates its own arrays with GPU buffer references

  // Release buffers back to pool (but JAX arrays still live!)
  pool.release(qBufId1);
  pool.release(kBufId1);
  pool.release(vBufId1);

  // Allocate again (may reuse same IDs)
  const qBufId2 = pool.allocate(8192);
  const kBufId2 = pool.allocate(2048);
  const vBufId2 = pool.allocate(2048);

  console.log(`Iteration 2: Q=${qBufId2}, K=${kBufId2}, V=${vBufId2}`);

  // BUG: IDs may be identical!
  // qBufId1 === qBufId2 (same string, e.g., "buf_0")
  // But JAX array from iteration 1 still holds reference to old GPU memory
  // Now iteration 2 overwrites that memory → CORRUPTION

  if (qBufId1 === qBufId2) {
    console.warn(
      `⚠️  CRITICAL: Buffer IDs reused without proper GPU memory management!`,
    );
    console.warn(`   qBufId1=${qBufId1}, qBufId2=${qBufId2} (SAME!)`);
    console.warn(
      `   JAX array from iteration 1 still lives but pool reused its ID`,
    );
    console.warn(`   Result: Silent data corruption in decode step`);
  }

  const stats = pool.getStats();
  console.log(
    `\nPool stats: ${stats.allocatedBuffers} allocated, ${stats.freeBuffers} free`,
  );
});

Deno.test("Buffer Pool UAF: Fix by removing unused allocations", () => {
  // The minimal fix is to NOT use buffer pool for JAX arrays at all
  // since JAX manages its own GPU memory.

  // In lfm_attention.ts, remove these lines:
  // - const qBufferId = globalBufferPool.allocate(...);
  // - const kBufferId = globalBufferPool.allocate(...);
  // - const vBufferId = globalBufferPool.allocate(...);
  // - globalBufferPool.release(qBufferId);
  // - globalBufferPool.release(kBufferId);
  // - globalBufferPool.release(vBufferId);

  console.log(
    "✅ Fix: Remove 6 lines of unused buffer pool allocation/release",
  );
  console.log(
    "   Buffer pool was intended for WebGPU memory, but JAX handles its own.",
  );
  console.log("   Unused allocations only create use-after-free risk.");
});
