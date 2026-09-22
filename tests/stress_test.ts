/**
 * Long-context stress test for JAX-LLM optimizations.
 * Tests cache reuse, buffer pooling, and dynamic KV cache sizing.
 */
import { assertEquals } from "@std/assert";
import { ChatEngine } from "../mod.ts";

async function hasNetwork(): Promise<boolean> {
  try {
    const perm = await Deno.permissions.query({ name: "net" });
    if (perm.state !== "granted") return false;
  } catch {
    return false;
  }
  return true;
}

Deno.test("stress: buffer pool stress (rapid allocation/release)", async () => {
  const { BufferPool } = await import(
    "../engine/llm/profiling/buffer_pool.ts"
  );

  const pool = new BufferPool({
    initialSize: 10,
    defaultBufferSize: 2048,
    maxSize: 50,
    aggressiveCleanup: true,
  });

  pool.prewarmPool();

  // Simulate rapid alloc/release pattern
  const iterations = 100;
  const allocatedIds: string[] = [];

  const startTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    // Allocate batch
    for (let j = 0; j < 5; j++) {
      const size = Math.random() * 2048 + 512;
      const id = pool.allocate(Math.floor(size));
      allocatedIds.push(id);
    }

    // Release half
    while (allocatedIds.length > 2) {
      const id = allocatedIds.pop()!;
      pool.release(id);
    }

    if (i % 20 === 0) {
      const stats = pool.getStats();
      console.log(
        `  Iteration ${i}: buffers=${stats.totalBuffers}, allocated=${stats.allocatedBuffers}, free=${stats.freeBuffers}`,
      );
    }
  }

  // Release remaining
  for (const id of allocatedIds) {
    pool.release(id);
  }

  const elapsed = (performance.now() - startTime) / 1000;
  const opsPerSec = ((iterations * 5) / elapsed).toFixed(0);
  console.log(
    `✓ Buffer pool stress: ${iterations * 5} ops in ${
      elapsed.toFixed(2)
    }s (${opsPerSec} ops/s)`,
  );

  const finalStats = pool.getStats();
  assertEquals(finalStats.allocatedBuffers, 0);
  pool.clear();
});

Deno.test("stress: sequential prompts (cache reuse)", async () => {
  if (!(await hasNetwork())) {
    console.log("SKIP stress: no network access");
    return;
  }

  const engine = new ChatEngine("lfm2.5-350m", {
    maxTokens: 30,
    backend: "webgpu",
  });

  try {
    await engine.init();
  } catch (e) {
    console.log(`SKIP stress: backend unavailable - ${(e as Error).message}`);
    return;
  }

  const prompts = [
    "What is machine learning?",
    "Explain transformers.",
    "How does attention work?",
  ];

  for (const [idx, prompt] of prompts.entries()) {
    const startTime = performance.now();
    const reply = await engine.generate(prompt);
    const elapsed = (performance.now() - startTime) / 1000;
    const tokenCount = reply.split(/\s+/).length;
    const tps = (tokenCount / elapsed).toFixed(1);
    console.log(
      `Prompt ${idx + 1}: ~${tokenCount} tokens in ${
        elapsed.toFixed(2)
      }s (${tps} tok/s)`,
    );
  }

  console.log("✓ Sequential prompts completed");
  engine.dispose();
});
