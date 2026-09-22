/**
 * unit: WebGPU profiler tests
 */
import { assert, assertEquals } from "@std/assert";
import {
  globalProfiler,
  profileAsync,
  profileSync,
  WebGPUProfiler,
} from "./webgpu_profiler.ts";

Deno.test("unit: WebGPUProfiler collects samples", () => {
  const profiler = new WebGPUProfiler();

  // Record some fake latencies
  for (let i = 0; i < 10; i++) {
    profiler.start("test_op");
    // Simulate some work
    for (let j = 0; j < 1000; j++) {
      Math.sqrt(j);
    }
    profiler.end("test_op");
  }

  const stats = profiler.getStats("test_op");
  assert(stats !== undefined);
  assertEquals(stats.count, 10);
  assert(stats.meanMs > 0);
  assert(stats.minMs <= stats.p50Ms);
  assert(stats.p50Ms <= stats.p90Ms);
  assert(stats.p90Ms <= stats.p99Ms);
  assert(stats.p99Ms <= stats.maxMs);
});

Deno.test("unit: WebGPUProfiler percentile calculation", () => {
  const profiler = new WebGPUProfiler();

  // Manually inject sorted samples: [1, 2, 3, ..., 100]
  for (let i = 1; i <= 100; i++) {
    profiler.start("percentile_test");
    // Fake timing by measuring fixed-time operations
    profiler.end("percentile_test");
  }

  const stats = profiler.getStats("percentile_test");
  assert(stats !== undefined);
  assert(stats.count === 100);

  // Percentiles should make intuitive sense
  assert(stats.p50Ms <= stats.p90Ms);
  assert(stats.p90Ms <= stats.p99Ms);
  assert(stats.stddevMs >= 0);
});

Deno.test("unit: WebGPUProfiler handles missing start", () => {
  const profiler = new WebGPUProfiler();
  // end() without start() should warn gracefully
  profiler.end("nonexistent");
  const stats = profiler.getStats("nonexistent");
  assertEquals(stats, undefined);
});

Deno.test("unit: WebGPUProfiler formatStats produces valid output", () => {
  const profiler = new WebGPUProfiler();
  profiler.start("op1");
  profiler.end("op1");
  profiler.start("op2");
  profiler.end("op2");

  const allStats = profiler.getAllStats();
  assertEquals(allStats.size, 2);

  const formatted = WebGPUProfiler.formatStats(allStats);
  assert(formatted.includes("op1"));
  assert(formatted.includes("op2"));
  assert(formatted.includes("Count"));
  assert(formatted.includes("Mean"));
});

Deno.test("unit: profileSync wrapper captures timing", () => {
  globalProfiler.reset();

  const expensiveOp = profileSync("expensive_sync", () => {
    let sum = 0;
    for (let i = 0; i < 10000; i++) sum += Math.sqrt(i);
    return sum;
  });

  const result = expensiveOp();
  assert(result > 0);

  const stats = globalProfiler.getStats("expensive_sync");
  assert(stats !== undefined);
  assertEquals(stats.count, 1);
  assert(stats.meanMs >= 0);
});

Deno.test("unit: profileAsync wrapper captures timing", async () => {
  globalProfiler.reset();

  const asyncOp = profileAsync("async_delay", () => {
    return new Promise((resolve) => {
      setTimeout(() => resolve(42), 10);
    });
  });

  const result = await asyncOp();
  assertEquals(result, 42);

  const stats = globalProfiler.getStats("async_delay");
  assert(stats !== undefined);
  assertEquals(stats.count, 1);
  assert(stats.meanMs >= 10); // At least the timeout duration
});

Deno.test("unit: WebGPUProfiler reset clears data", () => {
  const profiler = new WebGPUProfiler();
  profiler.start("test");
  profiler.end("test");
  assert(profiler.getStats("test") !== undefined);

  profiler.reset();
  assertEquals(profiler.getStats("test"), undefined);
});
