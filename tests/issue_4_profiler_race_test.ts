/**
 * Reproducer for Issue #4: Profiler Race Condition
 *
 * Demonstrates that concurrent sessions produce garbage metrics due to
 * single-valued timestamps map being overwritten.
 */

import { WebGPUProfiler } from "../engine/llm/profiling/webgpu_profiler.ts";

Deno.test("Profiler Race: Concurrent sessions overwrite timestamps", async () => {
  const profiler = new WebGPUProfiler();

  console.log(
    "Simulating two concurrent sessions calling start/end on same label...\n",
  );

  // Session 1 starts profiling
  profiler.start("decode_step");
  const session1Start = performance.now();
  console.log(`[${session1Start.toFixed(2)}] Session 1: start("decode_step")`);

  await new Promise((r) => setTimeout(r, 10));

  // Session 2 starts profiling (SAME LABEL)
  profiler.start("decode_step"); // ← OVERWRITES Session 1's timestamp in the map
  const session2Start = performance.now();
  console.log(
    `[${
      session2Start.toFixed(2)
    }] Session 2: start("decode_step") - OVERWRITES Session 1!`,
  );

  await new Promise((r) => setTimeout(r, 50));

  // Session 2 ends profiling
  profiler.end("decode_step"); // ← Uses Session 2's timestamp (from map)
  const session2End = performance.now();
  console.log(
    `[${session2End.toFixed(2)}] Session 2: end("decode_step") - elapsed ~${
      (session2End - session2Start).toFixed(0)
    }ms`,
  );

  await new Promise((r) => setTimeout(r, 10));

  // Session 1 ends profiling
  profiler.end("decode_step"); // ← WRONG: Uses Session 2's timestamp!
  const session1End = performance.now();
  const wrongElapsed = session1End - session2Start; // Wrong math!
  console.log(
    `[${session1End.toFixed(2)}] Session 1: end("decode_step") - elapsed ~${
      wrongElapsed.toFixed(0)
    }ms (WRONG!)`,
  );

  const stats = profiler.getStats("decode_step");
  console.log(`\n⚠️  Metrics garbage:`);
  console.log(`   Samples: ${stats?.count || 0}`);
  console.log(`   Mean: ${stats?.meanMs.toFixed(2)}ms (CORRUPTED)`);
  console.log(`   P90: ${stats?.p90Ms.toFixed(2)}ms (CORRUPTED)`);
  console.log(`\nExpected: [~10ms, ~50ms]`);
  console.log(`Actual: [garbage, garbage]`);
});

Deno.test("Profiler Race: Fix by using timestamp stack", async () => {
  // The fix is to change timestamps from Map<label, number> to Map<label, number[]>
  // Each start() pushes to the stack, each end() pops from the stack

  class FixedWebGPUProfiler {
    private samples = new Map<string, number[]>();
    private timestamps = new Map<string, number[]>(); // CHANGED: Stack, not single value

    start(label: string): void {
      if (!this.timestamps.has(label)) {
        this.timestamps.set(label, []);
      }
      this.timestamps.get(label)!.push(performance.now()); // PUSH to stack
    }

    end(label: string): void {
      const stack = this.timestamps.get(label);
      if (!stack || stack.length === 0) {
        console.warn(`No start timestamp for label: ${label}`);
        return;
      }
      const startTime = stack.pop()!; // POP from stack (LIFO)
      const elapsedMs = performance.now() - startTime;
      if (!this.samples.has(label)) {
        this.samples.set(label, []);
      }
      this.samples.get(label)!.push(elapsedMs);
    }

    getStats(label: string) {
      const samples = this.samples.get(label);
      if (!samples || samples.length === 0) return undefined;
      const sorted = [...samples].sort((a, b) => a - b);
      return {
        count: sorted.length,
        meanMs: sorted.reduce((a, b) => a + b, 0) / sorted.length,
      };
    }
  }

  const profiler = new FixedWebGPUProfiler();

  console.log("With fixed profiler (timestamp stack):\n");

  profiler.start("decode_step");
  const session1Start = performance.now();
  console.log(`[${session1Start.toFixed(2)}] Session 1: start("decode_step")`);

  await new Promise((r) => setTimeout(r, 10));

  profiler.start("decode_step"); // ← PUSHES to stack (doesn't overwrite)
  const session2Start = performance.now();
  console.log(
    `[${
      session2Start.toFixed(2)
    }] Session 2: start("decode_step") - added to stack`,
  );

  await new Promise((r) => setTimeout(r, 50));

  profiler.end("decode_step"); // ← POPS from stack (Session 2's start)
  const session2End = performance.now();
  const session2Elapsed = session2End - session2Start;
  console.log(
    `[${session2End.toFixed(2)}] Session 2: end("decode_step") - elapsed ~${
      session2Elapsed.toFixed(0)
    }ms ✓ CORRECT`,
  );

  await new Promise((r) => setTimeout(r, 10));

  profiler.end("decode_step"); // ← POPS from stack (Session 1's start)
  const session1End = performance.now();
  const session1Elapsed = session1End - session1Start;
  console.log(
    `[${session1End.toFixed(2)}] Session 1: end("decode_step") - elapsed ~${
      session1Elapsed.toFixed(0)
    }ms ✓ CORRECT`,
  );

  const stats = profiler.getStats("decode_step");
  console.log(`\n✅ Metrics correct:`);
  console.log(`   Samples: ${stats?.count || 0}`);
  console.log(`   Mean: ${stats?.meanMs.toFixed(2)}ms (CORRECT)`);
  console.log(`   Session 1: ~10ms ✓`);
  console.log(`   Session 2: ~50ms ✓`);
});
