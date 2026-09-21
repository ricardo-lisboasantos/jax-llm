/**
 * bench: timing + statistics helpers (offline, no model needed).
 *
 * Covered: `tokensPerSecond`, `summarizeLatencies`, `measureSync/Async`,
 * formatting helpers. Runs under both `deno task test` and `deno task bench`
 * (all test names carry the `bench:` prefix for the bench filter).
 */
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import {
  formatMs,
  formatTokensPerSecond,
  measureAsync,
  measureSync,
  summarizeLatencies,
  tokensPerSecond,
} from "./measure.ts";

Deno.test("bench: tokensPerSecond basic math", () => {
  assertEquals(tokensPerSecond(100, 1000), 100);
  assertEquals(tokensPerSecond(50, 500), 100);
  assertEquals(tokensPerSecond(0, 1000), 0);
  assertEquals(tokensPerSecond(100, 0), 0);
  assertEquals(tokensPerSecond(100, -5), 0);
});

Deno.test("bench: summarizeLatencies empty + single", () => {
  assertEquals(summarizeLatencies([]).count, 0);
  const single = summarizeLatencies([12.5]);
  assertEquals(single.count, 1);
  assertEquals(single.meanMs, 12.5);
  assertEquals(single.p50Ms, 12.5);
  assertEquals(single.p90Ms, 12.5);
});

Deno.test("bench: summarizeLatencies distribution", () => {
  const s = summarizeLatencies([10, 20, 30, 40, 50]);
  assertEquals(s.count, 5);
  assertEquals(s.meanMs, 30);
  assertEquals(s.minMs, 10);
  assertEquals(s.maxMs, 50);
  assertEquals(s.p50Ms, 30);
  // p90 interpolates between 40 and 50.
  assert(s.p90Ms > 40 && s.p90Ms <= 50, `p90 ${s.p90Ms} out of range`);
});

Deno.test("bench: measureSync uses injected clock", () => {
  let t = 100;
  const clock = () => (t += 25);
  const { result, elapsedMs } = measureSync(() => "ok", clock);
  assertEquals(result, "ok");
  assertEquals(elapsedMs, 25);
});

Deno.test("bench: measureAsync uses injected clock", async () => {
  let t = 0;
  const clock = () => (t += 40);
  const { result, elapsedMs } = await measureAsync(
    () => 42,
    clock,
  );
  assertEquals(result, 42);
  assertEquals(elapsedMs, 40);
});

Deno.test("bench: format helpers", () => {
  assertStringIncludes(formatMs(12.345), "ms");
  assertStringIncludes(formatTokensPerSecond(123.456), "tok/s");
  assertEquals(formatTokensPerSecond(Infinity), "n/a");
});
