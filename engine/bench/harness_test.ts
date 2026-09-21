/**
 * bench: harness metrics with mocked tokenizer/session/stream (offline).
 *
 * Verifies the three headline metrics without a real model:
 *  - encode throughput via `benchmarkEncode` / `benchmarkDecodeText`
 *  - prefill/decode throughput via `benchmarkPrefillFn` / `benchmarkDecodeSteps`
 *  - TTFT + end-to-end generation via `measureTimeToFirstToken`,
 *    `benchmarkGeneration`, `benchmarkChatEngine`, `computeGenerationMetrics`
 */
import { assert, assertEquals } from "@std/assert";
import {
  benchmarkChatEngine,
  benchmarkDecodeSteps,
  benchmarkDecodeText,
  benchmarkEncode,
  benchmarkGeneration,
  benchmarkPrefillFn,
  computeGenerationMetrics,
  measureTimeToFirstToken,
} from "./harness.ts";
import { tokensPerSecond } from "./measure.ts";
import {
  BENCH_PROMPTS,
  buildBenchPrompt,
  formatSweepRow,
  syntheticText,
} from "./fixtures.ts";

/** Whitespace tokenizer stub: one token per word. */
function fakeTokenizer() {
  return {
    encode: (text: string): number[] =>
      text.split(/\s+/).filter(Boolean).map((_, i) => i),
    decode: (tokens: number[]): string => tokens.map(String).join(" "),
  };
}

Deno.test("bench: benchmarkEncode counts tokens", () => {
  const m = benchmarkEncode(fakeTokenizer(), "hello world foo", {
    iterations: 4,
    clock: (() => {
      let t = 0;
      return () => (t += 10);
    })(),
  });
  assertEquals(m.promptTokens, 3);
  assertEquals(m.promptChars, "hello world foo".length);
  // 3 tokens x 4 runs over a 10ms window.
  assertEquals(m.tokensPerSecond, tokensPerSecond(12, 10));
});

Deno.test("bench: benchmarkDecodeText throughput", () => {
  const m = benchmarkDecodeText(fakeTokenizer(), [1, 2, 3], {
    iterations: 2,
    clock: (() => {
      let t = 0;
      return () => (t += 5);
    })(),
  });
  assertEquals(m.tokens, 3);
  assertEquals(m.tokensPerSecond, tokensPerSecond(6, 5));
});

Deno.test("bench: benchmarkPrefillFn honors warmup + iterations", async () => {
  let calls = 0;
  const m = await benchmarkPrefillFn(
    () => {
      calls++;
    },
    128,
    {
      warmup: 1,
      iterations: 3,
      clock: (() => {
        let t = 0;
        return () => (t += 20);
      })(),
    },
  );
  assertEquals(calls, 4);
  assertEquals(m.promptTokens, 128);
  assertEquals(m.latency.count, 3);
  assertEquals(m.tokensPerSecond, tokensPerSecond(128, 20));
});

Deno.test("bench: benchmarkDecodeSteps counts generated tokens", async () => {
  const seen: number[] = [];
  const m = await benchmarkDecodeSteps(
    (t: number) => {
      seen.push(t);
    },
    [7, 8, 9],
    {
      iterations: 2,
      clock: (() => {
        let t = 0;
        return () => (t += 2);
      })(),
    },
  );
  assertEquals(m.generatedTokens, 6);
  assertEquals(seen, [7, 8, 9, 7, 8, 9]);
  assertEquals(m.latency.count, 6);
});

Deno.test("bench: computeGenerationMetrics deterministic math", () => {
  const m = computeGenerationMetrics(100, 10, 500, 1500);
  assertEquals(m.promptTokens, 100);
  assertEquals(m.generatedTokens, 10);
  assertEquals(m.ttftMs, 500);
  assertEquals(m.totalMs, 1500);
  assertEquals(m.prefillTokensPerSecond, 200); // 100 tok / 0.5s
  assertEquals(m.decodeTokensPerSecond, 9); // 9 tok / 1.0s (excl. first)
  const total = (10 * 1000) / 1500;
  assert(Math.abs(m.totalTokensPerSecond - total) < 1e-9);
});

Deno.test("bench: computeGenerationMetrics single-token run", () => {
  const m = computeGenerationMetrics(64, 1, 100, 400);
  // No steady-state window — decode falls back to the total rate.
  assertEquals(m.decodeTokensPerSecond, tokensPerSecond(1, 400));
});

Deno.test("bench: measureTimeToFirstToken first chunk + empty", async () => {
  async function* two() {
    yield "hello";
    yield "hello world";
  }
  const { ttftMs, firstChunk } = await measureTimeToFirstToken(() => two());
  assertEquals(firstChunk, "hello");
  assert(ttftMs >= 0);

  async function* empty(): AsyncGenerator<string, void, unknown> {
    // no yields
  }
  const e = await measureTimeToFirstToken(() => empty());
  assertEquals(e.firstChunk, "");
  assert(e.ttftMs >= 0);
});

Deno.test("bench: benchmarkGeneration counts yields as tokens", async () => {
  async function* three() {
    yield "a";
    yield "a b";
    yield "a b c";
  }
  const m = await benchmarkGeneration(() => three(), 50);
  assertEquals(m.generatedTokens, 3);
  assertEquals(m.promptTokens, 50);
  assert(m.ttftMs >= 0);
  assert(m.totalMs >= m.ttftMs);
});

Deno.test("bench: benchmarkGeneration TTFT captures first-token delay", async () => {
  async function* delayed() {
    await new Promise((r) => setTimeout(r, 20));
    yield "first";
    yield "first second";
  }
  const m = await benchmarkGeneration(() => delayed(), 10);
  assertEquals(m.generatedTokens, 2);
  assert(
    m.ttftMs > 0,
    `expected positive TTFT after 20ms delay, got ${m.ttftMs}`,
  );
});

Deno.test("bench: benchmarkChatEngine wires prompt count + stream", async () => {
  const stub = {
    chatStream: async function* (_h: unknown[]) {
      yield "hi";
      yield "hi there";
    },
    getRuntime: () => ({
      definition: { encodePrompt: () => [1, 2, 3, 4] },
      getTokenizer: () => ({ encode: (_t: string) => [0] }),
    }),
  };
  // deno-lint-ignore no-explicit-any
  const m = await benchmarkChatEngine(stub as any, [
    { role: "user", content: "hi" },
  ]);
  assertEquals(m.promptTokens, 4);
  assertEquals(m.generatedTokens, 2);
});

Deno.test("bench: fixtures synthetic + sweep labels", () => {
  assertEquals(syntheticText(0), "");
  const t = syntheticText(5000);
  assert(t.length >= 5000);
  assertEquals(BENCH_PROMPTS.map((p) => p.label), [
    "short",
    "medium",
    "long",
  ]);
  const custom = buildBenchPrompt("custom", 100);
  assert(custom.text.length >= 100);
  assert(custom.history.length === 1);
  const row = formatSweepRow("short", 32, 12.3, 100, 50);
  assert(row.includes("short") && row.includes("TTFT"));
});
