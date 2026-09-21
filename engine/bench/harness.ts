/**
 * @module bench/harness
 *
 * Measurement harness for the three headline metrics:
 *  - `benchmarkEncode` — tokens/sec while tokenizing the input context.
 *  - `benchmarkPrefillFn` / `benchmarkDecodeSteps` — session-level prefill
 *    throughput and autoregressive decode throughput from `prefill()`/`step()`
 *    callbacks (closure-based so both real `np.Array` sessions and lightweight
 *    fakes can be measured without importing JAX types here).
 *  - `benchmarkGeneration` — end-to-end TTFT + prefill/decode tok/s from any
 *    `chatStream()`-shaped async generator (e.g. `ChatEngine.chatStream`).
 */

import type { ChatMessage } from "../chat/types.ts";
import type {
  DecodeMetrics,
  EncodeMetrics,
  GenerationMetrics,
  PrefillMetrics,
} from "./types.ts";
import {
  type Clock,
  now,
  summarizeLatencies,
  tokensPerSecond,
} from "./measure.ts";

/** Minimal encode surface needed for tokenizer benchmarks. */
export type EncodeTokenizer = {
  encode(text: string): number[];
  decode(tokens: number[]): string;
};

/** Stream factory shaped like `ChatEngine.chatStream` (cumulative yields). */
export type StreamFactory = () => AsyncGenerator<string, void, unknown>;

/** Options shared by the harness functions. */
export type BenchmarkOptions = {
  /** Warmup runs excluded from timing (default `1` for session fns, `0` elsewhere). */
  warmup?: number;
  /** Timed iterations (default `5` for sync fns, `3` for session fns). */
  iterations?: number;
  /** Clock source (inject a fake in tests; defaults to `performance.now`). */
  clock?: Clock;
};

/**
 * Pure math for end-to-end generation metrics (deterministic, clock-free).
 *
 * @param promptTokens Input-context token count.
 * @param generatedTokens Generated token count (one per stream yield).
 * @param ttftMs Time to first token in milliseconds.
 * @param totalMs Total wall time in milliseconds.
 * @returns Full {@linkcode GenerationMetrics}.
 */
export function computeGenerationMetrics(
  promptTokens: number,
  generatedTokens: number,
  ttftMs: number,
  totalMs: number,
): GenerationMetrics {
  const decodeWindowMs = Math.max(0, totalMs - ttftMs);
  // Single-token runs have no steady-state window — report the total rate.
  const decodeTokensPerSecond = generatedTokens <= 1
    ? tokensPerSecond(generatedTokens, totalMs)
    : tokensPerSecond(generatedTokens - 1, decodeWindowMs);
  return {
    promptTokens,
    generatedTokens,
    ttftMs,
    totalMs,
    prefillTokensPerSecond: tokensPerSecond(promptTokens, ttftMs),
    decodeTokensPerSecond,
    totalTokensPerSecond: tokensPerSecond(generatedTokens, totalMs),
  };
}

/**
 * Benchmark raw-text tokenization throughput (`encode()` only).
 *
 * @param tokenizer Tokenizer under test.
 * @param text Prompt text to encode repeatedly.
 * @param options Warmup/iteration/clock overrides.
 * @returns Encode throughput metrics.
 */
export function benchmarkEncode(
  tokenizer: EncodeTokenizer,
  text: string,
  options: BenchmarkOptions = {},
): EncodeMetrics {
  const { warmup = 0, iterations = 5, clock = now } = options;
  for (let i = 0; i < warmup; i++) tokenizer.encode(text);
  const runs = Math.max(1, iterations);
  const start = clock();
  let ids: number[] = [];
  for (let i = 0; i < runs; i++) ids = tokenizer.encode(text);
  const elapsedMs = Math.max(0, clock() - start);
  const promptTokens = ids.length;
  const totalTokens = promptTokens * runs;
  return {
    promptChars: text.length,
    promptTokens,
    elapsedMs,
    tokensPerSecond: tokensPerSecond(totalTokens, elapsedMs),
  };
}

/**
 * Benchmark tokenizer decode throughput (`decode()` only).
 *
 * @param tokenizer Tokenizer under test.
 * @param tokens Token IDs to decode repeatedly.
 * @param options Warmup/iteration/clock overrides.
 * @returns Elapsed time plus tokens/sec over `tokens.length * iterations`.
 */
export function benchmarkDecodeText(
  tokenizer: EncodeTokenizer,
  tokens: number[],
  options: BenchmarkOptions = {},
): { tokens: number; elapsedMs: number; tokensPerSecond: number } {
  const { warmup = 0, iterations = 5, clock = now } = options;
  for (let i = 0; i < warmup; i++) tokenizer.decode(tokens);
  const runs = Math.max(1, iterations);
  const start = clock();
  for (let i = 0; i < runs; i++) tokenizer.decode(tokens);
  const elapsedMs = Math.max(0, clock() - start);
  return {
    tokens: tokens.length,
    elapsedMs,
    tokensPerSecond: tokensPerSecond(tokens.length * runs, elapsedMs),
  };
}

/**
 * Benchmark prefill throughput via a caller-supplied callback.
 *
 * Pass `(ids) => session.prefill(np.array(ids, { dtype: np.uint32 }))` for a
 * real session, or a fake closure in unit tests.
 *
 * @param prefill Callback that runs one full-context prefill.
 * @param promptTokens Token count fed per prefill (for tok/s math).
 * @param options Warmup/iteration/clock overrides.
 * @returns Prefill throughput metrics.
 */
export async function benchmarkPrefillFn(
  prefill: () => void | Promise<unknown>,
  promptTokens: number,
  options: BenchmarkOptions = {},
): Promise<PrefillMetrics> {
  const { warmup = 1, iterations = 3, clock = now } = options;
  for (let i = 0; i < warmup; i++) await prefill();
  const runs = Math.max(1, iterations);
  const samples: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = clock();
    await prefill();
    samples.push(Math.max(0, clock() - start));
  }
  const total = samples.reduce((a, b) => a + b, 0);
  const meanMs = total / runs;
  return {
    promptTokens,
    elapsedMs: meanMs,
    latency: summarizeLatencies(samples),
    tokensPerSecond: tokensPerSecond(promptTokens, meanMs),
  };
}

/**
 * Benchmark autoregressive decode throughput via a caller-supplied step.
 *
 * Pass `(t) => session.step(t)` for a real session, or a fake closure in
 * unit tests. Per-token latencies feed the `p50`/`p90` distribution.
 *
 * @param step Callback that runs one decode step for the given token.
 * @param steps Token IDs to feed (length = generated tokens measured).
 * @param options Warmup/iteration/clock overrides (`iterations` repeats the whole sweep).
 * @returns Decode throughput metrics.
 */
export async function benchmarkDecodeSteps(
  step: (token: number) => void | Promise<unknown>,
  steps: number[],
  options: BenchmarkOptions = {},
): Promise<DecodeMetrics> {
  const { warmup = 0, iterations = 1, clock = now } = options;
  const token = steps[0] ?? 0;
  for (let i = 0; i < warmup; i++) await step(token);
  const runs = Math.max(1, iterations);
  const perToken: number[] = [];
  const start = clock();
  for (let r = 0; r < runs; r++) {
    for (const t of steps) {
      const s = clock();
      await step(t);
      perToken.push(Math.max(0, clock() - s));
    }
  }
  const elapsedMs = Math.max(0, clock() - start);
  const generatedTokens = steps.length * runs;
  return {
    generatedTokens,
    elapsedMs,
    latency: summarizeLatencies(perToken),
    tokensPerSecond: tokensPerSecond(generatedTokens, elapsedMs),
  };
}

/**
 * Time until the first chunk of a `chatStream()`-shaped generator.
 *
 * @param streamFactory Factory returning a fresh async generator per call.
 * @param clock Clock source (inject a fake in tests).
 * @returns TTFT plus the first chunk (empty string when the stream is empty).
 */
export async function measureTimeToFirstToken(
  streamFactory: StreamFactory,
  clock: Clock = now,
): Promise<{ ttftMs: number; firstChunk: string }> {
  const start = clock();
  for await (const chunk of streamFactory()) {
    return { ttftMs: Math.max(0, clock() - start), firstChunk: chunk };
  }
  return { ttftMs: Math.max(0, clock() - start), firstChunk: "" };
}

/**
 * End-to-end generation benchmark: TTFT + prefill/decode tok/s.
 *
 * Each yield counts as one generated token (matches `ChatEngine.chatStream`,
 * which yields once per token). The stream is fully consumed so `totalMs`
 * covers the whole reply.
 *
 * @param streamFactory Factory returning a fresh async generator per call.
 * @param promptTokens Input-context token count (from `encodePrompt`).
 * @param clock Clock source (inject a fake in tests).
 * @returns Full {@linkcode GenerationMetrics}.
 */
export async function benchmarkGeneration(
  streamFactory: StreamFactory,
  promptTokens: number,
  clock: Clock = now,
): Promise<GenerationMetrics> {
  const start = clock();
  let ttftMs = 0;
  let generatedTokens = 0;
  let seenFirst = false;
  for await (const _chunk of streamFactory()) {
    void _chunk;
    generatedTokens++;
    if (!seenFirst) {
      seenFirst = true;
      ttftMs = Math.max(0, clock() - start);
    }
  }
  const totalMs = Math.max(0, clock() - start);
  // Empty stream (immediate stop): TTFT equals total time, no tokens.
  if (!seenFirst) ttftMs = totalMs;
  return computeGenerationMetrics(
    promptTokens,
    generatedTokens,
    ttftMs,
    totalMs,
  );
}

/**
 * Minimal engine surface needed to benchmark a live `ChatEngine` without
 * importing it (avoids a chat→bench dependency cycle in type space).
 */
export type BenchmarkableEngine = {
  chatStream(history: ChatMessage[]): AsyncGenerator<string, void, unknown>;
  getRuntime(): {
    definition: {
      encodePrompt(
        tokenizer: { encode(text: string): number[] },
        history: ChatMessage[],
      ): number[];
    };
    getTokenizer(): { encode(text: string): number[] };
  };
};

/**
 * Benchmark a live engine end-to-end (encode count + streamed generation).
 *
 * @param engine Loaded `ChatEngine` (or any matching structural stub in tests).
 * @param history Conversation history to send.
 * @param clock Clock source (inject a fake in tests).
 * @returns Full {@linkcode GenerationMetrics}.
 */
export async function benchmarkChatEngine(
  engine: BenchmarkableEngine,
  history: ChatMessage[],
  clock: Clock = now,
): Promise<GenerationMetrics> {
  const runtime = engine.getRuntime();
  const promptTokens = runtime.definition.encodePrompt(
    runtime.getTokenizer(),
    history,
  ).length;
  return await benchmarkGeneration(
    () => engine.chatStream(history),
    promptTokens,
    clock,
  );
}
