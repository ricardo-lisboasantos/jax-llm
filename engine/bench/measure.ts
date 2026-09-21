/**
 * @module bench/measure
 *
 * Monotonic timing + statistics helpers for the benchmark stack.
 *
 * All wall-time measurements use `performance.now()` so results are
 * comparable between unit (mocked) and integration (real model) runs.
 */

/** Clock function returning milliseconds (defaults to `performance.now`). */
export type Clock = () => number;

import type { LatencyStats } from "./types.ts";

/** Current wall time in milliseconds. */
export function now(): number {
  return performance.now();
}

/**
 * Time a synchronous function.
 *
 * @param fn Work to time.
 * @param clock Clock source (inject a fake in tests).
 * @returns The return value plus elapsed milliseconds.
 */
export function measureSync<T>(
  fn: () => T,
  clock: Clock = now,
): { result: T; elapsedMs: number } {
  const start = clock();
  const result = fn();
  return { result, elapsedMs: Math.max(0, clock() - start) };
}

/**
 * Time an async function.
 *
 * @param fn Work to time.
 * @param clock Clock source (inject a fake in tests).
 * @returns The resolved value plus elapsed milliseconds.
 */
export async function measureAsync<T>(
  fn: () => T | Promise<T>,
  clock: Clock = now,
): Promise<{ result: T; elapsedMs: number }> {
  const start = clock();
  const result = await fn();
  return { result, elapsedMs: Math.max(0, clock() - start) };
}

/**
 * Tokens per second for a token count over an elapsed window.
 *
 * Returns `0` for non-positive windows instead of `Infinity` so benchmark
 * tables stay sortable and assertions stay total.
 *
 * @param tokenCount Number of tokens processed.
 * @param elapsedMs Elapsed wall time in milliseconds.
 * @returns Tokens per second.
 */
export function tokensPerSecond(
  tokenCount: number,
  elapsedMs: number,
): number {
  if (tokenCount <= 0 || elapsedMs <= 0) return 0;
  return (tokenCount * 1000) / elapsedMs;
}

/**
 * Summarize latency samples into mean/min/max/p50/p90.
 *
 * @param samplesMs Raw per-sample latencies in milliseconds.
 * @returns Distribution summary (zeros when empty).
 */
export function summarizeLatencies(samplesMs: number[]): LatencyStats {
  if (samplesMs.length === 0) {
    return { count: 0, meanMs: 0, minMs: 0, maxMs: 0, p50Ms: 0, p90Ms: 0 };
  }
  const sorted = [...samplesMs].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const quantile = (q: number): number => {
    if (sorted.length === 1) return sorted[0];
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  return {
    count: sorted.length,
    meanMs: sum / sorted.length,
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
    p50Ms: quantile(0.5),
    p90Ms: quantile(0.9),
  };
}

/**
 * Format milliseconds for benchmark tables (`"12.3 ms"`).
 *
 * @param ms Milliseconds.
 * @returns Human-readable string.
 */
export function formatMs(ms: number): string {
  return `${ms.toFixed(1)} ms`;
}

/**
 * Format a tokens/sec value for benchmark tables (`"123.4 tok/s"`).
 *
 * @param tps Tokens per second.
 * @returns Human-readable string.
 */
export function formatTokensPerSecond(tps: number): string {
  if (!Number.isFinite(tps)) return "n/a";
  return `${tps.toFixed(1)} tok/s`;
}
