/**
 * @module bench/types
 *
 * Shared metric types for the benchmark stack.
 *
 * Three headline metrics:
 *  - **Time to first token (TTFT)**: wall time from request start until the
 *    first generated token is emitted (covers prompt encoding + prefill +
 *    first sample).
 *  - **Prefill / encode throughput**: input-context tokens per second while
 *    tokenizing + prefilling the prompt.
 *  - **Decode throughput**: generated tokens per second during the
 *    autoregressive `step()` loop.
 */

/** Latency distribution over repeated samples (all values in milliseconds). */
export type LatencyStats = {
  /** Number of samples. */
  count: number;
  /** Arithmetic mean. */
  meanMs: number;
  /** Minimum observed. */
  minMs: number;
  /** Maximum observed. */
  maxMs: number;
  /** Median (p50). */
  p50Ms: number;
  /** 90th percentile. */
  p90Ms: number;
};

/** Throughput for tokenizing raw text into token IDs. */
export type EncodeMetrics = {
  /** Full prompt text that was encoded. */
  promptChars: number;
  /** Token count produced by `encode()`. */
  promptTokens: number;
  /** Wall time spent in `encode()` calls. */
  elapsedMs: number;
  /** `promptTokens / (elapsedMs / 1000)`. */
  tokensPerSecond: number;
};

/** Throughput for the prefill pass (full context consumed at once). */
export type PrefillMetrics = {
  /** Input-context token count fed to `prefill()`. */
  promptTokens: number;
  /** Wall time spent in `prefill()` (averaged when `iterations > 1`). */
  elapsedMs: number;
  /** Per-iteration latency distribution (single entry when `iterations = 1`). */
  latency: LatencyStats;
  /** `promptTokens / (elapsedMs / 1000)`. */
  tokensPerSecond: number;
};

/** Throughput for the autoregressive decode loop (`step()` per token). */
export type DecodeMetrics = {
  /** Number of generated tokens measured. */
  generatedTokens: number;
  /** Total wall time for all `step()` calls. */
  elapsedMs: number;
  /** Per-token latency distribution. */
  latency: LatencyStats;
  /** `generatedTokens / (elapsedMs / 1000)`. */
  tokensPerSecond: number;
};

/**
 * End-to-end generation metrics for one `chatStream()` run.
 *
 * `ttftMs` covers everything up to the first emitted token (prompt
 * formatting, `encode()`, `prefill()`, first `logits.data()` + sample).
 * `decodeTokensPerSecond` excludes that first-token latency so it reflects
 * steady-state generation speed.
 */
export type GenerationMetrics = {
  /** Input-context token count (incl. BOS + prompt template). */
  promptTokens: number;
  /** Number of generated tokens (one per `chatStream()` yield). */
  generatedTokens: number;
  /** Time to first token in milliseconds. */
  ttftMs: number;
  /** Total wall time from request start to last token. */
  totalMs: number;
  /** Input tokens per TTFT second (`promptTokens / (ttftMs / 1000)`). */
  prefillTokensPerSecond: number;
  /** Generated tokens per decode second (excludes TTFT window). */
  decodeTokensPerSecond: number;
  /** Generated tokens per total second (`generatedTokens / total`). */
  totalTokensPerSecond: number;
};

/** One row of the multi-context benchmark matrix (see `fixtures.ts`). */
export type ContextSweepRow = {
  /** Label for the row (e.g. `"short"`, `"medium"`, `"long"`). */
  label: string;
  /** Prompt text used. */
  promptChars: number;
  /** Full generation metrics for this context size. */
  generation: GenerationMetrics;
  /** Tokenizer-only encode metrics for this context size. */
  encode: EncodeMetrics;
};
