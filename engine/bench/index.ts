/**
 * @module bench
 *
 * Benchmark stack for LLM inference metrics:
 *  - Time to first token (TTFT)
 *  - Input-context tokenization / prefill throughput (tok/s)
 *  - Autoregressive generation throughput (tok/s)
 *
 * The harness is backend-agnostic: unit benchmarks run fully offline against
 * mocked tokenizers/sessions, while `tests/bench_test.ts` wires the same
 * helpers to a live `ChatEngine` (skipped gracefully without network/GPU).
 */

export * from "./types.ts";
export * from "./measure.ts";
export * from "./harness.ts";
export * from "./fixtures.ts";
