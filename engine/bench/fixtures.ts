/**
 * @module bench/fixtures
 *
 * Standard prompts for the benchmark stack.
 *
 * Sizes are heuristic (~4 chars per token for English prose): `short` fits a
 * smoke test, `medium` exercises a realistic chat turn, and `long` stresses
 * prefill with a multi-kilobyte context. `syntheticText()` builds arbitrary
 * context sizes for sweeps without network access.
 */

import type { ChatMessage } from "../chat/types.ts";

/** One named benchmark prompt. */
export type BenchPrompt = {
  /** Stable label used as the sweep row name. */
  label: string;
  /** Single-turn history sent to `chat()` / `chatStream()`. */
  history: ChatMessage[];
  /** Raw prompt text (for tokenizer-only benchmarks). */
  text: string;
};

const LOREM =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. ";

/**
 * Build deterministic filler text of at least `targetChars` characters.
 *
 * Repeats a Lorem-ipsum paragraph (no network, no randomness) so tokenizer
 * benchmarks scale predictably with context size.
 *
 * @param targetChars Minimum output length in characters.
 * @returns Filler text with `length >= targetChars`.
 */
export function syntheticText(targetChars: number): string {
  if (targetChars <= 0) return "";
  let out = "";
  while (out.length < targetChars) out += LOREM;
  return out.slice(0, Math.max(targetChars, LOREM.length));
}

/**
 * Build a single-turn user prompt backed by `targetChars` of filler text.
 *
 * @param label Row label for reports.
 * @param targetChars Minimum prompt body length in characters.
 * @returns Named benchmark prompt.
 */
export function buildBenchPrompt(
  label: string,
  targetChars: number,
): BenchPrompt {
  const text = `Benchmark context (${label}): ` + syntheticText(targetChars);
  return {
    label,
    text,
    history: [{ role: "user", content: text }],
  };
}

/** Short smoke-test prompt (~100 chars, tens of tokens). */
export const SHORT_PROMPT: BenchPrompt = {
  label: "short",
  text: "Say hi in five words or less.",
  history: [{ role: "user", content: "Say hi in five words or less." }],
};

/** Medium realistic chat turn (~2k chars, ~500 tokens). */
export const MEDIUM_PROMPT: BenchPrompt = buildBenchPrompt("medium", 2000);

/** Long prefill stress prompt (~8k chars, ~2k tokens). */
export const LONG_PROMPT: BenchPrompt = buildBenchPrompt("long", 8000);

/** Standard sweep: short → medium → long context sizes. */
export const BENCH_PROMPTS: BenchPrompt[] = [
  SHORT_PROMPT,
  MEDIUM_PROMPT,
  LONG_PROMPT,
];

/**
 * Render a one-line summary row for console benchmark tables.
 *
 * @param label Row label.
 * @param promptTokens Input-context tokens.
 * @param ttftMs Time to first token.
 * @param prefillTps Prefill tokens/sec.
 * @param decodeTps Decode tokens/sec.
 * @returns Fixed-width summary line.
 */
export function formatSweepRow(
  label: string,
  promptTokens: number,
  ttftMs: number,
  prefillTps: number,
  decodeTps: number,
): string {
  return `${label.padEnd(8)} | ctx ${String(promptTokens).padStart(5)} tok | ` +
    `TTFT ${ttftMs.toFixed(1).padStart(8)} ms | ` +
    `prefill ${prefillTps.toFixed(1).padStart(8)} tok/s | ` +
    `decode ${decodeTps.toFixed(1).padStart(8)} tok/s`;
}
