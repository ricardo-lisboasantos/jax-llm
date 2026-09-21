/**
 * bench: end-to-end TTFT + tokenize/generate throughput on a live model.
 *
 * Runs the same harness as the offline `engine/bench/` tests but wired to a
 * real `ChatEngine` (smallest default model, greedy sampling, few tokens so
 * CI stays fast). Skips gracefully without network access or without a usable
 * compute backend (WebGPU/WASM), mirroring `tests/integration_test.ts`.
 *
 * Live scope is deliberately narrow: short + medium prompts only, one
 * prefill per session. The multi-KB long prompt (KV-cache growth to 3072)
 * trips a jax-js refcount bug (`UseAfterFreeError` on a `float32[3072,8,64]`
 * cache leaf at `session.dispose()`), and same-session re-prefill is not
 * part of the session lifecycle `ChatEngine.chatStream` exercises (one
 * prefill + N steps per session). The full short→medium→long sweep math is
 * covered offline with mocked sessions in `engine/bench/harness_test.ts`.
 *
 * Run with:
 * ```bash
 * deno task bench
 * ```
 */
import { assert } from "@std/assert";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { ChatEngine } from "../mod.ts";
import {
  BENCH_PROMPTS,
  benchmarkChatEngine,
  benchmarkDecodeSteps,
  benchmarkEncode,
  benchmarkPrefillFn,
  formatSweepRow,
} from "../engine/bench/index.ts";

// Live prompts: short + medium only. The long (~8k) prompt is excluded —
// see the module docstring for why.
const LIVE_PROMPTS = BENCH_PROMPTS.filter((p) => p.label !== "long");

async function hasNetwork(): Promise<boolean> {
  try {
    const perm = await Deno.permissions.query({ name: "net" });
    if (perm.state !== "granted") return false;
  } catch {
    return false;
  }
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 5000);
    await fetch("https://huggingface.co", {
      method: "HEAD",
      signal: ctrl.signal,
    });
    clearTimeout(t);
    return true;
  } catch {
    return false;
  }
}

/** True for init failures caused by a missing compute backend (e.g. no GPU). */
function isBackendUnavailable(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return msg.includes("Backend not initialized") || msg.includes("WebGPU");
}

Deno.test("bench: TTFT + encode/prefill/decode throughput (needs network)", async () => {
  if (!(await hasNetwork())) {
    console.log("SKIP bench: no network access");
    return;
  }

  for (const backend of ["webgpu", "wasm"] as const) {
    // Greedy sampling for deterministic token counts; small cap for speed.
    const engine = new ChatEngine("lfm2.5-350m", {
      backend,
      maxTokens: 16,
      sampling: { temperature: 0 },
    });
    try {
      await engine.init();
    } catch (e) {
      if (!isBackendUnavailable(e)) throw e;
      console.log(`SKIP bench backend ${backend}: ${(e as Error).message}`);
      continue;
    }

    try {
      const runtime = engine.getRuntime();
      const tokenizer = runtime.getTokenizer();

      console.log(`\nbench: backend=${backend} model=${runtime.definition.id}`);
      for (const prompt of LIVE_PROMPTS) {
        const encode = benchmarkEncode(tokenizer, prompt.text, {
          warmup: 1,
          iterations: 5,
        });
        const gen = await benchmarkChatEngine(engine, prompt.history);

        console.log(
          formatSweepRow(
            prompt.label,
            gen.promptTokens,
            gen.ttftMs,
            gen.prefillTokensPerSecond,
            gen.decodeTokensPerSecond,
          ) +
            ` | encode ${encode.tokensPerSecond.toFixed(1)} tok/s` +
            ` | total ${gen.totalTokensPerSecond.toFixed(1)} tok/s` +
            ` (${gen.generatedTokens} gen tok)`,
        );

        assert(gen.promptTokens > 0, `${prompt.label}: empty prompt tokens`);
        assert(
          gen.generatedTokens > 0,
          `${prompt.label}: model generated no tokens`,
        );
        assert(gen.ttftMs > 0, `${prompt.label}: TTFT not measured`);
        assert(
          gen.prefillTokensPerSecond > 0,
          `${prompt.label}: prefill tok/s not measured`,
        );
        assert(
          gen.decodeTokensPerSecond > 0,
          `${prompt.label}: decode tok/s not measured`,
        );
        assert(
          encode.tokensPerSecond > 0,
          `${prompt.label}: encode tok/s not measured`,
        );
      }

      // Session-level prefill vs decode split on the short prompt (one
      // prefill + an 8-step decode sweep, forcing `.data()` so lazy GPU
      // execution is included in the timing). Exactly one prefill per
      // session — same lifecycle `ChatEngine.chatStream` uses.
      const shortIds = runtime.definition.encodePrompt(
        tokenizer,
        BENCH_PROMPTS[0].history,
      );
      const session = runtime.createSession();
      try {
        const prefill = await benchmarkPrefillFn(
          async () => {
            const logits = session.prefill(
              np.array(shortIds, { dtype: np.uint32 }),
            );
            await logits.data();
          },
          shortIds.length,
          { warmup: 0, iterations: 1 },
        );
        const decode = await benchmarkDecodeSteps(
          async (t: number) => {
            const logits = session.step(t);
            await logits.data();
          },
          Array.from(
            { length: 8 },
            (_, i) => shortIds[i % shortIds.length] ?? 0,
          ),
        );

        console.log(
          `session | prefill ${prefill.tokensPerSecond.toFixed(1)} tok/s ` +
            `(${
              prefill.elapsedMs.toFixed(1)
            } ms for ${prefill.promptTokens} tok) | ` +
            `decode ${decode.tokensPerSecond.toFixed(1)} tok/s ` +
            `(p50 ${decode.latency.p50Ms.toFixed(1)} ms/tok)`,
        );
        assert(prefill.tokensPerSecond > 0, "prefill tok/s not measured");
        assert(decode.tokensPerSecond > 0, "decode tok/s not measured");
      } finally {
        session.dispose();
      }
    } finally {
      engine.dispose();
    }
    return;
  }
  console.log("SKIP bench: no usable backend (webgpu nor wasm)");
});
