/**
 * PLD verification tests (live model, LFM): batched `scoreTokens` must
 * exactly match sequential `step` calls, and truncate/confirm must
 * preserve session equivalence.
 *
 * Mirrors `tests/bench_test.ts` skip guards (network + compute backend).
 * Weights resolve from disk cache when warm.
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { ChatEngine } from "../mod.ts";

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

function isBackendUnavailable(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return msg.includes("Backend not initialized") || msg.includes("WebGPU");
}

function argmax(data: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i]! > data[best]!) best = i;
  }
  return best;
}

async function toF32(a: np.Array): Promise<Float32Array> {
  const raw = await a.data();
  return raw instanceof Float32Array
    ? raw
    : new Float32Array(raw as ArrayLike<number>);
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assertEquals(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

Deno.test("pld: scoreTokens matches sequential steps + truncate/confirm", async () => {
  if (!(await hasNetwork())) {
    console.log("SKIP pld: no network access");
    return;
  }
  const engine = new ChatEngine("lfm2.5-350m", {
    backend: "webgpu",
    maxTokens: 16,
    sampling: { temperature: 0 },
  });
  try {
    await engine.init();
  } catch (e) {
    if (!isBackendUnavailable(e)) throw e;
    console.log(`SKIP pld backend: ${(e as Error).message}`);
    return;
  }
  try {
    const runtime = engine.getRuntime();
    assert(
      typeof runtime.createSession().scoreTokens === "function",
      "LFM session must expose scoreTokens",
    );
    const tokenizer = runtime.getTokenizer();
    const prompt = runtime.definition.encodePrompt(tokenizer, [
      { role: "user", content: "The capital of France is" },
    ]);
    const drafts = [prompt[prompt.length - 1]!, 7, 7, 7].slice(0, 4);
    const k = drafts.length;

    // Baseline: prefill + k sequential steps.
    const base = runtime.createSession();
    const baseRows: Float32Array[] = [];
    try {
      let logits = await base.prefill(
        np.array(prompt, { dtype: np.uint32 }),
      );
      await logits.data();
      for (const t of drafts) {
        logits = await base.step(t);
        baseRows.push(await toF32(logits));
      }
    } finally {
      base.dispose();
    }

    // Scored: prefill + one batched scoreTokens.
    const scored = runtime.createSession();
    try {
      const prefillLogits = await scored.prefill(
        np.array(prompt, { dtype: np.uint32 }),
      );
      await prefillLogits.data();
      const all = await toF32(
        (await scored.scoreTokens!(np.array(drafts, { dtype: np.uint32 })))!,
      );
      const V = all.length / k;
      assertEquals(Number.isInteger(V), true);
      for (let i = 0; i < k; i++) {
        const row = all.slice(i * V, (i + 1) * V);
        assertEquals(
          argmax(row),
          argmax(baseRows[i]!),
          `row ${i} argmax must match`,
        );
        assertAlmostEquals(maxAbsDiff(row, baseRows[i]!), 0, 1e-2);
      }

      // Confirm path: state must equal post-step state — next step matches.
      scored.confirmDraft!();
      const nextScored = await toF32(await scored.step!(drafts[k - 1]!));
      const check = runtime.createSession();
      try {
        let l = await check.prefill(np.array(prompt, { dtype: np.uint32 }));
        await l.data();
        for (const t of drafts) {
          l = await check.step(t);
          await l.data();
        }
        const nextBase = await toF32(await check.step(drafts[k - 1]!));
        assertEquals(argmax(nextScored), argmax(nextBase));
      } finally {
        check.dispose();
      }
    } finally {
      scored.dispose();
    }

    // Truncate path: score then roll back — next step matches baseline.
    const rolled = runtime.createSession();
    try {
      const rolledPrefill = await rolled.prefill(
        np.array(prompt, { dtype: np.uint32 }),
      );
      await rolledPrefill.data();
      await rolled.scoreTokens!(np.array(drafts, { dtype: np.uint32 }));
      rolled.truncateDraft!();
      const after = await toF32(await rolled.step(drafts[0]!));
      assertEquals(argmax(after), argmax(baseRows[0]!));
      assertAlmostEquals(maxAbsDiff(after, baseRows[0]!), 0, 1e-2);
    } finally {
      rolled.dispose();
    }
  } finally {
    engine.dispose();
  }
});
