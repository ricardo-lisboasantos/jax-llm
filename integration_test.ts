/**
 * integration: full engine initialization and model downloads.
 *
 * Requires network access. Skips gracefully when offline so
 * `deno test` and `deno task dev` run clean without `--allow-net`.
 */
import { assert } from "@std/assert";
import { ChatEngine } from "./mod.ts";
import { resolveModel } from "./engine/runtime/registry.ts";

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

Deno.test("integration: registry resolves all built-ins offline", () => {
  for (const id of ["lfm2.5-350m", "gemma-3-270m", "qwen2.5-0.5b"]) {
    const def = resolveModel(id);
    assert(def.weightsUrl.startsWith("https://"), id);
  }
});

Deno.test("integration: ChatEngine init + chat (needs network)", async () => {
  if (!(await hasNetwork())) {
    console.log("SKIP integration: no network access");
    return;
  }
  const engine = new ChatEngine("lfm2.5-350m", { maxTokens: 16 });
  await engine.init();
  const reply = await engine.chat("Say hi in five words or less.");
  assert(reply.length > 0, "expected non-empty reply");
});
