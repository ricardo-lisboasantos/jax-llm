/**
 * integration: full engine initialization and model downloads.
 *
 * Requires network access. Skips gracefully when offline so
 * `deno test` and `deno task dev` run clean without `--allow-net`.
 */
import { assert } from "@std/assert";
import { ChatEngine } from "../mod.ts";
import { resolveModel } from "../engine/runtime/registry.ts";

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

Deno.test("integration: ChatEngine init + generate (needs network)", async () => {
  if (!(await hasNetwork())) {
    console.log("SKIP integration: no network access");
    return;
  }
  // CI containers have no GPU, so webgpu init throws
  // ("Backend not initialized: webgpu"). Prefer webgpu, fall back to wasm,
  // and skip only if neither backend initializes.
  for (const backend of ["webgpu", "wasm"] as const) {
    const engine = new ChatEngine("lfm2.5-350m", { maxTokens: 16, backend });
    try {
      await engine.init();
    } catch (e) {
      if (!isBackendUnavailable(e)) throw e;
      console.log(`SKIP backend ${backend}: ${(e as Error).message}`);
      continue;
    }
    try {
      const reply = await engine.generate("Say hi in five words or less.");
      assert(reply.length > 0, "expected non-empty reply");
    } finally {
      engine.dispose();
    }
    return;
  }
  console.log("SKIP integration: no usable backend (webgpu nor wasm)");
});

/** True for init failures caused by a missing compute backend (e.g. no GPU). */
function isBackendUnavailable(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return msg.includes("Backend not initialized") || msg.includes("WebGPU");
}
