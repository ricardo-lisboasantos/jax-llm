/**
 * unit: configuration handling.
 */
import { assert, assertEquals } from "@std/assert";
import { createConfig, DEFAULT_CONFIG, loadConfig } from "./config.ts";

Deno.test("unit: DEFAULT_CONFIG has sane values", () => {
  assertEquals(DEFAULT_CONFIG.chat.backend, "webgpu");
  assertEquals(DEFAULT_CONFIG.chat.modelId, "lfm2.5-350m");
  assert(DEFAULT_CONFIG.chat.maxTokens > 0);
});

Deno.test("unit: loadConfig returns defaults when no file", () => {
  // No config.json in repo root by default — falls back to defaults.
  // Point at a definitely-missing file to force the fallback path.
  Deno.env.set("JAX_JS_CONFIG_PATH", "./does-not-exist-12345.json");
  try {
    const cfg = loadConfig();
    assertEquals(cfg.chat.modelId, DEFAULT_CONFIG.chat.modelId);
  } finally {
    Deno.env.delete("JAX_JS_CONFIG_PATH");
  }
});

Deno.test("unit: createConfig maps aliases and backends", () => {
  const a = createConfig({ modelId: "gemma" });
  assertEquals(a.chat.modelId, "gemma-3-270m");
  const b = createConfig({ modelId: "lfm" });
  assertEquals(b.chat.modelId, "lfm2.5-350m");
  const c = createConfig({ backend: "wasm" });
  assertEquals(c.chat.backend, "wasm");
  const d = createConfig({ modelId: "org/custom" });
  assertEquals(d.chat.modelId, "org/custom");
});
