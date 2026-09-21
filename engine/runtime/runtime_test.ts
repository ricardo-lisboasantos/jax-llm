/**
 * unit: ModelRuntime lifecycle (offline — no device init, no network).
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelRuntime } from "./runtime.ts";

Deno.test("unit: runtime resolves definition on construct", () => {
  const rt = new ModelRuntime("lfm");
  assertEquals(rt.definition.id, "lfm2.5-350m");
  assertEquals(rt.isLoaded, false);
});

Deno.test("unit: runtime accepts overrides", () => {
  const rt = new ModelRuntime("gemma", {
    backend: "wasm",
    dtype: "float32" as never,
    weightOverrides: { "gemma-3-270m": "https://mirror/w" },
  });
  assertEquals(rt.definition.weightsUrl, "https://mirror/w");
});

Deno.test("unit: runtime guards throw before load()", () => {
  const rt = new ModelRuntime("lfm2.5-350m");
  let threwTokenizer = false;
  try {
    rt.getTokenizer();
  } catch {
    threwTokenizer = true;
  }
  assert(threwTokenizer, "getTokenizer should throw before load()");

  let threwSession = false;
  try {
    rt.createSession();
  } catch {
    threwSession = true;
  }
  assert(threwSession, "createSession should throw before load()");

  // createSession guard is synchronous.
  assertThrows(() => rt.createSession(), Error);
});
