/**
 * unit: cache layer (ModelsCache, LayersCache, PromptsCache, RagCache, KV).
 */
import { assert, assertEquals } from "@std/assert";
import { LayersCache, ModelsCache, PromptsCache, RagCache } from "./caches.ts";
import { MemoryKVCache } from "../llm/cache/kv_cache.ts";
import { roundCacheCapacity } from "../llm/cache/pad_cache.ts";

Deno.test("unit: LayersCache add/get/set", () => {
  const c = new LayersCache();
  const k = c.add({ w: 1 });
  assert(c.get(k) !== undefined);
  c.set(k, { w: 2 });
  assertEquals((c.get(k) as { w: number }).w, 2);
});

Deno.test("unit: PromptsCache stores strings", () => {
  const c = new PromptsCache();
  const k = c.add("hello");
  assertEquals(c.get(k), "hello");
  c.set(k, "world");
  assertEquals(c.get(k), "world");
});

Deno.test("unit: RagCache stores string arrays", () => {
  const c = new RagCache();
  const k = c.add(["a", "b"]);
  assertEquals(c.get(k), ["a", "b"]);
});

Deno.test("unit: ModelsCache add/get/set", async () => {
  const { CHAT_MODELS } = await import("../llm/model.ts");
  const c = new ModelsCache();
  const model = CHAT_MODELS["gpt2"];
  const id = c.add(model);
  assert(c.get(id) !== undefined);
  assertEquals(c.size, 1);
  c.set(id, model);
  assertEquals(c.size, 1);
});

Deno.test("unit: MemoryKVCache contract", () => {
  const c = new MemoryKVCache<string, number>();
  assertEquals(c.size, 0);
  c.set("a", 1);
  assert(c.has("a"));
  assertEquals(c.get("a"), 1);
  c.clear();
  assertEquals(c.size, 0);
});

Deno.test("unit: roundCacheCapacity blocks", () => {
  assertEquals(roundCacheCapacity(1), 512);
  assertEquals(roundCacheCapacity(600), 1024);
});
