/**
 * unit: Deno KV database implementation.
 */
import { assertEquals } from "@std/assert";
import { DenoKVDB } from "./kv.ts";

Deno.test("unit: DenoKVDB add/get/update roundtrip", async () => {
  const db = new DenoKVDB<string, string>();
  const key = `test:${Date.now()}:${Math.random()}`;
  await db.add(key, "hello");
  assertEquals(await db.get(key), "hello");
  await db.update(key, "world");
  assertEquals(await db.get(key), "world");
});
