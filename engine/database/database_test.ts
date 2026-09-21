/**
 * unit: Database interface contract (in-memory stand-in).
 */
import { assertEquals } from "@std/assert";
import type { Database } from "./database.ts";

class MemoryDB<K, V> implements Database<K, V> {
  private store = new Map<K, V>();
  get(key: K): Promise<V> {
    const v = this.store.get(key);
    if (v === undefined) return Promise.reject(new Error("not found"));
    return Promise.resolve(v);
  }
  add(key: K, value: V): Promise<void> {
    this.store.set(key, value);
    return Promise.resolve();
  }
  update(key: K, new_value: V): Promise<void> {
    this.store.set(key, new_value);
    return Promise.resolve();
  }
}

Deno.test("unit: database add/get/update roundtrip", async () => {
  const db: Database<string, string> = new MemoryDB();
  await db.add("k1", "v1");
  assertEquals(await db.get("k1"), "v1");
  await db.update("k1", "v2");
  assertEquals(await db.get("k1"), "v2");
});

Deno.test("unit: database supports numeric values", async () => {
  const db: Database<string, number> = new MemoryDB();
  await db.add("n", 42);
  assertEquals(await db.get("n"), 42);
});
