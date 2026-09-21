import type { Database } from "./database.ts";

/** Deno KV-backed implementation of the {@linkcode Database} interface. */
export class DenoKVDB<K extends Deno.KvKeyPart, V> implements Database<K, V> {
  private kv: Promise<Deno.Kv>;

  constructor() {
    this.kv = Deno.openKv();
  }
  /** Fetch the value stored under a key. */
  public async get(key: K): Promise<V> {
    return (await (await this.kv).get([key])).value as V;
  }

  /** Store a value under a key. */
  public async add(key: K, value: V): Promise<void> {
    await (await this.kv).set([key], value);
  }

  /** Overwrite the value stored under an existing key. */
  public async update(key: K, new_value: V): Promise<void> {
    await (await this.kv).set([key], new_value);
  }
}
