import type { Database } from "./database.ts";

export class DenoKVDB<K extends Deno.KvKeyPart, V> implements Database<K, V> {
  private kv: Promise<Deno.Kv>;

  constructor() {
    this.kv = Deno.openKv();
  }
  public async get(key: K): Promise<V> {
    return (await (await this.kv).get([key])).value as V;
  }

  public async add(key: K, value: V): Promise<void> {
    await (await this.kv).set([key], value);
  }

  public async update(key: K, new_value: V): Promise<void> {
    await (await this.kv).set([key], new_value);
  }
}
