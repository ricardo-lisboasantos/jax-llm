import { randomUUID, type UUID } from "node:crypto";
import type { ChatModel } from "../llm/model.ts";

export interface Cache<K, V> {
  add(value: V): K;
  get(key: K): V | undefined;
  set(key: K, new_val: V): void;
}

export class ModelsCache implements Cache<UUID, ChatModel> {
  private store = new Map<UUID, ChatModel>();

  add(value: ChatModel): UUID {
    const id = randomUUID();
    this.store.set(id, value);
    return id;
  }
  get(key: UUID): ChatModel | undefined {
    return this.store.get(key);
  }
  set(key: UUID, new_val: ChatModel): void {
    this.store.set(key, new_val);
  }

  get size(): number {
    return this.store.size;
  }
}

export class LayersCache implements Cache<string, unknown> {
  private store = new Map<string, unknown>();

  add(value: unknown): string {
    const id = randomUUID();
    this.store.set(id, value);
    return id;
  }
  get(key: string): unknown {
    return this.store.get(key);
  }
  set(key: string, new_val: unknown): void {
    this.store.set(key, new_val);
  }
}

export class PromptsCache implements Cache<string, string> {
  private store = new Map<string, string>();

  add(value: string): string {
    const key = `prompt:${this.store.size}`;
    this.store.set(key, value);
    return key;
  }
  get(key: string): string | undefined {
    return this.store.get(key);
  }
  set(key: string, new_val: string): void {
    this.store.set(key, new_val);
  }
}

export class RagCache implements Cache<string, string[]> {
  private store = new Map<string, string[]>();

  add(value: string[]): string {
    const key = `rag:${this.store.size}`;
    this.store.set(key, value);
    return key;
  }
  get(key: string): string[] | undefined {
    return this.store.get(key);
  }
  set(key: string, new_val: string[]): void {
    this.store.set(key, new_val);
  }
}
