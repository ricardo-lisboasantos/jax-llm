/**
 * unit: ChatEngine API (offline — construction + guards only).
 */
import { assert, assertEquals, assertRejects } from "@std/assert";
import { ChatEngine } from "../chat/chat_engine.ts";
import type { ChatMessage } from "../chat/types.ts";
import { ChatEngine as ChatEngineReexport } from "./chat.ts";

Deno.test("unit: chat re-export matches ChatEngine", () => {
  assert(ChatEngineReexport === ChatEngine);
});

Deno.test("unit: chat engine constructs with aliases", () => {
  const a = new ChatEngine("lfm");
  assert(a instanceof ChatEngine);
  const b = new ChatEngine("gemma", { backend: "wasm", maxTokens: 512 });
  assert(b instanceof ChatEngine);
});

Deno.test("unit: chat without init() throws", async () => {
  const engine = new ChatEngine("lfm2.5-350m");
  await assertRejects(() => engine.chat("hello"), Error, "init()");
});

Deno.test("unit: chat returns final reply without duplicating stream chunks", async () => {
  // chatStream() yields cumulative text; chat() must keep the latest chunk,
  // not concatenate them ("Hello" + "Hello world" + ...).
  class FakeStreamingEngine extends ChatEngine {
    override async *chatStream(
      _history: ChatMessage[],
    ): AsyncGenerator<string, void, unknown> {
      yield "Hello";
      yield "Hello world";
      yield "Hello world!";
    }
  }
  const engine = new FakeStreamingEngine("lfm2.5-350m");
  assertEquals(await engine.chat("hi"), "Hello world!");
});

Deno.test("unit: chatStream without init() throws", async () => {
  const engine = new ChatEngine("lfm2.5-350m");
  await assertRejects(
    async () => {
      for await (
        const _ of engine.chatStream([{ role: "user", content: "hi" }])
      ) {
        // unreachable
      }
    },
    Error,
    "init()",
  );
});
