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
  await assertRejects(
    () => engine.chat([{ role: "user", content: "hello" }]),
    Error,
    "init()",
  );
});

Deno.test("unit: generate without init() throws", async () => {
  const engine = new ChatEngine("lfm2.5-350m");
  await assertRejects(() => engine.generate("hello"), Error, "init()");
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
  assertEquals(
    await engine.chat([{ role: "user", content: "hi" }]),
    "Hello world!",
  );
});

Deno.test("unit: generate wraps prompt as single user message", async () => {
  class FakeChatEngine extends ChatEngine {
    lastHistory?: ChatMessage[];
    override chat(history: ChatMessage[]): Promise<string> {
      this.lastHistory = history;
      return Promise.resolve("fake-reply");
    }
  }
  const engine = new FakeChatEngine("lfm2.5-350m");
  assertEquals(await engine.generate("hello"), "fake-reply");
  assertEquals(engine.lastHistory, [{ role: "user", content: "hello" }]);
});

Deno.test("unit: generateStream delegates to chatStream", async () => {
  class FakeStreamEngine extends ChatEngine {
    override async *chatStream(
      _history: ChatMessage[],
    ): AsyncGenerator<string, void, unknown> {
      yield "a";
      yield "ab";
    }
  }
  const chunks: string[] = [];
  for await (
    const c of new FakeStreamEngine("lfm2.5-350m").generateStream("hi")
  ) {
    chunks.push(c);
  }
  assertEquals(chunks, ["a", "ab"]);
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
