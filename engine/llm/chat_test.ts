/**
 * unit: ChatEngine API (offline — construction + guards only).
 */
import { assert, assertRejects } from "@std/assert";
import { ChatEngine } from "../chat/chat_engine.ts";
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
