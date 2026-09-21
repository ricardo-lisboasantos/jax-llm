/**
 * unit: ChatModel registry (engine/llm/model.ts).
 */
import { assert, assertEquals } from "@std/assert";
import { CHAT_MODEL_IDS, CHAT_MODELS } from "./model.ts";

Deno.test("unit: model registry has all families", () => {
  assert(CHAT_MODEL_IDS.length >= 7, `got ${CHAT_MODEL_IDS.length}`);
  for (const id of CHAT_MODEL_IDS) {
    const m = CHAT_MODELS[id];
    assert(m.weightsUrl.length > 0, `${id} weightsUrl`);
    assert(m.tokenizerUrl.length > 0, `${id} tokenizerUrl`);
    assert(m.contextSize > 0, `${id} contextSize`);
    assert(m.defaults.temperature !== undefined);
  }
});

Deno.test("unit: model formatPrompt is non-empty", () => {
  for (const id of CHAT_MODEL_IDS) {
    const m = CHAT_MODELS[id];
    const out = m.formatPrompt([{ role: "user", content: "Hi" }]);
    assert(out.length > 0, `${id} formatPrompt empty`);
    assert(out.includes("Hi"), `${id} prompt missing content`);
  }
});

Deno.test("unit: model encodePrompt prepends BOS + stopTokens", () => {
  const fakeTok = {
    bosToken: 1,
    eosToken: 2,
    encode: (t: string) => [t.length],
    decode: (t: number[]) => t.join(","),
    decodeGenerated: (t: number[]) => t.join(","),
  };
  for (const id of CHAT_MODEL_IDS) {
    const m = CHAT_MODELS[id];
    const ids = m.encodePrompt(fakeTok, [{ role: "user", content: "Hi" }]);
    assertEquals(ids[0], 1);
    const stops = m.stopTokens(fakeTok);
    assert(stops.length > 0, `${id} stopTokens empty`);
  }
});
