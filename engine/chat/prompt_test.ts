/**
 * unit: prompt formatting per model family.
 */
import { assert, assertStringIncludes } from "@std/assert";
import {
  bonsaiPrompt,
  gemmaPrompt,
  genericPrompt,
  gptPrompt,
  lfmPrompt,
  maplePrompt,
  phiPrompt,
  qwenPrompt,
} from "./prompt.ts";
import type { ChatMessage } from "./types.ts";

const history: ChatMessage[] = [
  { role: "user", content: "Hello" },
  { role: "assistant", content: "Hi there" },
];

Deno.test("unit: gemmaPrompt uses start_of_turn markers", () => {
  const out = gemmaPrompt(history);
  assertStringIncludes(out, "<start_of_turn>user\nHello<end_of_turn>");
  assertStringIncludes(out, "<start_of_turn>model\nHi there<end_of_turn>");
  assert(out.endsWith("<start_of_turn>model\n"));
});

Deno.test("unit: lfmPrompt uses ChatML markers", () => {
  const out = lfmPrompt([{ role: "user", content: "Hi" }]);
  assertStringIncludes(out, "<|im_start|>user\nHi<|im_end|>");
  assert(out.endsWith("<|im_start|>assistant\n"));
});

Deno.test("unit: genericPrompt delegates to lfmPrompt", () => {
  assert(genericPrompt(history) === lfmPrompt(history));
});

Deno.test("unit: qwenPrompt and maplePrompt match lfmPrompt", () => {
  assert(qwenPrompt(history) === lfmPrompt(history));
  assert(maplePrompt(history) === lfmPrompt(history));
});

Deno.test("unit: bonsaiPrompt wraps user in INST", () => {
  const out = bonsaiPrompt(history);
  assertStringIncludes(out, "[INST]");
  assertStringIncludes(out, "Hello");
  assertStringIncludes(out, "</s>");
});

Deno.test("unit: bonsaiPrompt folds system message into INST", () => {
  const out = bonsaiPrompt([
    { role: "system", content: "Be brief" },
    { role: "user", content: "Hi" },
  ]);
  assertStringIncludes(out, "Be brief");
  assertStringIncludes(out, "[INST]");
});

Deno.test("unit: gptPrompt uses User/Assistant convention", () => {
  const out = gptPrompt(history);
  assertStringIncludes(out, "User: Hello");
  assertStringIncludes(out, "Assistant: Hi there");
  assert(out.endsWith("Assistant:"));
});

Deno.test("unit: phiPrompt uses Instruct/Output markers", () => {
  const out = phiPrompt(history);
  assertStringIncludes(out, "Instruct: Hello");
  assertStringIncludes(out, "Output: Hi there");
  assert(out.endsWith("Output:"));
});

Deno.test("unit: prompts skip empty content", () => {
  const out = lfmPrompt([
    { role: "user", content: "   " },
    { role: "user", content: "Real" },
  ]);
  assertStringIncludes(out, "Real");
  assert(!out.includes("<|im_start|>user\n<|im_end|>"));
});
