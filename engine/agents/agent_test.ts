/**
 * unit: agents (Persona, Memory, ContextManager, Toolbox, tools).
 */
import { assert, assertEquals, assertStringIncludes } from "@std/assert";
import { Agent } from "./agent.ts";
import { Persona } from "./persona.ts";
import { Memory } from "./memory.ts";
import { ContextManager } from "./context/manager.ts";
import { Toolbox } from "./tools/toolbox.ts";
import { Edit } from "./tools/edit.ts";
import { Fetch } from "./tools/fetch.ts";
import { Terminal } from "./tools/terminal.ts";

Deno.test("unit: Persona system message", () => {
  const p = new Persona("Ada", "Be concise.");
  assertStringIncludes(p.get_system_message(), "Ada");
  assertStringIncludes(p.get_system_message(), "Be concise");
  const q = new Persona("Bob");
  assertStringIncludes(q.get_system_message(), "Bob");
});

Deno.test("unit: Memory remember/recall/clear", () => {
  const m = new Memory();
  m.remember("user", "hi");
  assertEquals(m.size, 1);
  assertEquals(m.recall()[0].content, "hi");
  m.clear();
  assertEquals(m.size, 0);
});

Deno.test("unit: ContextManager windows + renders", () => {
  const cm = new ContextManager(2);
  const entries = [
    { role: "user", content: "a", timestamp: new Date() },
    { role: "user", content: "b", timestamp: new Date() },
    { role: "user", content: "c", timestamp: new Date() },
  ];
  assertEquals(cm.window(entries).map((e) => e.content), ["b", "c"]);
  assertStringIncludes(cm.render(entries), "c");
});

Deno.test("unit: ContextManager funnel summarizes older, keeps recent", async () => {
  const calls: string[] = [];
  const fakeLm = {
    chat: (prompt: string) => {
      calls.push(prompt);
      return Promise.resolve("User wants X; decided Y.");
    },
  };
  const cm = new ContextManager(10, fakeLm, { keepRecent: 2 });
  const entries = [
    { role: "user", content: "I want X", timestamp: new Date() },
    { role: "assistant", content: "We decided Y", timestamp: new Date() },
    { role: "user", content: "latest question", timestamp: new Date() },
    { role: "assistant", content: "latest answer", timestamp: new Date() },
  ];
  const funneled = await cm.funnel(entries);
  assertEquals(funneled.summary, "User wants X; decided Y.");
  assertEquals(
    funneled.recent.map((e) => e.content),
    ["latest question", "latest answer"],
  );
  assertEquals(funneled.consumed, 4);
  // The LM prompt contains the older context, not the recent tail.
  assertStringIncludes(calls[0], "I want X");
  const text = cm.renderFunneled(funneled);
  assertStringIncludes(text, "User wants X; decided Y.");
  assertStringIncludes(text, "latest question");
});

Deno.test("unit: ContextManager funnel needs no summary within budget", async () => {
  const cm = new ContextManager(10, {
    chat: () => Promise.reject(new Error("should not be called")),
  });
  const entries = [{ role: "user", content: "hi", timestamp: new Date() }];
  const funneled = await cm.funnel(entries);
  assertEquals(funneled.summary, "");
  assertEquals(funneled.recent.length, 1);
  assertEquals(cm.renderFunneled(funneled), "user: hi");
});

Deno.test("unit: ContextManager summarize falls back without LM", async () => {
  const cm = new ContextManager(10);
  const entries = [
    { role: "user", content: "hello world", timestamp: new Date() },
    { role: "assistant", content: "hi there", timestamp: new Date() },
  ];
  const out = await cm.summarize(entries);
  assertStringIncludes(out, "user: hello world");
  assertStringIncludes(out, "assistant: hi there");
  assertEquals(await cm.summarize([]), "");
});

Deno.test("unit: ContextManager summarize falls back on LM failure", async () => {
  const cm = new ContextManager(10, {
    chat: () => Promise.reject(new Error("offline")),
  });
  const out = await cm.summarize([
    { role: "user", content: "important fact", timestamp: new Date() },
  ]);
  assertStringIncludes(out, "important fact");
});

Deno.test("unit: Toolbox add/list", () => {
  const box = new Toolbox();
  assertEquals(box.size, 0);
  box.add_tool(new Edit());
  assertEquals(box.size, 1);
  assertEquals(box.list().length, 1);
});

Deno.test("unit: Agent exposes system message + tools", () => {
  const agent = new Agent(new Persona("Ada"), new Toolbox([new Edit()]));
  assertStringIncludes(agent.systemMessage, "Ada");
  assertEquals(agent.tools.length, 1);
});

Deno.test("unit: tools have descriptions", () => {
  assert(new Edit().get_description().length > 0);
  assert(new Fetch().get_description().length > 0);
  assert(new Terminal().get_description().length > 0);
});

Deno.test("unit: Edit tool validates params", async () => {
  const tool = new Edit();
  const bad = await tool.run({ read: () => "not-json" });
  assertStringIncludes(await bad.read(), "error");
  const missing = await tool.run({ read: () => "{}" });
  assertStringIncludes(await missing.read(), "error");
});
