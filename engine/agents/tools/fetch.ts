import type { Tool, ToolParams, ToolResult } from "./tool.ts";

export class Fetch implements Tool {
  get_description(): string {
    return "Fetch a URL: params { url } returns text.";
  }
  async run(params: ToolParams): Promise<ToolResult> {
    let parsed: { url?: string };
    try {
      parsed = JSON.parse(params.read());
    } catch {
      return { read: () => Promise.resolve("error: params must be JSON") };
    }
    if (!parsed.url) {
      return { read: () => Promise.resolve("error: missing url") };
    }
    try {
      const resp = await fetch(parsed.url);
      const text = await resp.text();
      const clipped = text.slice(0, 8000);
      return { read: () => Promise.resolve(clipped) };
    } catch (e) {
      const msg = `error: ${(e as Error).message}`;
      return { read: () => Promise.resolve(msg) };
    }
  }
}
