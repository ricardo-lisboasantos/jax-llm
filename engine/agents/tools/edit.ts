import type { Tool, ToolParams, ToolResult } from "./tool.ts";

function toResult(text: string): ToolResult {
  return { read: () => Promise.resolve(text) };
}

export class Edit implements Tool {
  get_description(): string {
    return "Edit a text file: params { path, oldText, newText }.";
  }
  async run(params: ToolParams): Promise<ToolResult> {
    const raw = params.read();
    let parsed: { path?: string; oldText?: string; newText?: string };
    try {
      parsed = JSON.parse(raw);
    } catch {
      return toResult("error: params must be JSON");
    }
    if (
      !parsed.path || parsed.oldText === undefined ||
      parsed.newText === undefined
    ) {
      return toResult("error: missing path/oldText/newText");
    }
    try {
      const current = await Deno.readTextFile(parsed.path);
      if (!current.includes(parsed.oldText)) {
        return toResult("error: oldText not found");
      }
      const updated = current.replace(parsed.oldText, parsed.newText);
      await Deno.writeTextFile(parsed.path, updated);
      return toResult("ok");
    } catch (e) {
      return toResult(`error: ${(e as Error).message}`);
    }
  }
}
