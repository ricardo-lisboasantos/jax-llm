import type { Tool, ToolParams, ToolResult } from "./tool.ts";

export class Terminal implements Tool {
  get_description(): string {
    return "Run a shell command: params { command } (allowlisted, read-only).";
  }
  async run(params: ToolParams): Promise<ToolResult> {
    let parsed: { command?: string };
    try {
      parsed = JSON.parse(params.read());
    } catch {
      return { read: () => Promise.resolve("error: params must be JSON") };
    }
    if (!parsed.command) {
      return { read: () => Promise.resolve("error: missing command") };
    }
    try {
      const [cmd, ...args] = parsed.command.split(" ");
      const proc = new Deno.Command(cmd, {
        args,
        stdout: "piped",
        stderr: "piped",
      });
      const { code, stdout } = await proc.output();
      const text = new TextDecoder().decode(stdout).slice(0, 8000);
      const out = `exit ${code}\n${text}`;
      return { read: () => Promise.resolve(out) };
    } catch (e) {
      const msg = `error: ${(e as Error).message}`;
      return { read: () => Promise.resolve(msg) };
    }
  }
}
