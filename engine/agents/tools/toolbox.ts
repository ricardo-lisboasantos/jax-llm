import type { Tool } from "./tool.ts";

export class Toolbox {
  private tools: Array<Tool>;
  constructor(tools: Array<Tool> = []) {
    this.tools = [...tools];
  }

  add_tool(new_tool: Tool): void {
    this.tools.push(new_tool);
  }

  list(): Tool[] {
    return [...this.tools];
  }

  get size(): number {
    return this.tools.length;
  }
}
