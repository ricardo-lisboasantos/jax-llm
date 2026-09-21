import type { Tool } from "./tools/tool.ts";

export type Action = {
  ToolCalled: Tool;
  Timestamp: Date;
};
