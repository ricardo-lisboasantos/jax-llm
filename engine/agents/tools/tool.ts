export interface ToolResult {
  read(): Promise<string>;
}

export interface ToolParams {
  read(): string;
}

export interface Tool {
  get_description(): string;
  run(params: ToolParams): Promise<ToolResult>;
}
