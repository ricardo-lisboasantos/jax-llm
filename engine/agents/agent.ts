import type { Persona } from "./persona.ts";
import type { Tool } from "./tools/tool.ts";
import type { Toolbox } from "./tools/toolbox.ts";

export class Agent {
  private persona: Persona;
  private toolbox: Toolbox;

  constructor(persona: Persona, toolbox: Toolbox) {
    this.persona = persona;
    this.toolbox = toolbox;
  }

  get systemMessage(): string {
    return this.persona.get_system_message();
  }

  get tools(): Tool[] {
    return this.toolbox.list();
  }
}
