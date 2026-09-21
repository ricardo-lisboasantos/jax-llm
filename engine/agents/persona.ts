import { randomUUID, type UUID } from "node:crypto";

export class Persona {
  private name: string;
  private memory_id: UUID;
  private instructions: string;

  constructor(name: string, instructions = "") {
    this.name = name;
    this.memory_id = randomUUID();
    this.instructions = instructions;
  }

  public get_system_message(): string {
    const base = `You are ${this.name}, a helpful AI assistant.`;
    return this.instructions ? `${base} ${this.instructions}` : base;
  }

  public get name_(): string {
    return this.name;
  }

  public get memoryId(): UUID {
    return this.memory_id;
  }
}
