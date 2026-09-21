export type MemoryEntry = {
  role: string;
  content: string;
  timestamp: Date;
};

export class Memory {
  private entries: MemoryEntry[] = [];

  remember(role: string, content: string): void {
    this.entries.push({ role, content, timestamp: new Date() });
  }

  recall(): MemoryEntry[] {
    return [...this.entries];
  }

  clear(): void {
    this.entries = [];
  }

  get size(): number {
    return this.entries.length;
  }
}
