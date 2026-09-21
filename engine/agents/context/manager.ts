import type { MemoryEntry } from "../memory.ts";

/**
 * Minimal language-model interface for context summarization.
 * Structurally compatible with `ChatEngine` (`engine/chat/chat_engine.ts`),
 * so a live engine can be passed directly — or a lightweight fake in tests.
 */
export type ContextSummarizer = {
  generate(prompt: string): Promise<string>;
};

export type FunnelOptions = {
  /** Recent entries kept verbatim instead of summarized (default: maxEntries). */
  keepRecent?: number;
  /** Max transcript chars sent to the LM per call (default 12_000). */
  maxPromptChars?: number;
  /** Instruction prefix for the summary prompt. */
  instruction?: string;
};

export type FunneledContext = {
  /** Compressed summary of the older (non-recent) entries. Empty when nothing overflowed. */
  summary: string;
  /** Most recent entries, kept verbatim. */
  recent: MemoryEntry[];
  /** Total entries consumed. */
  consumed: number;
};

const DEFAULT_INSTRUCTION =
  "Summarize the important parts of the following conversation context " +
  "into a short reduced text. Preserve key facts, decisions, user goals, " +
  "and open questions. Drop greetings and filler.";

export class ContextManager {
  private maxEntries: number;
  private summarizer?: ContextSummarizer;
  private funnelOptions: Required<FunnelOptions>;

  constructor(
    maxEntries = 100,
    summarizer?: ContextSummarizer,
    options: FunnelOptions = {},
  ) {
    this.maxEntries = maxEntries;
    this.summarizer = summarizer;
    this.funnelOptions = {
      keepRecent: options.keepRecent ?? maxEntries,
      maxPromptChars: options.maxPromptChars ?? 12_000,
      instruction: options.instruction ?? DEFAULT_INSTRUCTION,
    };
  }

  /** Trim history to the most recent entries within the window. */
  window(entries: MemoryEntry[]): MemoryEntry[] {
    if (entries.length <= this.maxEntries) return [...entries];
    return entries.slice(entries.length - this.maxEntries);
  }

  /** Render entries as a plain-text transcript. */
  render(entries: MemoryEntry[]): string {
    return this.window(entries)
      .map((e) => `${e.role}: ${e.content}`)
      .join("\n");
  }

  /** Build the LM prompt that funnels a transcript into a summary. */
  buildSummaryPrompt(entries: MemoryEntry[]): string {
    let transcript = entries
      .map((e) => `${e.role}: ${e.content}`)
      .join("\n");
    const max = this.funnelOptions.maxPromptChars;
    if (transcript.length > max) {
      transcript = transcript.slice(0, max) + "\n[…truncated…]";
    }
    return `${this.funnelOptions.instruction}\n\n${transcript}`;
  }

  /**
   * Funnel context through the language model into a reduced summary text.
   *
   * Uses the injected LM when available; falls back to a deterministic
   * extractive summary (first lines per entry) when no LM is configured,
   * the LM returns nothing useful, or the LM call fails — so callers
   * always get usable reduced text, including offline.
   */
  async summarize(
    entries: MemoryEntry[],
    summarizer?: ContextSummarizer,
  ): Promise<string> {
    if (entries.length === 0) return "";
    const lm = summarizer ?? this.summarizer;
    if (!lm) return this.summarizeExtractive(entries);
    try {
      const out = (await lm.generate(this.buildSummaryPrompt(entries))).trim();
      return out.length > 0 ? out : this.summarizeExtractive(entries);
    } catch {
      return this.summarizeExtractive(entries);
    }
  }

  /**
   * Split history into a compressed summary of older entries plus the most
   * recent entries kept verbatim — the classic context funnel.
   *
   * When everything fits in the recent window, `summary` is empty and
   * `recent` holds all entries (no information is lost to compression).
   */
  async funnel(
    entries: MemoryEntry[],
    summarizer?: ContextSummarizer,
  ): Promise<FunneledContext> {
    const keep = Math.max(1, this.funnelOptions.keepRecent);
    const recent = entries.slice(Math.max(0, entries.length - keep));
    const older = entries.slice(0, entries.length - recent.length);
    const summary = older.length > 0
      ? await this.summarize(older, summarizer)
      : "";
    return { summary, recent, consumed: entries.length };
  }

  /** Render a funneled context back into prompt-ready text. */
  renderFunneled(ctx: FunneledContext): string {
    const recent = ctx.recent.map((e) => `${e.role}: ${e.content}`).join("\n");
    if (!ctx.summary) return recent;
    const summary = `[Summary of earlier context]\n${ctx.summary}`;
    return recent ? `${summary}\n\n[Recent messages]\n${recent}` : summary;
  }

  /** Deterministic extractive fallback: first ~200 chars per entry. */
  summarizeExtractive(entries: MemoryEntry[]): string {
    return entries
      .map((e) => {
        const content = e.content.trim().split("\n")[0].slice(0, 200);
        return `${e.role}: ${content}`;
      })
      .join("\n");
  }
}
