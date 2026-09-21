/**
 * @module tokenizer/decoder
 *
 * Token-to-text decoder counterpart to `tokenizer/encoder.ts`.
 */

export type DecoderVocab = Record<number, string>;

/** Simple ID-to-string decoder. */
export class TokenizerDecoder {
  readonly invVocab: Map<number, string>;
  readonly specialTokenIds: Set<number>;

  constructor(
    vocab: Record<string, number> = {},
    specialTokenIds: Iterable<number> = [],
  ) {
    this.invVocab = new Map(
      Object.entries(vocab).map(([piece, id]) => [id, piece]),
    );
    this.specialTokenIds = new Set(specialTokenIds);
  }

  /** Decode token IDs back to text. Unknown IDs become `<unk:{id}>`. */
  decode(tokens: number[]): string {
    return tokens
      .map((id) => this.invVocab.get(id) ?? `<unk:${id}>`)
      .join("");
  }

  /** Decode generated tokens, filtering special/control tokens. */
  decodeGenerated(tokens: number[]): string {
    return this.decode(tokens.filter((t) => !this.specialTokenIds.has(t)));
  }
}
