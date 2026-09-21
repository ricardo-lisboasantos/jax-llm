/**
 * @module tokenizer/encoder
 *
 * Byte-pair encoding (BPE) encoder used by {@link HuggingFaceBpeTokenizer}.
 *
 * This is a lightweight, dependency-free encoder for offline use and tests.
 * It supports greedy longest-match tokenization against a vocab map plus
 * special-token handling.
 */

export type EncoderVocab = Record<string, number>;

/** Simple greedy BPE / word-level encoder. */
export class TokenizerEncoder {
  readonly vocab: Map<string, number>;
  readonly specialTokens: Map<string, number>;

  constructor(vocab: EncoderVocab = {}, specialTokens: EncoderVocab = {}) {
    this.vocab = new Map(Object.entries(vocab));
    this.specialTokens = new Map(Object.entries(specialTokens));
  }

  /** Encode text into token IDs using greedy longest-match. */
  encode(text: string): number[] {
    if (text.length === 0) return [];
    // Fast path: whole-text special token (e.g. "<s>").
    const special = this.specialTokens.get(text);
    if (special !== undefined) return [special];

    const ids: number[] = [];
    let i = 0;
    while (i < text.length) {
      let matchId: number | undefined;
      let matchLen = 0;
      // Try longest match up to 16 chars ahead.
      const maxLen = Math.min(16, text.length - i);
      for (let len = maxLen; len >= 1; len--) {
        const piece = text.slice(i, i + len);
        const id = this.vocab.get(piece) ?? this.specialTokens.get(piece);
        if (id !== undefined) {
          matchId = id;
          matchLen = len;
          break;
        }
      }
      if (matchId === undefined) {
        // Unknown character: fall back to byte value + vocab offset guard.
        // Use 0 when vocab is empty so behavior stays deterministic.
        const fallback = this.vocab.get(text[i]);
        ids.push(fallback ?? 0);
        i += 1;
      } else {
        ids.push(matchId);
        i += matchLen;
      }
    }
    return ids;
  }

  /** Number of entries in the vocab. */
  get size(): number {
    return this.vocab.size + this.specialTokens.size;
  }
}
