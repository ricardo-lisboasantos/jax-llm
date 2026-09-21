import "./opfs.ts";
import { tokenizers } from "npm:@jax-js/loaders@^0.1.3";

type AddedToken = {
  id: number;
  content: string;
  special: boolean;
};

/** Union of pre-tokenizer shapes found in HuggingFace tokenizer.json files. */
type PreTokenizer = {
  type: string;
  pretokenizers?: PreTokenizer[];
  pattern?: { Regex?: string; String?: string };
} | null;

type HuggingFaceBpeData = {
  added_tokens: AddedToken[];
  pre_tokenizer: PreTokenizer;
  normalizer?: {
    type: string;
    normalizers?: {
      type: string;
      prepend?: string;
      pattern?: { String?: string };
      content?: string;
    }[];
  };
  decoder?: {
    type: string;
    decoders?: {
      type: string;
      pattern?: { String?: string };
      content?: string;
      start?: number;
      stop?: number;
    }[];
  };
  model: {
    type: string;
    vocab: Record<string, number>;
    merges?: string[][] | string[];
    byte_fallback?: boolean;
    unk_token?: string;
    continuing_subword_prefix?: string | null;
    end_of_word_suffix?: string | null;
  };
};

/** Byte-level decoder map (GPT-2 / LFM2.5 style). */
function createByteDecoder(): Map<string, number> {
  const bytes: number[] = [];
  for (let i = 33; i <= 126; i++) bytes.push(i);
  for (let i = 161; i <= 172; i++) bytes.push(i);
  for (let i = 174; i <= 255; i++) bytes.push(i);

  const chars = [...bytes];
  let extra = 0;
  for (let byte = 0; byte < 256; byte++) {
    if (bytes.includes(byte)) continue;
    bytes.push(byte);
    chars.push(256 + extra++);
  }
  return new Map(
    chars.map((char, i) => [String.fromCodePoint(char), bytes[i]]),
  );
}

/** Decode a byte-level piece (GPT-2 style) to a hex string. */
function decodeByteLevelPiece(
  piece: string,
  byteDecoder: Map<string, number>,
): string {
  let hex = "";
  for (const char of piece) {
    const byte = byteDecoder.get(char);
    if (byte === undefined) {
      throw new Error(`Invalid byte-level tokenizer character: ${char}`);
    }
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

/** Convert a Uint8Array to a lowercase hex string. */
function bytesToHex(bytes: Uint8Array): string {
  let hex = "";
  for (const b of bytes) hex += b.toString(16).padStart(2, "0");
  return hex;
}

// ── SentencePiece-style BPE encoder ───────────────────────────────────────

/**
 * A BPE encoder that operates on Unicode characters instead of raw bytes.
 * Used for SentencePiece-style tokenizers (Llama, Mistral, Bonsai, etc.)
 * where vocab entries use `▁` (U+2581) as word boundaries and `<0xNN>`
 * tokens for byte fallback.
 */
class SentencePieceBpeEncoder {
  readonly #vocab: Map<string, number>;
  readonly #invVocab: Map<number, string>;
  readonly #mergeRanks: Map<string, number>;
  readonly #specialTokens: Map<string, number>;
  readonly #byteFallback: boolean;
  readonly #unkToken: string;

  constructor(
    vocab: Record<string, number>,
    merges: string[][] | string[] | undefined,
    specialTokens: Map<string, number>,
    byteFallback: boolean,
    unkToken: string,
  ) {
    this.#vocab = new Map(Object.entries(vocab));
    this.#invVocab = new Map();
    for (const [piece, id] of Object.entries(vocab)) {
      this.#invVocab.set(id, piece);
    }
    this.#specialTokens = specialTokens;
    this.#byteFallback = byteFallback;
    this.#unkToken = unkToken || "<unk>";

    // Build merge rank map from the merges list.
    // Lower rank = higher merge priority.
    this.#mergeRanks = new Map();
    if (merges) {
      const mergeList = Array.isArray(merges[0])
        ? (merges as string[][])
        : (merges as string[]).map((m) => m.split(" "));
      for (let i = 0; i < mergeList.length; i++) {
        const [a, b] = mergeList[i];
        this.#mergeRanks.set(`${a}${b}`, i);
      }
    }
  }

  encode(text: string): number[] {
    const result: number[] = [];

    // Split into segments: ▁ + word chars, or non-space non-▁ sequences.
    // This matches the SentencePiece pre-tokenization pattern.
    const regex = /▁[^\s▁]*|[^\s▁]+/gu;
    for (const segment of text.matchAll(regex)) {
      const word = segment[0];

      // Check if the full word is in the vocab.
      const fullId = this.#vocab.get(word);
      if (fullId !== undefined) {
        result.push(fullId);
        continue;
      }

      // Apply BPE merges on the word, character by character.
      result.push(...this.bpeEncode(word));
    }

    return result;
  }

  /** Apply BPE merges greedily to a word.
   *
   * Segmentation is tracked as a `boundaries` array of indices into `word`:
   * segment i is `word.slice(boundaries[i], boundaries[i + 1])`. Merging
   * adjacent segments i and i + 1 means removing `boundaries[i + 1]`. This
   * avoids rebuilding a string array on every merge — only integer shifts. */
  bpeEncode(word: string): number[] {
    const n = word.length;
    // boundaries: [0, 1, 2, ..., n] — segment i = word.slice(boundaries[i], boundaries[i + 1]).
    const boundaries: number[] = [];
    for (let i = 0; i <= n; i++) boundaries.push(i);

    while (boundaries.length > 2) {
      let bestRank = Infinity;
      let bestIdx = -1; // index of the LEFT boundary of the pair to merge.

      for (let i = 0; i < boundaries.length - 2; i++) {
        const pair = word.slice(boundaries[i], boundaries[i + 1]) +
          word.slice(boundaries[i + 1], boundaries[i + 2]);
        const rank = this.#mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break; // No more merges possible.

      // Merge segments bestIdx and bestIdx + 1 by dropping the shared boundary.
      boundaries.splice(bestIdx + 1, 1);
    }

    // Look up each segment in the vocab.
    const tokens: number[] = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const sym = word.slice(boundaries[i], boundaries[i + 1]);
      const id = this.#vocab.get(sym);
      if (id !== undefined) {
        tokens.push(id);
      } else if (this.#byteFallback) {
        // Fall back to byte tokens (<0xNN>).
        const encoded = new TextEncoder().encode(sym);
        for (const byte of encoded) {
          const byteToken = `<0x${
            byte.toString(16).toUpperCase().padStart(2, "0")
          }>`;
          const byteId = this.#vocab.get(byteToken);
          if (byteId !== undefined) {
            tokens.push(byteId);
          } else {
            // Unknown byte — use unk token.
            const unkId = this.#vocab.get(this.#unkToken);
            if (unkId !== undefined) tokens.push(unkId);
          }
        }
      } else {
        // No byte fallback — use unk token.
        const unkId = this.#vocab.get(this.#unkToken);
        if (unkId !== undefined) tokens.push(unkId);
      }
    }

    return tokens;
  }

  decode(tokens: number[]): string {
    let result = "";
    for (const id of tokens) {
      const piece = this.#invVocab.get(id);
      if (piece === undefined) {
        // Try special tokens.
        for (const [content, sid] of this.#specialTokens) {
          if (sid === id) {
            result += content;
            break;
          }
        }
        continue;
      }

      // Handle byte fallback tokens: <0xNN> → actual byte.
      const byteMatch = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
      if (byteMatch) {
        result += String.fromCharCode(parseInt(byteMatch[1], 16));
      } else {
        result += piece;
      }
    }
    // Denormalize: ▁ → space.
    result = result.replace(/▁/g, " ");
    return result;
  }
}

// ── Main tokenizer class ──────────────────────────────────────────────────

/**
 * Minimal Hugging Face BPE tokenizer.
 *
 * Supports two vocab styles found in tokenizer.json files:
 *
 *  1. **Byte-level BPE** (LFM2.5, GPT-2): vocab entries are byte-level
 *     encoded strings; uses a ByteLevel pre-tokenizer with a regex split.
 *     Uses `BpeEncoding` from @jax-js/loaders internally.
 *
 *  2. **SentencePiece-style BPE** (Bonsai, Llama, Mistral, etc.): vocab
 *     entries use `▁` (U+2581) as the word-boundary marker; uses a
 *     normalizer that prepends `▁` and replaces spaces with `▁`; falls
 *     back to `<0xNN>` byte tokens for unknown characters.
 *     Uses a custom Unicode-level BPE encoder.
 */
export class HuggingFaceBpeTokenizer {
  readonly bosToken: number;
  readonly eosToken: number;
  readonly padToken: number;
  readonly specialTokenIds: Set<number>;
  readonly #encoding: tokenizers.BpeEncoding | null;
  readonly #spEncoder: SentencePieceBpeEncoder | null;
  readonly #isSentencePieceStyle: boolean;

  constructor(data: HuggingFaceBpeData) {
    if (data.model.type !== "BPE") {
      throw new Error(`Expected a BPE tokenizer, found ${data.model.type}`);
    }

    // ── Detect tokenizer style ────────────────────────────────────────
    const hasByteLevelPreTokenizer = data.pre_tokenizer?.type === "ByteLevel" ||
      (data.pre_tokenizer?.type === "Sequence" &&
        data.pre_tokenizer.pretokenizers?.some(
          (p) => p?.type === "ByteLevel",
        ));
    this.#isSentencePieceStyle = !hasByteLevelPreTokenizer;

    // ── Extract special tokens ────────────────────────────────────────
    const addedTokensMap = new Map(
      data.added_tokens.map((t) => [t.id, t]),
    );
    const specialTokensEncoder: Record<string, number> = {};
    this.specialTokenIds = new Set<number>();
    for (const token of data.added_tokens) {
      if (token.special || !(token.content in data.model.vocab)) {
        specialTokensEncoder[token.content] = token.id;
      }
      if (token.special) this.specialTokenIds.add(token.id);
    }

    // Determine BOS / EOS / pad from added_tokens (with sensible defaults).
    this.bosToken = data.added_tokens.find(
      (t) =>
        t.content === "<s>" || t.content === "<bos>" ||
        t.content === "<|begin_of_text|>",
    )?.id ?? 1;
    this.eosToken = data.added_tokens.find(
      (t) =>
        t.content === "</s>" || t.content === "<eos>" ||
        t.content === "<|end_of_text|>",
    )?.id ?? 2;
    this.padToken = data.added_tokens.find(
      (t) => t.content === "<pad>" || t.content === "<|pad|>",
    )?.id ?? 0;

    if (this.#isSentencePieceStyle) {
      // ── SentencePiece-style: use custom Unicode BPE encoder ────────
      this.#encoding = null;

      // Build vocab without special tokens.
      const vocab: Record<string, number> = {};
      for (const [piece, id] of Object.entries(data.model.vocab)) {
        if (addedTokensMap.get(id)?.special) continue;
        vocab[piece] = id;
      }

      this.#spEncoder = new SentencePieceBpeEncoder(
        vocab,
        data.model.merges,
        new Map(Object.entries(specialTokensEncoder)),
        data.model.byte_fallback ?? false,
        data.model.unk_token ?? "<unk>",
      );
    } else {
      // ── Byte-level style: use BpeEncoding ──────────────────────────
      this.#spEncoder = null;

      const encoder = new Map<string, number>();
      const byteDecoder = createByteDecoder();
      for (const [piece, id] of Object.entries(data.model.vocab)) {
        if (addedTokensMap.get(id)?.special) continue;
        try {
          encoder.set(decodeByteLevelPiece(piece, byteDecoder), id);
        } catch {
          // Skip pieces that can't be byte-level decoded.
          encoder.set(bytesToHex(new TextEncoder().encode(piece)), id);
        }
      }

      // Determine regex pattern from pre-tokenizer.
      const split = data.pre_tokenizer?.type === "Sequence"
        ? data.pre_tokenizer.pretokenizers?.[0]
        : data.pre_tokenizer;
      let regex: RegExp;
      if (split?.type === "Split" && split.pattern?.Regex) {
        const pattern = split.pattern.Regex.replace(
          /^\(\?i:([^)]*)\)/,
          "(?:$1)",
        );
        regex = new RegExp(pattern, "giu");
      } else {
        regex = /\S+/gu;
      }

      this.#encoding = new tokenizers.BpeEncoding(
        encoder,
        specialTokensEncoder,
        regex,
      );
    }
  }

  static fromBinary(data: Uint8Array): HuggingFaceBpeTokenizer {
    const parsed = JSON.parse(new TextDecoder().decode(data));
    return new HuggingFaceBpeTokenizer(parsed);
  }

  encode(text: string): number[] {
    if (this.#isSentencePieceStyle) {
      // Normalize: prepend ▁, replace spaces with ▁.
      text = "▁" + text.replace(/ /g, "▁");
      return this.#spEncoder!.encode(text);
    }
    return this.#encoding!.encodeWithSpecialTokens(text);
  }

  decode(tokens: number[]): string {
    if (this.#isSentencePieceStyle) {
      let text = this.#spEncoder!.decode(tokens);
      // Strip the leading space from the prepended ▁.
      if (text.startsWith(" ")) text = text.slice(1);
      return text;
    }
    return this.#encoding!.decode(tokens);
  }
}
