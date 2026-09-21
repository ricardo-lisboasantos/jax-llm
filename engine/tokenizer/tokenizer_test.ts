/**
 * unit: tokenizer (HuggingFace BPE + lightweight encoder/decoder).
 */
import {
  assert,
  assertEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { HuggingFaceBpeTokenizer } from "./tokenizer.ts";
import { TokenizerEncoder } from "./encoder.ts";
import { TokenizerDecoder } from "./decoder.ts";

function byteLevelData() {
  return {
    added_tokens: [
      { id: 0, content: "<s>", special: true },
      { id: 1, content: "</s>", special: true },
    ],
    pre_tokenizer: { type: "ByteLevel" },
    model: {
      type: "BPE",
      vocab: { hello: 10, world: 11, "!": 12 },
      merges: [["h", "ello"], ["w", "orld"]],
    },
  };
}

function sentencePieceData() {
  return {
    added_tokens: [
      { id: 0, content: "<s>", special: true },
      { id: 1, content: "</s>", special: true },
      { id: 2, content: "<unk>", special: true },
    ],
    pre_tokenizer: null,
    model: {
      type: "BPE",
      vocab: { "▁hello": 10, "▁world": 11 },
      merges: [],
    },
  };
}

Deno.test("unit: tokenizer byte-level encode/decode", () => {
  // deno-lint-ignore no-explicit-any
  const tok = new HuggingFaceBpeTokenizer(byteLevelData() as any);
  assertEquals(tok.bosToken, 0);
  assertEquals(tok.eosToken, 1);
  assert(tok.specialTokenIds.has(0));
  const ids = tok.encode("hello");
  assert(Array.isArray(ids));
  const text = tok.decode(ids);
  assertStringIncludes(text, "hello");
});

Deno.test("unit: tokenizer fromBinary parses JSON", () => {
  const bytes = new TextEncoder().encode(JSON.stringify(byteLevelData()));
  const tok = HuggingFaceBpeTokenizer.fromBinary(bytes);
  assertEquals(tok.bosToken, 0);
});

Deno.test("unit: tokenizer sentencepiece-style encode/decode", () => {
  // deno-lint-ignore no-explicit-any
  const tok = new HuggingFaceBpeTokenizer(sentencePieceData() as any);
  assertEquals(tok.bosToken, 0);
  const ids = tok.encode("hello world");
  assert(ids.length > 0);
  const text = tok.decode(ids);
  assertStringIncludes(text, "hello");
});

Deno.test("unit: tokenizer rejects non-BPE model", () => {
  assertThrows(
    // deno-lint-ignore no-explicit-any
    () => new HuggingFaceBpeTokenizer({ model: { type: "WordPiece" } } as any),
    Error,
    "BPE",
  );
});

Deno.test("unit: TokenizerEncoder greedy longest-match", () => {
  const enc = new TokenizerEncoder({ ab: 1, a: 2, b: 3 });
  assertEquals(enc.encode("ab"), [1]);
  assertEquals(enc.encode(""), []);
  assertEquals(enc.size, 3);
  // Unknown chars fall back per-character (deterministic 0s).
  assertEquals(enc.encode("<s>"), [0, 0, 0]);
});

Deno.test("unit: TokenizerEncoder special-token fast path", () => {
  const enc = new TokenizerEncoder({}, { "<s>": 99 });
  assertEquals(enc.encode("<s>"), [99]);
});

Deno.test("unit: TokenizerDecoder roundtrip + generated filter", () => {
  const dec = new TokenizerDecoder({ hello: 10, world: 11 }, [0]);
  assertEquals(dec.decode([10, 11]), "helloworld");
  assertStringIncludes(dec.decode([999]), "unk:999");
  assertEquals(dec.decodeGenerated([0, 10]), "hello");
});
