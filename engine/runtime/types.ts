/**
 * @module runtime/types
 *
 * Low-level type definitions for the JAX-JS model runtime layer.
 * These types are concerned with device management, tokenization,
 * weight loading, inference sessions, and training — not chat semantics.
 */

// Type-only import — np.DType is used only in type positions.
import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";

/** Compute backend selection. */
export type Backend = "webgpu" | "wasm";

/** Sampling parameters used as per-model defaults. */
export type SamplingDefaults = {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
};

/**
 * Tokenizer interface shared by all model families.
 * `decodeGenerated` filters control/special tokens before decoding,
 * hiding model-specific output quirks from the caller.
 */
export type TokenizerInterface = {
  bosToken: number;
  eosToken: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  decodeGenerated(tokens: number[]): string;
};

/**
 * A single prefill/decode inference sequence.
 * Created per-conversation (or per-turn) and disposed when done.
 */
export type InferenceSession = {
  /** Process the full prompt at once; returns logits for the last token. */
  prefill(tokenIds: np.Array): np.Array;
  /** Process a single generated token; returns next-token logits. */
  step(token: number): np.Array;
  /** Release KV-cache and intermediate buffers. */
  dispose(): void;
};

/** A checkpoint loaded onto the active device, ready to create sessions. */
export type LoadedModel = {
  modelId: string;
  device: string;
  createSession(): InferenceSession;
  dispose(): void;
};

/**
 * The full specification for a model family, combining runtime concerns
 * (loading, tokenization, inference) with chat-level prompt formatting.
 * The runtime layer only accesses the runtime methods; the chat layer
 * accesses the prompt/sampling methods.
 */
export type ModelDefinition = {
  id: string;
  label: string;
  downloadSize: string;
  weightsUrl: string;
  tokenizerUrl: string;
  contextSize: number;
  defaults: SamplingDefaults;

  // --- Runtime concerns ---
  createTokenizer(data: Uint8Array): TokenizerInterface;
  loadCheckpoint(
    data: Uint8Array<ArrayBuffer>,
    dtype: np.DType,
    device: string,
  ): Promise<LoadedModel>;

  // --- Chat concerns ---
  formatPrompt(history: import("../chat/types.ts").ChatMessage[]): string;
  encodePrompt(
    tokenizer: TokenizerInterface,
    history: import("../chat/types.ts").ChatMessage[],
  ): number[];
  stopTokens(tokenizer: TokenizerInterface): number[];
};

/** Configuration for the runtime layer. */
export type RuntimeConfig = {
  backend: Backend;
  dtype: np.DType;
  /** Override weight URLs per model ID (e.g. for local mirrors). */
  weightOverrides?: Record<string, string>;
  /** Override tokenizer URLs per model ID. */
  tokenizerOverrides?: Record<string, string>;
};

export const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  backend: "webgpu",
  dtype: "float32" as np.DType,
};
