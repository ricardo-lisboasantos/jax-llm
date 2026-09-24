/**
 * @module runtime/types
 *
 * Low-level type definitions for the JAX-JS model runtime layer.
 * These types are concerned with device management, tokenization,
 * weight loading, inference sessions, and training — not chat semantics.
 */

// Type-only import — np.DType is used only in type positions.
import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import type { safetensors } from "npm:@jax-js/loaders@^0.1.3";

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
 * Async to allow weight paging (e.g. MoE expert on-demand hydration);
 * fully device-resident models resolve immediately.
 */
export type InferenceSession = {
  /** Process the full prompt at once; returns logits for the last token. */
  prefill(tokenIds: np.Array): Promise<np.Array>;
  /** Process a single generated token; returns next-token logits. */
  step(token: number): Promise<np.Array>;
  /**
   * Score k drafted tokens in one batched forward (speculative
   * verification). Returns per-token logits `[k, vocab]`. Optional —
   * present only when the model supports draft scoring (LFM).
   */
  scoreTokens?(draftIds: np.Array): Promise<np.Array>;
  /** Keep scored state after full draft acceptance. Requires scoreTokens. */
  confirmDraft?(): void;
  /** Roll back to the pre-score state after draft rejection. */
  truncateDraft?(): void;
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
 * Incremental sharded-load handle (B2): consume one parsed shard at a
 * time so peak memory stays near a single shard instead of the full
 * checkpoint. Mirrors `PagedModelStaging` in the model layer.
 */
export type PagedLoadHandle = {
  stageShard(file: safetensors.File): Promise<void>;
  finish(device: string): Promise<LoadedModel>;
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

  // Phase 3.0: Optional quantization support
  quantizationEnabled?: boolean;
  quantizationBits?: number;

  // --- Runtime concerns ---
  createTokenizer(data: Uint8Array): TokenizerInterface;
  loadCheckpoint(
    data: Uint8Array<ArrayBuffer>,
    dtype: np.DType,
    device: string,
  ): Promise<LoadedModel>;
  /**
   * Load from pre-parsed shard files (multi-shard `model.safetensors.index.json`
   * checkpoints). Optional — models without sharded variants omit it and the
   * runtime falls back to single-file `loadCheckpoint`.
   */
  loadCheckpointFromFiles?(
    files: safetensors.File[],
    dtype: np.DType,
    device: string,
  ): Promise<LoadedModel>;
  /**
   * Incremental sharded load (B2): one shard at a time, peak ≈ 1 shard.
   * Optional — falls back to `loadCheckpointFromFiles` / `loadCheckpoint`.
   */
  beginPagedLoad?(dtype: np.DType): Promise<PagedLoadHandle>;

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

/** Default {@linkcode RuntimeConfig} using the WebGPU backend and float32 weights. */
export const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  backend: "webgpu",
  dtype: "float32" as np.DType,
};
