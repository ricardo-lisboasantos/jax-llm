/**
 * @module runtime/runtime
 *
 * `ModelRuntime` — the low-level engine for JAX-JS model lifecycle management.
 *
 * Responsibilities:
 *  - Device initialization (WebGPU / WASM)
 *  - Tokenizer loading with multi-format fallback
 *  - Weight loading and checkpoint hydration onto the active device
 *  - JIT-compiled inference session creation (prefill / step / dispose)
 *  - Memory management via reference counting and explicit disposal
 *  - Training support via optax (see `TrainingRunner`)
 *
 * This layer is deliberately unaware of chat semantics (prompts, sampling,
 * conversation history). The high-level `ChatEngine` builds on top of it.
 */

import { defaultDevice, init } from "npm:@jax-js/jax@^0.1.25";
import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { tokenizers } from "npm:@jax-js/loaders@^0.1.3";

import { resolveModel, tokenizerUrlCandidates } from "./registry.ts";
import type {
  InferenceSession,
  LoadedModel,
  ModelDefinition,
  RuntimeConfig,
  TokenizerInterface,
} from "./types.ts";

/**
 * Low-level JAX-JS model lifecycle manager: device init, tokenizer and
 * weight loading, inference sessions, and disposal.
 */
export class ModelRuntime {
  /** The resolved model definition. */
  readonly definition: ModelDefinition;

  private config: RuntimeConfig;
  private tokenizer?: TokenizerInterface;
  private model?: LoadedModel;
  private initialized = false;

  /** Create a runtime for a model ID with an optional runtime config. */
  constructor(
    modelId: string,
    config: RuntimeConfig = {
      backend: "webgpu",
      dtype: "float32" as np.DType,
    },
  ) {
    this.definition = resolveModel(modelId, {
      weightOverrides: config.weightOverrides,
      tokenizerOverrides: config.tokenizerOverrides,
    });
    this.config = config;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────

  /**
   * Full load sequence: device init → tokenizer → weights.
   * This is the one-call entry point for getting a model ready for inference.
   */
  async load(): Promise<void> {
    await this.initDevice();
    await Promise.all([
      this.loadTokenizer(),
      this.loadWeights(),
    ]);
  }

  /** Initialize the JAX-JS runtime and select the compute device. */
  async initDevice(): Promise<void> {
    if (this.initialized) return;
    await init();
    defaultDevice(this.config.backend);
    this.initialized = true;
  }

  /**
   * Load the tokenizer, trying multiple URL formats and tokenizer types.
   * BPE is tried first (LFM-style); SentencePiece is the fallback (Gemma-style).
   */
  async loadTokenizer(): Promise<TokenizerInterface> {
    if (this.tokenizer) return this.tokenizer;

    const def = this.definition;
    const base = this.extractBase(def.tokenizerUrl);
    const candidates = tokenizerUrlCandidates(base, def.tokenizerUrl);

    // Try each candidate URL. For each one that returns data, attempt to
    // parse it with the format-appropriate tokenizer. JSON data → BPE/
    // HuggingFace tokenizer; binary data → SentencePiece (protobuf).
    for (const url of candidates) {
      let tokenizerData: Uint8Array | null = null;
      try {
        const resp = await fetch(url);
        if (resp.ok) {
          tokenizerData = new Uint8Array(await resp.arrayBuffer());
        }
      } catch {
        // Network error — try next candidate
        continue;
      }
      if (!tokenizerData) continue;

      const isJson = tokenizerData[0] === 0x7b; // '{'

      if (isJson) {
        // JSON data → try the model's createTokenizer (BPE / HuggingFace).
        try {
          this.tokenizer = def.createTokenizer(tokenizerData);
          return this.tokenizer;
        } catch (e) {
          // BPE constructor failed — this might be a different JSON
          // tokenizer type (WordPiece, Unigram, etc.). Try next URL.
          const msg = (e as Error).message;
          console.warn?.(
            `Tokenizer at ${url} could not be parsed as BPE: ${msg}`,
          );
          continue;
        }
      } else {
        // Binary data → try SentencePiece (protobuf format).
        try {
          const sp = tokenizers.SentencePiece.fromBinary(tokenizerData);
          this.tokenizer = {
            bosToken: sp.bosToken,
            eosToken: sp.eosToken,
            encode: (text) => sp.encode(text),
            decode: (tokens) => sp.decode(tokens),
            decodeGenerated: (tokens) => sp.decode(tokens),
          };
          return this.tokenizer;
        } catch {
          // Not a valid SentencePiece file — try next URL.
          continue;
        }
      }
    }

    throw new Error(
      `Failed to load tokenizer for "${def.id}" from any known path. ` +
        `Tried: ${candidates.join(", ")}`,
    );
  }

  /**
   * Download weights and hydrate the model checkpoint onto the device.
   * The resulting `LoadedModel` can create multiple inference sessions.
   */
  async loadWeights(): Promise<LoadedModel> {
    if (this.model) return this.model;

    const def = this.definition;
    const resp = await fetch(def.weightsUrl);
    if (!resp.ok) {
      throw new Error(
        `Failed to load model weights: ${resp.status} ${resp.statusText}`,
      );
    }
    const data = new Uint8Array(await resp.arrayBuffer());
    this.model = await def.loadCheckpoint(
      data,
      this.config.dtype,
      this.config.backend,
    );
    return this.model;
  }

  // ── Accessors ──────────────────────────────────────────────────────────

  /** The loaded tokenizer. Throws if not yet loaded. */
  getTokenizer(): TokenizerInterface {
    if (!this.tokenizer) {
      throw new Error("Tokenizer not loaded. Call `load()` first.");
    }
    return this.tokenizer;
  }

  /** The loaded model. Throws if not yet loaded. */
  getModel(): LoadedModel {
    if (!this.model) {
      throw new Error("Model not loaded. Call `load()` first.");
    }
    return this.model;
  }

  /** Whether the runtime has been fully loaded. */
  get isLoaded(): boolean {
    return this.initialized && !!this.tokenizer && !!this.model;
  }

  // ── Inference ──────────────────────────────────────────────────────────

  /**
   * Create a new inference session for a single conversation turn.
   * Each session owns its own KV-cache state and must be disposed when done.
   */
  createSession(): InferenceSession {
    return this.getModel().createSession();
  }

  // ── Cleanup ────────────────────────────────────────────────────────────

  /** Release all device memory held by the model weights. */
  dispose(): void {
    this.model?.dispose();
    this.model = undefined;
    this.tokenizer = undefined;
  }

  // ── Internal ───────────────────────────────────────────────────────────

  /** Extract the base URL (directory) from a tokenizer or weights URL. */
  private extractBase(url: string): string | undefined {
    const idx = url.lastIndexOf("/");
    if (idx < 0) return undefined;
    // Only strip the filename if it looks like a file (has an extension or
    // is a known tokenizer filename).
    const filename = url.slice(idx + 1);
    if (
      filename.includes(".") ||
      filename === "tokenizer" ||
      filename === "model"
    ) {
      return url.slice(0, idx);
    }
    return undefined;
  }
}
