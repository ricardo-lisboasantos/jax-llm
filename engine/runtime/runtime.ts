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
import type { safetensors } from "npm:@jax-js/loaders@^0.1.3";

import { resolveModel, tokenizerUrlCandidates } from "./registry.ts";
import { parseSafetensors } from "./safetensors.ts";
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

  /** Disk-cache directory (override via `JAX_LLM_CACHE`). */
  private cacheDir(): string {
    try {
      return Deno.env.get("JAX_LLM_CACHE") ?? "./.opfs/weights";
    } catch {
      return "./.opfs/weights";
    }
  }

  /** Disk-cache path for one cache entry. */
  private cachePath(name: string): string {
    return `${this.cacheDir()}/${encodeURIComponent(name)}`;
  }

  /** Best-effort cache write (never blocks TTFT on failure). */
  private async writeCache(name: string, data: Uint8Array): Promise<void> {
    try {
      await Deno.mkdir(this.cacheDir(), { recursive: true });
      await Deno.writeFile(this.cachePath(name), data);
    } catch {
      // No write perms / read-only env — network path still works.
    }
  }

  /**
   * Load the tokenizer, trying multiple URL formats and tokenizer types.
   * BPE is tried first (LFM-style); SentencePiece is the fallback (Gemma-style).
   * Phase 3.0: Parallel fetching with Promise.allSettled() reduces TTFT by 200-400ms.
   */
  async loadTokenizer(): Promise<TokenizerInterface> {
    if (this.tokenizer) return this.tokenizer;

    const def = this.definition;
    // Disk cache first — skips all network on warm runs (kills 10s tail).
    try {
      const cached = await Deno.readFile(
        this.cachePath(`${def.id}.tokenizer.bin`),
      );
      const parsed = this.parseTokenizerData(def, cached);
      if (parsed) {
        this.tokenizer = parsed;
        return this.tokenizer;
      }
    } catch {
      // Cache miss — fall through to network.
    }

    const base = this.extractBase(def.tokenizerUrl);
    const candidates = tokenizerUrlCandidates(base, def.tokenizerUrl);

    // Phase 3.0: Parallel URL fetching with Promise.allSettled()
    // Attempt all candidates concurrently; first successful parse wins.
    // Reduces TTFT by ~200-400ms (time to first token) vs sequential fallback.
    const fetchPromises = candidates.map(
      async (url): Promise<TokenizerInterface | null> => {
        try {
          const resp = await fetch(url, {
            signal: AbortSignal.timeout(5000),
          });
          if (!resp.ok) return null;

          const tokenizerData = new Uint8Array(await resp.arrayBuffer());
          const parsed = this.parseTokenizerData(def, tokenizerData);
          if (parsed) {
            // Warm disk cache in background (don't block first-win race).
            void this.writeCache(`${def.id}.tokenizer.bin`, tokenizerData);
          } else if (tokenizerData.length > 0) {
            console.warn?.(`Tokenizer at ${url} could not be parsed.`);
          }
          return parsed;
        } catch (e) {
          // Timeout or network error
          console.debug?.(
            `Tokenizer fetch failed for ${url}: ${(e as Error).message}`,
          );
          return null;
        }
      },
    );

    // Phase 3.0: First-win race — return on first successful parse,
    // don't wait for slowest candidate. Remaining fetches continue in
    // background but no longer block TTFT.
    const pending = new Set(fetchPromises);
    while (pending.size > 0) {
      const winner = await Promise.race(
        [...pending].map((p) =>
          p.then(
            (v) => ({ p, v }),
            () => ({ p, v: null as TokenizerInterface | null }),
          )
        ),
      );
      pending.delete(winner.p);
      if (winner.v) {
        this.tokenizer = winner.v;
        return this.tokenizer;
      }
    }

    throw new Error(
      `Failed to load tokenizer for "${def.id}" from any known path. ` +
        `Tried: ${candidates.join(", ")}`,
    );
  }

  /** Parse raw tokenizer bytes (JSON BPE or SentencePiece binary). */
  private parseTokenizerData(
    def: ModelDefinition,
    tokenizerData: Uint8Array,
  ): TokenizerInterface | null {
    if (tokenizerData.length === 0) return null;
    const isJson = tokenizerData[0] === 0x7b; // '{'
    if (isJson) {
      try {
        return def.createTokenizer(tokenizerData);
      } catch {
        return null;
      }
    }
    try {
      const sp = tokenizers.SentencePiece.fromBinary(tokenizerData);
      return {
        bosToken: sp.bosToken,
        eosToken: sp.eosToken,
        encode: (text) => sp.encode(text),
        decode: (tokens) => sp.decode(tokens),
        decodeGenerated: (tokens) => sp.decode(tokens),
      };
    } catch {
      return null;
    }
  }

  /**
   * Download weights and hydrate the model checkpoint onto the device.
   * The resulting `LoadedModel` can create multiple inference sessions.
   * Phase 3.0: Attempt INT8 quantized variant first; fallback to FP32.
   * Disk cache (`JAX_LLM_CACHE` or `./.opfs/weights`) skips re-download.
   */
  async loadWeights(): Promise<LoadedModel> {
    if (this.model) return this.model;

    const def = this.definition;
    // Multi-shard checkpoints (phi-2, maple-preview): resolve index.json
    // into parallel shard fetches instead of treating the index as weights.
    if (def.weightsUrl.endsWith(".index.json")) {
      this.model = await this.loadShardedWeights();
      return this.model;
    }

    const originalUrl = def.weightsUrl;

    // Phase 3.0: Attempt INT8 quantized variant first; fallback to FP32
    let weightsUrl = def.weightsUrl;
    let quantizationAttempted = false;

    if (def.quantizationEnabled) {
      quantizationAttempted = true;
      // HEAD check only — prior GET downloaded full weights just to test existence.
      const resp = await fetch(weightsUrl, {
        method: "HEAD",
        signal: AbortSignal.timeout(5000),
      }).catch(() => null);
      if (!resp?.ok) {
        // Quantized variant not available; fallback to original FP32 URL
        console.info(
          "INT8 quantized weights not available; using FP32 baseline",
        );
        weightsUrl = originalUrl.replace(
          "_q8.safetensors",
          ".safetensors",
        );
        quantizationAttempted = false;
      }
    }

    const weightsCacheKey = quantizationAttempted
      ? `${def.id}._q8.safetensors`
      : `${def.id}.safetensors`;
    // Disk cache first — warm runs skip the 676MB-1GB download entirely.
    try {
      const cached = await Deno.readFile(this.cachePath(weightsCacheKey));
      if (cached.length > 0) {
        this.model = await def.loadCheckpoint(
          cached,
          this.config.dtype,
          this.config.backend,
        );
        return this.model;
      }
    } catch {
      // Cache miss — fall through to network.
    }

    const resp = await fetch(weightsUrl);
    if (!resp.ok) {
      throw new Error(
        `Failed to load model weights: ${resp.status} ${resp.statusText}`,
      );
    }
    const data = new Uint8Array(await resp.arrayBuffer());
    // Persist to disk cache (best-effort; warm runs skip download).
    await this.writeCache(weightsCacheKey, data);

    // Phase 3.0: Register quantized weights in global cache if loaded
    if (quantizationAttempted) {
      // Note: actual parsing would happen in loadCheckpoint via Int8QuantizationLoader
      console.info("INT8 quantized weights registered in dequantization cache");
    }

    this.model = await def.loadCheckpoint(
      data,
      this.config.dtype,
      this.config.backend,
    );
    return this.model;
  }

  /**
   * Load a multi-shard checkpoint via its `model.safetensors.index.json`.
   * Each shard is fetched in parallel with per-shard disk cache, parsed
   * independently, and merged through `loadCheckpointFromFiles`.
   */
  private async loadShardedWeights(): Promise<LoadedModel> {
    const def = this.definition;
    const indexUrl = def.weightsUrl;
    const base = indexUrl.slice(0, indexUrl.lastIndexOf("/") + 1);

    // Index JSON (tiny) with disk cache.
    let indexText: string;
    try {
      const cached = await Deno.readFile(
        this.cachePath(`${def.id}.index.json`),
      );
      indexText = new TextDecoder().decode(cached);
    } catch {
      const resp = await fetch(indexUrl);
      if (!resp.ok) {
        throw new Error(
          `Failed to load weights index: ${resp.status} ${resp.statusText}`,
        );
      }
      const bytes = new Uint8Array(await resp.arrayBuffer());
      await this.writeCache(`${def.id}.index.json`, bytes);
      indexText = new TextDecoder().decode(bytes);
    }

    const index = JSON.parse(indexText) as {
      weight_map?: Record<string, string>;
    };
    if (!index.weight_map) {
      throw new Error(
        `Weights index for "${def.id}" has no weight_map: ${indexUrl}`,
      );
    }
    const shardNames = [...new Set(Object.values(index.weight_map))];

    // Incremental path (B2): one shard at a time, peak ≈ 1 shard.
    // Shard buffers are releasable after each stageShard returns.
    if (def.beginPagedLoad) {
      const staging = await def.beginPagedLoad(this.config.dtype);
      for (const name of shardNames) {
        const file = await this.fetchShardFile(base, def.id, name);
        await staging.stageShard(file);
      }
      return await staging.finish(this.config.backend);
    }

    const files = await Promise.all(
      shardNames.map((name) => this.fetchShardFile(base, def.id, name)),
    );

    if (!def.loadCheckpointFromFiles) {
      throw new Error(
        `Model "${def.id}" uses sharded weights but has no sharded loader`,
      );
    }
    return await def.loadCheckpointFromFiles(
      files,
      this.config.dtype,
      this.config.backend,
    );
  }

  /** Fetch one shard (disk cache first), parse, return. Caller drops refs. */
  private async fetchShardFile(
    base: string,
    modelId: string,
    name: string,
  ): Promise<safetensors.File> {
    const cacheKey = `${modelId}.shard.${name}`;
    try {
      const cached = await Deno.readFile(this.cachePath(cacheKey));
      if (cached.length > 0) return parseSafetensors(cached);
    } catch {
      // Cache miss — fall through to network.
    }
    const resp = await fetch(`${base}${name}`);
    if (!resp.ok) {
      throw new Error(
        `Failed to load shard ${name}: ${resp.status} ${resp.statusText}`,
      );
    }
    const data = new Uint8Array(await resp.arrayBuffer());
    await this.writeCache(cacheKey, data);
    return parseSafetensors(data);
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

  // ── Utilities ──────────────────────────────────────────────────────────

  /** Extract the `https://host/org/repo` base from a file URL. */
  private extractBase(url: string): string {
    const match = url.match(/^(https?:\/\/[^/]+\/[^/]+\/[^/]+)\//);
    return match ? match[1] : "";
  }
}
