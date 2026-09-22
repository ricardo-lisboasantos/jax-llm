/**
 * @module chat/chat_engine
 *
 * `ChatEngine` — the high-level chat API.
 *
 * This is the only class most consumers need to interact with:
 *
 * ```ts
 * const engine = new ChatEngine("lfm2.5-350m");
 * await engine.init();
 *
 * // Single prompt
 * const reply = await engine.generate("Explain quantum computing.");
 *
 * // Multi-turn conversation
 * const chatReply = await engine.chat([
 *   { role: "user", content: "What is JAX?" },
 *   { role: "assistant", content: "JAX is a numerical computing library." },
 *   { role: "user", content: "And how is it used here?" },
 * ]);
 * ```
 *
 * It delegates all low-level concerns (device init, tokenizer/weight
 * loading, JIT inference) to `ModelRuntime` and all sampling logic to
 * the `Sampler` module. The engine itself is a thin orchestrator that
 * manages conversation state, prompt formatting, and the generate loop.
 */

import { numpy as np } from "npm:@jax-js/jax@^0.1.25";

import { ModelRuntime } from "../runtime/runtime.ts";
import type { RuntimeConfig, SamplingDefaults } from "../runtime/types.ts";
import { resolveSamplingDefaults, sampleLogits } from "./sampler.ts";
import { globalProfiler } from "../llm/profiling/webgpu_profiler.ts";
import type { ChatEngineOptions, ChatMessage, SystemInfo } from "./types.ts";

/**
 * Performance metrics collected during chat generation.
 * Only populated when profiling is enabled.
 */
export interface ChatMetrics {
  /** Time to first token (prefill latency) in milliseconds. */
  ttftMs: number;
  /** Prefill throughput in tokens per second. */
  prefillTokPerSec: number;
  /** Decode throughput in tokens per second. */
  decodeTokPerSec: number;
  /** 50th percentile decode step latency in milliseconds. */
  p50JitterMs: number;
  /** 90th percentile decode step latency in milliseconds. */
  p90JitterMs: number;
}

/**
 * High-level chat API: pass a model name, call `init()`, then `chat()`.
 * Delegates device/weight management to `ModelRuntime` and token
 * selection to the sampler module.
 */
export class ChatEngine {
  private runtime: ModelRuntime;
  private maxTokens: number;
  private samplingOverrides?: Partial<SamplingDefaults>;
  private resolvedSampling?: SamplingDefaults;
  // Phase 3.0: Performance monitoring
  private profilingEnabled = false;
  private metrics: ChatMetrics = {
    ttftMs: 0,
    prefillTokPerSec: 0,
    decodeTokPerSec: 0,
    p50JitterMs: 0,
    p90JitterMs: 0,
  };

  /**
   * Create a chat engine for a given model.
   *
   * @param model Model name — a built-in ID (`"lfm2.5-350m"`), a short
   *              alias (`"lfm"`, `"gemma"`), or a HuggingFace repo ID
   *              (`"org/model-name"`).
   * @param options Optional configuration (backend, max tokens, sampling).
   *
   * @example
   * ```ts
   * // Simplest form
   * const engine = new ChatEngine("lfm2.5-350m");
   * // With options
   * const engine = new ChatEngine("gemma", { backend: "wasm", maxTokens: 512 });
   * ```
   */
  constructor(model: string, options?: ChatEngineOptions) {
    const runtimeConfig: RuntimeConfig = {
      backend: options?.backend ?? "webgpu",
      dtype: "float32" as np.DType,
      weightOverrides: options?.weightOverrides,
      tokenizerOverrides: options?.tokenizerOverrides,
    };
    this.runtime = new ModelRuntime(model, runtimeConfig);
    this.maxTokens = options?.maxTokens ?? 4096;
    this.samplingOverrides = options?.sampling;
  }

  /**
   * Load the model: initialize the device, fetch the tokenizer, and
   * hydrate the model weights. Must be called before `chat()`.
   */
  async init(): Promise<void> {
    await this.runtime.load();
    this.resolvedSampling = resolveSamplingDefaults(
      this.runtime.definition.defaults,
      this.samplingOverrides,
    );
  }

  /**
   * Enable performance profiling (Phase 3.0).
   * Tracks TTFT, tok/s, and latency percentiles.
   * Call before chat() or generate() to collect metrics.
   *
   * @example
   * ```ts
   * engine.enableProfiling();
   * await engine.chat(history);
   * const metrics = engine.getMetrics();
   * console.log(`TTFT: ${metrics.ttftMs}ms, Decode: ${metrics.decodeTokPerSec} tok/s`);
   * ```
   */
  enableProfiling(): void {
    this.profilingEnabled = true;
    globalProfiler.reset();
  }

  /**
   * Get current performance metrics.
   * Returns null if profiling was not enabled before the last generation.
   *
   * @returns Metrics object with TTFT, throughput, and latency percentiles,
   *          or null if profiling disabled.
   */
  getMetrics(): ChatMetrics | null {
    if (!this.profilingEnabled) {
      return null;
    }
    const stats = globalProfiler.getAllStats();
    const result = { ...this.metrics };

    // TTFT: time to first token (from prefill start to first decode step)
    const ttftStat = stats.get("prefill");
    if (ttftStat) {
      result.ttftMs = ttftStat.meanMs;
    }

    // Decode step latency (p50/p90 jitter)
    const decodeStat = stats.get("decode_step");
    if (decodeStat) {
      result.decodeTokPerSec = 1000 / decodeStat.meanMs;
      result.p50JitterMs = decodeStat.p50Ms;
      result.p90JitterMs = decodeStat.p90Ms;
    }

    return result;
  }

  /**
   * Send a conversation history and get the full response string.
   *
   * This is the chatbot entry point — pass the full turn history and get
   * back the complete assistant reply (no streaming). For a single prompt
   * use `generate()`; for streaming use `chatStream()`.
   *
   * @param history The conversation history (system + user + assistant turns).
   * @returns The assistant's response.
   */
  async chat(history: ChatMessage[]): Promise<string> {
    let response = "";
    for await (const chunk of this.chatStream(history)) {
      // chatStream() yields cumulative text (full reply so far), so each
      // chunk supersedes the previous one — keep only the latest.
      response = chunk;
    }
    return response;
  }

  /**
   * Generate a completion for a single prompt and get the full response.
   *
   * Convenience wrapper around `chat()` for single-shot generation without
   * conversation history.
   *
   * @param prompt The user's input text.
   * @returns The assistant's response.
   */
  generate(prompt: string): Promise<string> {
    return this.chat([{ role: "user", content: prompt }]);
  }

  /**
   * Stream a completion for a single prompt as an async generator.
   *
   * Convenience wrapper around `chatStream()` — each yield is the cumulative
   * generated text so far.
   *
   * @param prompt The user's input text.
   * @yields Cumulative generated text strings.
   */
  async *generateStream(
    prompt: string,
  ): AsyncGenerator<string, void, unknown> {
    yield* this.chatStream([{ role: "user", content: prompt }]);
  }

  /**
   * Stream generated tokens as an async generator.
   *
   * Each yield produces the full generated text so far (cumulative),
   * so the caller can diff against the previous yield to get just the
   * new delta.
   *
   * @param history The conversation history (system + user + assistant turns).
   * @yields Cumulative generated text strings.
   */
  async *chatStream(
    history: ChatMessage[],
  ): AsyncGenerator<string, void, unknown> {
    if (!this.runtime.isLoaded) {
      throw new Error("Engine not initialized. Call `init()` first.");
    }

    const def = this.runtime.definition;
    const tokenizer = this.runtime.getTokenizer();
    const promptTokens = def.encodePrompt(tokenizer, history);
    const generatedTokens: number[] = [];
    const inputIds = np.array(promptTokens, { dtype: np.uint32 });
    const session = this.runtime.createSession();
    const stopTokens = def.stopTokens(tokenizer);

    let logits: np.Array | null = null;

    try {
      // Phase 3.0: Wrap prefill with profiler if enabled
      if (this.profilingEnabled) {
        globalProfiler.start("prefill");
      }
      logits = session.prefill(inputIds);
      if (this.profilingEnabled) {
        globalProfiler.end("prefill");
      }

      for (let i = 0; i < this.maxTokens; i++) {
        // Phase 3.0: Wrap decode step with profiler if enabled
        if (this.profilingEnabled) {
          globalProfiler.start("decode_step");
        }
        const nextToken = await this.sampleNextToken(
          logits,
          [...promptTokens, ...generatedTokens],
        );
        if (this.profilingEnabled) {
          globalProfiler.end("decode_step");
        }

        if (stopTokens.includes(nextToken)) break;

        generatedTokens.push(nextToken);
        yield tokenizer.decodeGenerated(generatedTokens);

        if (i === this.maxTokens - 1) break;

        logits = session.step(nextToken);
      }
    } finally {
      session.dispose();
    }
  }

  /** Runtime status for display or monitoring. */
  getSystemInfo(): SystemInfo {
    return {
      device: this.runtime.definition.id ? "webgpu" : "unknown",
      model: this.runtime.definition.id,
      contextUsage: 0,
      contextSize: this.runtime.definition.contextSize,
    };
  }

  /** Access the underlying runtime (for advanced use cases). */
  getRuntime(): ModelRuntime {
    return this.runtime;
  }

  /** Release all resources. */
  dispose(): void {
    this.runtime.dispose();
  }

  // ── Internal ───────────────────────────────────────────────────────────

  /** Sample the next token from logits using the resolved sampling config. */
  private async sampleNextToken(
    logits: np.Array,
    previousTokens: number[],
  ): Promise<number> {
    const rawData = await logits.data();
    const data = rawData instanceof Float32Array
      ? rawData
      : new Float32Array(rawData as ArrayLike<number>);
    return sampleLogits(data, {
      temperature: this.resolvedSampling!.temperature,
      topK: this.resolvedSampling!.topK,
      topP: this.resolvedSampling!.topP,
      repetitionPenalty: this.resolvedSampling!.repetitionPenalty,
      previousTokens,
    });
  }
}
