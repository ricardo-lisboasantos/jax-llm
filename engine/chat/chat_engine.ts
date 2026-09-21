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
import type { ChatEngineOptions, ChatMessage, SystemInfo } from "./types.ts";

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
      logits = session.prefill(inputIds);

      for (let i = 0; i < this.maxTokens; i++) {
        const nextToken = await this.sampleNextToken(
          logits,
          [...promptTokens, ...generatedTokens],
        );

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
    const data = await logits.data() as Float32Array;
    return sampleLogits(data, {
      temperature: this.resolvedSampling!.temperature,
      topK: this.resolvedSampling!.topK,
      topP: this.resolvedSampling!.topP,
      repetitionPenalty: this.resolvedSampling!.repetitionPenalty,
      previousTokens,
    });
  }
}
