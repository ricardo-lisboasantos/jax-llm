/**
 * @module chat/types
 *
 * High-level type definitions for the chat layer.
 * These are the only types a consumer of `ChatEngine` needs to know about.
 */

import type { Backend, SamplingDefaults } from "../runtime/types.ts";

/** A single message in a conversation. */
export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

/** Options for constructing a `ChatEngine`. */
export type ChatEngineOptions = {
  /** Compute backend. Defaults to `"webgpu"`. */
  backend?: Backend;
  /** Maximum tokens to generate per response. */
  maxTokens?: number;
  /** Override sampling defaults. */
  sampling?: Partial<SamplingDefaults>;
  /** Override weight URLs per model ID. */
  weightOverrides?: Record<string, string>;
  /** Override tokenizer URLs per model ID. */
  tokenizerOverrides?: Record<string, string>;
};

/** Runtime status reported by `ChatEngine.getSystemInfo()`. */
export type SystemInfo = {
  device: string;
  model: string;
  contextUsage: number;
  contextSize: number;
};
