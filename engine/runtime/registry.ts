/**
 * @module runtime/registry
 *
 * Model registry: resolves a model name (built-in ID, short alias, or
 * HuggingFace repo identifier) into a full `ModelDefinition`.
 *
 * Built-in models are registered at module load time.
 * Dynamic HuggingFace repos are resolved on demand with sensible defaults.
 */

import { HuggingFaceBpeTokenizer } from "../tokenizer/tokenizer.ts";
import { CHAT_MODELS, type ChatModel, type ChatModelId } from "../llm/model.ts";
import type { ModelDefinition } from "./types.ts";

/** Short-name aliases mapped to full model IDs. */
const ALIASES: Record<string, ChatModelId> = {
  gemma: "gemma-3-270m",
  lfm: "lfm2.5-350m",
  "lfm2": "lfm2.5-350m",
  "lfm2.5": "lfm2.5-350m",
  qwen: "qwen2.5-0.5b",
  "qwen2": "qwen2.5-0.5b",
  "qwen2.5": "qwen2.5-0.5b",
  bonsai: "bonsai",
  gpt: "gpt2",
  "gpt2": "gpt2",
  phi: "phi-2",
  "phi2": "phi-2",
  maple: "maple-preview",
};

/** All registered built-in model IDs. */
export const MODEL_IDS = Object.keys(CHAT_MODELS) as ChatModelId[];

/** The default model used when none is specified. */
export const DEFAULT_MODEL_ID: ChatModelId = "lfm2.5-350m";

/** Type guard: is this a registered built-in model ID? */
export function isBuiltInModel(modelId: string): modelId is ChatModelId {
  return modelId in CHAT_MODELS;
}

/** Type guard: is this a known short alias? */
export function isAlias(modelId: string): modelId is keyof typeof ALIASES {
  return modelId in ALIASES;
}

/** Type guard: does this look like a HuggingFace repo identifier? */
export function isHuggingFaceRepo(modelId: string): boolean {
  return modelId.includes("/") && !modelId.includes("://");
}

/**
 * Resolve a model identifier into a full `ModelDefinition`.
 *
 * Resolution order:
 *  1. Built-in model IDs (e.g. `"lfm2.5-350m"`)
 *  2. Short aliases (e.g. `"gemma"`, `"lfm"`)
 *  3. HuggingFace repo IDs (e.g. `"deepgrove/Bonsai"`) — resolved dynamically
 *
 * @param modelId The model name to resolve.
 * @param overrides Optional URL overrides (weights, tokenizer).
 * @returns A `ModelDefinition` ready for loading.
 */
export function resolveModel(
  modelId: string,
  overrides?: {
    weightOverrides?: Record<string, string>;
    tokenizerOverrides?: Record<string, string>;
  },
): ModelDefinition {
  // 1. Built-in
  if (isBuiltInModel(modelId)) {
    const model = CHAT_MODELS[modelId] as ChatModel;
    return applyOverrides(model, modelId, overrides);
  }

  // 2. Alias
  if (isAlias(modelId)) {
    const resolved = ALIASES[modelId];
    const model = CHAT_MODELS[resolved] as ChatModel;
    return applyOverrides(model, resolved, overrides);
  }

  // 3. Dynamic HuggingFace repo
  if (isHuggingFaceRepo(modelId)) {
    return resolveHuggingFaceRepo(modelId, overrides);
  }

  throw new Error(
    `Unknown model "${modelId}". Use a built-in ID (${
      MODEL_IDS.join(", ")
    }), ` +
      `a short alias (${Object.keys(ALIASES).join(", ")}), or a ` +
      `HuggingFace repo identifier like "org/model-name".`,
  );
}

/** Apply URL overrides to a built-in model definition. */
function applyOverrides(
  model: ChatModel,
  modelId: string,
  overrides?: {
    weightOverrides?: Record<string, string>;
    tokenizerOverrides?: Record<string, string>;
  },
): ModelDefinition {
  const weightsUrl = overrides?.weightOverrides?.[modelId] ?? model.weightsUrl;
  const tokenizerUrl = overrides?.tokenizerOverrides?.[modelId] ??
    model.tokenizerUrl;
  return { ...model, weightsUrl, tokenizerUrl };
}

/**
 * Build a `ModelDefinition` for a dynamic HuggingFace repo.
 *
 * Uses common conventions for URL construction and falls back to the LFM
 * implementation for checkpoint loading (since it's the most generic
 * architecture supported). Tokenizer loading tries multiple formats.
 */
function resolveHuggingFaceRepo(
  repoId: string,
  overrides?: {
    weightOverrides?: Record<string, string>;
    tokenizerOverrides?: Record<string, string>;
  },
): ModelDefinition {
  const base = `https://huggingface.co/${repoId}/resolve/main`;
  const weightsUrl = overrides?.weightOverrides?.[repoId] ??
    `${base}/model.safetensors`;
  const tokenizerUrl = overrides?.tokenizerOverrides?.[repoId] ??
    `${base}/tokenizer.json`;

  // Reuse the LFM checkpoint loader as a generic fallback.
  const lfmDef = CHAT_MODELS["lfm2.5-350m"] as ChatModel;

  return {
    id: repoId,
    label: repoId,
    downloadSize: "unknown",
    weightsUrl,
    tokenizerUrl,
    contextSize: 4096,
    defaults: {
      temperature: 0.8,
      topK: 64,
      topP: 0.95,
      repetitionPenalty: 1,
    },
    createTokenizer: (data: Uint8Array) => {
      const raw = HuggingFaceBpeTokenizer.fromBinary(data);
      return {
        bosToken: raw.bosToken,
        eosToken: raw.eosToken,
        encode: (text: string) => raw.encode(text),
        decode: (tokens: number[]) => raw.decode(tokens),
        decodeGenerated: (tokens: number[]) =>
          raw.decode(tokens.filter((t) => !raw.specialTokenIds.has(t))),
      };
    },
    formatPrompt: (history) => {
      // Generic ChatML-style prompt.
      let text = "";
      for (const msg of history) {
        const content = msg.content.trim();
        if (content === "") continue;
        text += `<|im_start|>${msg.role}\n${content}<|im_end|>\n`;
      }
      return `${text}<|im_start|>assistant\n`;
    },
    encodePrompt: (tokenizer, history) => {
      // Use the formatPrompt defined above — no recursive resolution needed.
      const text = history.map((msg) => {
        const content = msg.content.trim();
        if (content === "") return "";
        return `<|im_start|>${msg.role}\n${content}<|im_end|>\n`;
      }).join("");
      const prompt = `${text}<|im_start|>assistant\n`;
      return [tokenizer.bosToken, ...tokenizer.encode(prompt)];
    },
    stopTokens: (tokenizer) => [tokenizer.eosToken],
    loadCheckpoint: async (data, dtype, device) => {
      // Parse with BF16 support first, then delegate to the LFM2.5 loader
      // as a generic fallback. Note: this will only work if the model has
      // an LFM2.5-compatible architecture. For other architectures (e.g.
      // Llama, Mistral), a dedicated model implementation is needed.
      try {
        return await lfmDef.loadCheckpoint(data, dtype, device);
      } catch (e) {
        const msg = (e as Error).message;
        if (
          msg.includes("layers") || msg.includes("Expected") ||
          msg.includes("undefined")
        ) {
          throw new Error(
            `Model "${repoId}" has an architecture that is not compatible ` +
              `with the built-in LFM2.5 checkpoint loader. ` +
              `A dedicated model implementation is needed for this architecture. ` +
              `Original error: ${msg}`,
          );
        }
        throw e;
      }
    },
  };
}

/**
 * Candidate tokenizer URLs to try when loading a model.
 * Different model families store the tokenizer under different filenames.
 */
export function tokenizerUrlCandidates(
  base: string | undefined,
  primaryUrl: string,
): string[] {
  if (!base) return [primaryUrl];
  return [
    primaryUrl,
    `${base}/tokenizer.json`,
    `${base}/tokenizer.model`,
    `${base}/tokenizer.spm`,
  ];
}
