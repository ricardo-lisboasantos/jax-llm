import type { Backend } from "./runtime/types.ts";
import { DEFAULT_MODEL_ID } from "./runtime/registry.ts";

/** Top-level library configuration. */
export interface Config {
  /** Chat engine settings: backend, model selection, and URL overrides. */
  chat: {
    backend: Backend;
    modelId: string;
    maxTokens: number;
    modelOverrides?: Record<
      string,
      { weightsUrl?: string; tokenizerUrl?: string }
    >;
  };
}

// Default configuration used when no custom config is provided.
/** Default configuration used when no custom config is provided. */
export const DEFAULT_CONFIG: Config = {
  chat: {
    backend: "webgpu",
    modelId: DEFAULT_MODEL_ID,
    maxTokens: 4096,
    modelOverrides: {},
  },
};

// ChatEngineOptions – options for constructing a ChatEngine programmatically.
// Re-exported from the chat layer for backward compatibility.
export type { ChatEngineOptions } from "./chat/types.ts";

/**
 * Load configuration from `config.json` (or the path given in
 * `JAX_JS_CONFIG_PATH`).  The file is merged onto the defaults and
 * then returned as a validated Config (no Zod validation).
 *
 * @returns a `Config` instance.
 */
export function loadConfig(): Config {
  let configPath = "config.json";
  try {
    const envPath = Deno.env.get("JAX_JS_CONFIG_PATH");
    if (envPath) configPath = envPath;
  } catch {
    // ignore – environment access not permitted in some contexts (e.g., tests)
  }

  let raw: string;
  try {
    raw = Deno.readTextFileSync(configPath);
  } catch {
    // No custom config – just return defaults.
    return DEFAULT_CONFIG;
  }

  const userConfig = JSON.parse(raw);
  // Merge user config with defaults.
  return { ...DEFAULT_CONFIG, ...userConfig } as Config;
}

/**
 * Helper used by the CLI to create a `ChatEngine` from
 * high-level options (model selection, backend selection, etc.).
 * It simply forwards to `loadConfig` and then applies any
 * additional overrides the user supplied via the CLI.
 */
export function createConfig(options: {
  model?: "gemma" | "lfm";
  modelId?: string;
  backend?: Backend;
}): Config {
  const base = loadConfig();
  let resolvedModelId = base.chat.modelId;
  if (options.modelId) {
    // Map well-known short names to built-in IDs, otherwise use as-is.
    if (options.modelId === "gemma") {
      resolvedModelId = "gemma-3-270m";
    } else if (options.modelId === "lfm") {
      resolvedModelId = "lfm2.5-350m";
    } else {
      resolvedModelId = options.modelId;
    }
  } else if (options.model) {
    // Legacy short name handling (gemma -> gemma-3-270m).
    resolvedModelId = options.model === "gemma"
      ? "gemma-3-270m"
      : resolvedModelId;
  }

  return {
    chat: {
      ...base.chat,
      backend: options.backend ?? base.chat.backend,
      modelId: resolvedModelId,
    },
  };
}
