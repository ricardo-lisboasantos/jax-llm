import { tree } from "npm:@jax-js/jax@^0.1.25";
import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { tokenizers } from "npm:@jax-js/loaders@^0.1.3";
import type { safetensors } from "npm:@jax-js/loaders@^0.1.3";
import { parseSafetensors } from "../runtime/safetensors.ts";

import { GEMMA_CONFIG } from "./configs/gemma_config.ts";
import { createGemmaState, type GemmaState } from "./state/gemma_state.ts";
import { fromSafetensors } from "./loaders/gemma.ts";
import { type GemmaModel, runGemmaPrefill, runGemmaStep } from "./gemma.ts";
import { HuggingFaceBpeTokenizer } from "../tokenizer/tokenizer.ts";
import {
  createLfmState,
  disposeLfmPagedCache,
  type LfmState,
} from "./state/lfm_state.ts";
import { lfmFromSafetensors } from "./loaders/lfm.ts";
import {
  confirmLfmDraft,
  type LfmModel,
  runLfmPrefill,
  runLfmScoreDrafts,
  runLfmStep,
  truncateLfmDraft,
} from "./lfm.ts";
import { type QwenModel, runQwenPrefill, runQwenStep } from "./qwen.ts";
import { createQwenState, type QwenState } from "./state/qwen_state.ts";
import { qwenFromSafetensors } from "./loaders/qwen.ts";
import { type BonsaiModel, runBonsaiPrefill, runBonsaiStep } from "./bonsai.ts";
import {
  createGptState,
  gptFromSafetensors,
  type GptModel,
  type GptState,
  runGptPrefill,
  runGptStep,
} from "./gpt.ts";
import { type PhiModel, runPhiPrefill, runPhiStep } from "./phi.ts";
import { createPhiState, type PhiState } from "./state/phi_state.ts";
import { phiFromSafetensors } from "./loaders/phi.ts";
import { type MapleModel, runMaplePrefill, runMapleStep } from "./maple.ts";

// Re-export shared types from the new layered architecture for backward
// compatibility. These now live in the runtime and chat layers.
export type { ChatMessage } from "../chat/types.ts";
export type {
  InferenceSession,
  LoadedModel,
  SamplingDefaults,
  TokenizerInterface,
} from "../runtime/types.ts";
import type { ChatMessage } from "../chat/types.ts";
import type { SamplingDefaults } from "../runtime/types.ts";
import {
  bonsaiPrompt,
  gemmaPrompt,
  gptPrompt,
  lfmPrompt,
  maplePrompt,
  phiPrompt,
  qwenPrompt,
} from "../chat/prompt.ts";

export {
  bonsaiPrompt,
  gemmaPrompt,
  gptPrompt,
  lfmPrompt,
  maplePrompt,
  phiPrompt,
  qwenPrompt,
};
import { bonsaiFromSafetensors } from "./loaders/bonsai.ts";
import { type BonsaiState, createBonsaiState } from "./state/bonsai_state.ts";
import {
  beginMaplePagedLoad,
  mapleFromSafetensors,
  mapleFromSafetensorsPaged,
} from "./loaders/maple.ts";
import { createMapleState, type MapleState } from "./state/maple_state.ts";

// Gemma chat-template control tokens in tokenizer.model.
const START_OF_TURN_TOKEN = 105;
const END_OF_TURN_TOKEN = 106;

/** A tokenizer with model-specific output filtering hidden behind it. */
export type ChatTokenizer = {
  bosToken: number;
  eosToken: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  decodeGenerated(tokens: number[]): string;
};

/** One stateful prefill/decode sequence. Async to allow weight paging. */
export type ChatModelSession = {
  prefill(tokenIds: np.Array): Promise<np.Array>;
  step(token: number): Promise<np.Array>;
  /** Batched draft scoring (present only when the model supports it). */
  scoreTokens?(draftIds: np.Array): Promise<np.Array>;
  /** Keep scored state after full draft acceptance. */
  confirmDraft?(): void;
  /** Roll back to the pre-score state after draft rejection. */
  truncateDraft?(): void;
  dispose(): void;
};

/** A checkpoint loaded onto the currently selected device. */
export type LoadedChatModel = {
  modelId: string;
  definition: ChatModel;
  device: string;
  createSession(): ChatModelSession;
  dispose(): void;
};

/**
 * A chat-capable model definition: checkpoint loading, tokenization,
 * and prompt formatting for one model family.
 */
export type ChatModel<Id extends string = string> = {
  id: Id;
  label: string;
  downloadSize: string;
  tokenizerUrl: string;
  weightsUrl: string;
  defaults: SamplingDefaults;
  contextSize: number;

  loadCheckpoint(
    data: Uint8Array<ArrayBuffer>,
    dtype: np.DType,
    device: string,
  ): Promise<LoadedChatModel>;

  /**
   * Load from pre-parsed shard files (multi-shard
   * `model.safetensors.index.json` checkpoints). Mirrors
   * `ModelDefinition.loadCheckpointFromFiles` in the runtime layer.
   */
  loadCheckpointFromFiles?(
    files: safetensors.File[],
    dtype: np.DType,
    device: string,
  ): Promise<LoadedChatModel>;
  beginPagedLoad?(
    dtype: np.DType,
  ): Promise<PagedModelStaging<LoadedChatModel>>;

  createTokenizer(data: Uint8Array): ChatTokenizer;
  formatPrompt(history: ChatMessage[]): string;
  encodePrompt(tokenizer: ChatTokenizer, history: ChatMessage[]): number[];
  stopTokens(tokenizer: ChatTokenizer): number[];
};

type BaseTokenizer = {
  bosToken: number;
  eosToken: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
};

type ChatModelImplementation<
  Id extends string,
  Model,
  State,
  Tokenizer extends BaseTokenizer,
> = {
  id: Id;
  label: string;
  downloadSize: string;
  weightsUrl: string;
  tokenizerUrl: string;
  defaults: SamplingDefaults;
  contextSize: number;
  createTokenizer(data: Uint8Array): Tokenizer;
  decodeGenerated(tokenizer: Tokenizer, tokens: number[]): string;
  formatPrompt(history: ChatMessage[]): string;
  stopTokens(tokenizer: ChatTokenizer): number[];
  loadModel(file: safetensors.File, dtype: np.DType): Promise<Model>;
  /**
   * Sharded variant: receives one parsed File per shard so large
   * checkpoints (MoE) can stage tensors incrementally instead of
   * merging all shard buffers at once. Falls back to merge + loadModel.
   */
  loadModelPaged?(
    files: safetensors.File[],
    dtype: np.DType,
  ): Promise<Model>;
  /**
   * Incremental sharded load (B2): one shard at a time, peak ≈ 1 shard.
   * Falls back to `loadModelPaged` / merge + `loadModel` when absent.
   */
  beginPagedLoad?(dtype: np.DType): Promise<PagedModelStaging<Model>>;
  createState(dtype: np.DType): State;
  // Sync or async — device-resident models return directly, paged models
  // (MoE expert on-demand) return a promise. The session wrapper awaits both.
  prefill(
    model: Model,
    tokenIds: np.Array,
    state: State,
  ): np.Array | Promise<np.Array>;
  step(model: Model, token: number, state: State): np.Array | Promise<np.Array>;
  /**
   * Batched draft scoring for speculative verification (optional).
   * Returns per-token logits; advances state exactly like sequential
   * steps. Pair with `acceptDraft` / `discardDraft` below.
   */
  scoreDrafts?(
    model: Model,
    draftIds: np.Array,
    state: State,
  ): np.Array | Promise<np.Array>;
  /** Keep scored state (all drafts accepted). */
  acceptDraft?(state: State): void;
  /** Roll back to the pre-score snapshot (draft rejected). */
  discardDraft?(state: State): void;
};

/**
 * Shared session/dispose wrapper so single-file and sharded loads behave
 * identically. Extracted from `defineChatModel` to avoid duplicating the
 * lifecycle closure.
 */
function buildLoadedChatModel<Model, State>(
  definition: { id: string; label: string },
  implementation: {
    id: string;
    createState(dtype: np.DType): State;
    prefill(
      model: Model,
      tokenIds: np.Array,
      state: State,
    ): np.Array | Promise<np.Array>;
    step(
      model: Model,
      token: number,
      state: State,
    ): np.Array | Promise<np.Array>;
    scoreDrafts?(
      model: Model,
      draftIds: np.Array,
      state: State,
    ): np.Array | Promise<np.Array>;
    acceptDraft?(state: State): void;
    discardDraft?(state: State): void;
  },
  model: Model,
  dtype: np.DType,
  device: string,
): LoadedChatModel {
  let disposed = false;

  return {
    modelId: definition.id,
    definition: definition as LoadedChatModel["definition"],
    device,

    createSession() {
      if (disposed) throw new Error(`${definition.label} is disposed`);
      const state = implementation.createState(dtype);
      let sessionDisposed = false;

      const assertActive = () => {
        if (sessionDisposed) {
          throw new Error(`${definition.label} session is disposed`);
        }
      };

      return {
        prefill(tokenIds) {
          assertActive();
          return Promise.resolve(
            implementation.prefill(tree.ref(model), tokenIds, state),
          );
        },

        step(token) {
          assertActive();
          return Promise.resolve(
            implementation.step(tree.ref(model), token, state),
          );
        },

        ...(implementation.scoreDrafts
          ? {
            scoreTokens(draftIds: np.Array) {
              assertActive();
              return Promise.resolve(
                implementation.scoreDrafts!(tree.ref(model), draftIds, state),
              );
            },
            confirmDraft() {
              assertActive();
              implementation.acceptDraft?.(state);
            },
            truncateDraft() {
              assertActive();
              implementation.discardDraft?.(state);
            },
          }
          : {}),

        dispose() {
          if (sessionDisposed) return;
          sessionDisposed = true;
          // Phase 3.0: Dispose paged cache if this is an LFM state
          if (implementation.id === "lfm2.5-350m") {
            const lfmState = state as LfmState;
            if (lfmState.pagedCache) {
              disposeLfmPagedCache(lfmState);
            }
          }
          tree.dispose(state);
        },
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      tree.dispose(model);
    },
  };
}
/**
 * Merge parsed shard files into one File (fallback for models without
 * `loadModelPaged`). Duplicate tensor names across shards are rejected.
 */
function mergeShardFiles(
  modelId: string,
  files: safetensors.File[],
): safetensors.File {
  const merged: safetensors.File = { tensors: {}, totalSize: 0 };
  for (const f of files) {
    if (f.metadata && !merged.metadata) merged.metadata = f.metadata;
    for (const [key, tensor] of Object.entries(f.tensors)) {
      if (key in merged.tensors) {
        throw new Error(
          `Duplicate tensor "${key}" across shards for "${modelId}"`,
        );
      }
      merged.tensors[key] = tensor;
    }
    merged.totalSize += f.totalSize;
  }
  return merged;
}
/**
 * Incremental sharded-load handle (B2): consume one parsed shard at a
 * time so peak memory stays near a single shard instead of the full
 * checkpoint. `finish` hydrates device arrays and returns the model.
 */
export type PagedModelStaging<Model> = {
  stageShard(file: safetensors.File): Promise<void>;
  finish(device: string): Promise<Model>;
};
/**
 * Keeps each model implementation fully typed while exposing a small,
 * type-erased interface to the page.
 */
function defineChatModel<
  const Id extends string,
  Model,
  State,
  Tokenizer extends BaseTokenizer,
>(
  implementation: ChatModelImplementation<Id, Model, State, Tokenizer>,
): ChatModel<Id> {
  const definition: ChatModel<Id> = {
    id: implementation.id,
    label: implementation.label,
    downloadSize: implementation.downloadSize,
    weightsUrl: implementation.weightsUrl,
    tokenizerUrl: implementation.tokenizerUrl,
    defaults: implementation.defaults,
    contextSize: implementation.contextSize,

    createTokenizer(data) {
      const tokenizer = implementation.createTokenizer(data);
      return {
        bosToken: tokenizer.bosToken,
        eosToken: tokenizer.eosToken,
        encode: (text) => tokenizer.encode(text),
        decode: (tokens) => tokenizer.decode(tokens),
        decodeGenerated: (tokens) =>
          implementation.decodeGenerated(tokenizer, tokens),
      };
    },

    formatPrompt: implementation.formatPrompt,
    stopTokens: implementation.stopTokens,
    encodePrompt(tokenizer, history) {
      return [
        tokenizer.bosToken,
        ...tokenizer.encode(implementation.formatPrompt(history)),
      ];
    },
    async loadCheckpoint(data, dtype, device) {
      const model = await implementation.loadModel(
        parseSafetensors(data),
        dtype,
      );
      return buildLoadedChatModel(
        definition,
        implementation,
        model,
        dtype,
        device,
      );
    },
    async loadCheckpointFromFiles(files, dtype, device) {
      const model = implementation.loadModelPaged
        ? await implementation.loadModelPaged(files, dtype)
        : await implementation.loadModel(
          mergeShardFiles(definition.id, files),
          dtype,
        );
      return buildLoadedChatModel(
        definition,
        implementation,
        model,
        dtype,
        device,
      );
    },
    ...(implementation.beginPagedLoad
      ? {
        beginPagedLoad: async (dtype: np.DType) => {
          const staging = await implementation.beginPagedLoad!(dtype);
          return {
            stageShard: (file: safetensors.File) => staging.stageShard(file),
            finish: async (device: string) => {
              const model = await staging.finish(device);
              return buildLoadedChatModel(
                definition,
                implementation,
                model,
                dtype,
                device,
              );
            },
          };
        },
      }
      : {}),
  };

  return definition;
}

const gemma: ChatModel<"gemma-3-270m"> = defineChatModel<
  "gemma-3-270m",
  GemmaModel,
  GemmaState,
  tokenizers.SentencePiece
>({
  id: "gemma-3-270m",
  label: "Gemma 3 270M",
  downloadSize: "536 MB",
  weightsUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m/model-it-fp16.safetensors",
  tokenizerUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m/tokenizer.model",
  contextSize: 8192,
  defaults: {
    temperature: 0.8,
    topK: 64,
    topP: 0.95,
    repetitionPenalty: 1,
  },
  createTokenizer: tokenizers.SentencePiece.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter(
        (token) =>
          token !== GEMMA_CONFIG.padTokenId &&
          token !== tokenizer.bosToken &&
          token !== tokenizer.eosToken &&
          token !== START_OF_TURN_TOKEN &&
          token !== END_OF_TURN_TOKEN,
      ),
    ),
  formatPrompt: gemmaPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken, END_OF_TURN_TOKEN],
  loadModel: fromSafetensors,
  createState: (dtype) => createGemmaState({ dtype }),
  prefill: runGemmaPrefill,
  step: runGemmaStep,
});

const lfm: ChatModel<"lfm2.5-350m"> = defineChatModel<
  "lfm2.5-350m",
  LfmModel,
  LfmState,
  HuggingFaceBpeTokenizer
>({
  id: "lfm2.5-350m",
  label: "LFM2.5 350M",
  downloadSize: "676 MB",
  weightsUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/lfm2.5-350m/model-fp16.safetensors",
  tokenizerUrl:
    "https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/tokenizer.json",
  contextSize: 4096,
  defaults: {
    temperature: 0.1,
    topK: 50,
    topP: 1,
    repetitionPenalty: 1.05,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: lfmPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: lfmFromSafetensors,
  createState: (dtype) => createLfmState({ dtype }),
  prefill: runLfmPrefill,
  step: runLfmStep,
  scoreDrafts: runLfmScoreDrafts,
  acceptDraft: confirmLfmDraft,
  discardDraft: truncateLfmDraft,
});

const qwen: ChatModel<"qwen2.5-0.5b"> = defineChatModel<
  "qwen2.5-0.5b",
  QwenModel,
  QwenState,
  HuggingFaceBpeTokenizer
>({
  id: "qwen2.5-0.5b",
  label: "Qwen2.5 0.5B",
  downloadSize: "1.0 GB",
  weightsUrl:
    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors",
  tokenizerUrl:
    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json",
  contextSize: 4096,
  defaults: {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: qwenPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: qwenFromSafetensors,
  createState: (dtype) => createQwenState({ dtype }),
  prefill: runQwenPrefill,
  step: runQwenStep,
});

const bonsai: ChatModel<"bonsai"> = defineChatModel<
  "bonsai",
  BonsaiModel,
  BonsaiState,
  HuggingFaceBpeTokenizer
>({
  id: "bonsai",
  label: "Bonsai (Llama)",
  downloadSize: "1.0 GB",
  weightsUrl:
    "https://huggingface.co/deepgrove/Bonsai/resolve/main/model.safetensors",
  tokenizerUrl:
    "https://huggingface.co/deepgrove/Bonsai/resolve/main/tokenizer.json",
  contextSize: 2048,
  defaults: {
    temperature: 0.8,
    topK: 64,
    topP: 0.95,
    repetitionPenalty: 1,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: bonsaiPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: bonsaiFromSafetensors,
  createState: (dtype) => createBonsaiState({ dtype }),
  prefill: runBonsaiPrefill,
  step: runBonsaiStep,
});

const gpt: ChatModel<"gpt2"> = defineChatModel<
  "gpt2",
  GptModel,
  GptState,
  HuggingFaceBpeTokenizer
>({
  id: "gpt2",
  label: "GPT-2",
  downloadSize: "548 MB",
  weightsUrl: "https://huggingface.co/gpt2/resolve/main/model.safetensors",
  tokenizerUrl: "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
  contextSize: 1024,
  defaults: {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: gptPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: gptFromSafetensors,
  createState: (dtype) => createGptState({ dtype }),
  prefill: runGptPrefill,
  step: runGptStep,
});

const phi: ChatModel<"phi-2"> = defineChatModel<
  "phi-2",
  PhiModel,
  PhiState,
  HuggingFaceBpeTokenizer
>({
  id: "phi-2",
  label: "Phi-2",
  downloadSize: "5.2 GB",
  weightsUrl:
    "https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors.index.json",
  tokenizerUrl:
    "https://huggingface.co/microsoft/phi-2/resolve/main/tokenizer.json",
  contextSize: 2048,
  defaults: {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.1,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: phiPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: phiFromSafetensors,
  createState: (dtype) => createPhiState({ dtype }),
  prefill: runPhiPrefill,
  step: runPhiStep,
});

const maple: ChatModel<"maple-preview"> = defineChatModel<
  "maple-preview",
  MapleModel,
  MapleState,
  HuggingFaceBpeTokenizer
>({
  id: "maple-preview",
  label: "Maple Preview (MoE)",
  downloadSize: "20B (9 shards)",
  weightsUrl:
    "https://huggingface.co/deepgrove/maple-preview/resolve/main/model.safetensors.index.json",
  tokenizerUrl:
    "https://huggingface.co/deepgrove/maple-preview/resolve/main/tokenizer.json",
  contextSize: 4096,
  defaults: {
    temperature: 0.6,
    topK: 50,
    topP: 0.95,
    repetitionPenalty: 1.05,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: maplePrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: mapleFromSafetensors,
  loadModelPaged: (files, dtype) => mapleFromSafetensorsPaged(files, dtype),
  beginPagedLoad: (dtype) => beginMaplePagedLoad(dtype),
  createState: (dtype) => createMapleState({ dtype }),
  prefill: runMaplePrefill,
  step: runMapleStep,
});

/** Registry of all built-in chat models, keyed by model ID. */
export const CHAT_MODELS = {
  [gemma.id]: gemma,
  [lfm.id]: lfm,
  [qwen.id]: qwen,
  [bonsai.id]: bonsai,
  [gpt.id]: gpt,
  [phi.id]: phi,
  [maple.id]: maple,
};

/** Union of built-in chat model IDs from {@linkcode CHAT_MODELS}. */
export type ChatModelId = keyof typeof CHAT_MODELS;

/** List of all built-in chat model IDs. */
export const CHAT_MODEL_IDS = Object.keys(CHAT_MODELS) as ChatModelId[];
/** Default chat model used when no model ID is provided. */
export const DEFAULT_CHAT_MODEL_ID: ChatModelId = lfm.id;
