# API reference

All symbols below are exported from the package root (`mod.ts` →
`engine/index.ts`). Run `deno doc mod.ts` for signatures; JSDoc on each export
is the source of truth.

## Chat (`engine/chat/`)

- `ChatEngine` — `new ChatEngine(model, options?)`; `init()`, `chat()`,
  `chatStream()`, `getSystemInfo()`, `getRuntime()`, `dispose()`.
- `ChatMessage` — `{ role: "system" | "user" | "assistant"; content: string }`.
- `ChatEngineOptions` —
  `{ backend?, maxTokens?, sampling?,
  weightOverrides?, tokenizerOverrides? }`.
- `SystemInfo` — `{ device, model, contextUsage, contextSize }`.
- `sampleLogits(logits, opts)` — temperature / top-K / top-p / repetition
  penalty sampling; temperature ≤ 0 is greedy.
- `SamplingOptions` — `SamplingDefaults` plus `previousTokens`.
- `SamplingDefaults` — `{ temperature, topK, topP, repetitionPenalty }`.
- `resolveSamplingDefaults(base, overrides?)` — merge per-model defaults.
- `gemmaPrompt`, `lfmPrompt`, `qwenPrompt`, `bonsaiPrompt`, `gptPrompt`,
  `phiPrompt`, `maplePrompt`, `genericPrompt` — per-family prompt formatters
  `(history: ChatMessage[]) => string`.

## Runtime (`engine/runtime/`)

- `ModelRuntime` — `new ModelRuntime(modelId, config?)`; `load()`,
  `initDevice()`, `loadTokenizer()`, `loadWeights()`, `getTokenizer()`,
  `getModel()`, `isLoaded`, `createSession()`, `dispose()`.
- `ModelDefinition` —
  `{ id, label, downloadSize, weightsUrl, tokenizerUrl,
  contextSize, defaults, createTokenizer, loadCheckpoint, formatPrompt,
  encodePrompt, stopTokens }`.
- `LoadedModel` — `{ modelId, device, createSession(), dispose() }`.
- `InferenceSession` — `{ prefill(tokenIds), step(token), dispose() }`.
- `TokenizerInterface` —
  `{ bosToken, eosToken, encode, decode,
  decodeGenerated }`.
- `RuntimeConfig` —
  `{ backend, dtype, weightOverrides?, tokenizerOverrides?
  }`;
  `DEFAULT_RUNTIME_CONFIG` (webgpu / float32); `Backend` (`"webgpu" | "wasm"`).
- `resolveModel(modelId, overrides?)` — built-in ID → alias → Hugging Face repo.
  Helpers: `isBuiltInModel`, `isAlias`, `isHuggingFaceRepo`,
  `tokenizerUrlCandidates`; constants `MODEL_IDS`, `DEFAULT_MODEL_ID`.
- `parseSafetensors(data)` — safetensors parser with BF16 → F32 support.

## Models (`engine/llm/model.ts`, `gemma.ts`, `lfm.ts`)

- `CHAT_MODELS` — registry keyed by model ID; `CHAT_MODEL_IDS` lists IDs;
  `ChatModelId` is the ID union; `DEFAULT_CHAT_MODEL_ID` is `lfm2.5-350m`.
- `ChatModel` —
  `{ id, label, downloadSize, tokenizerUrl, weightsUrl,
  defaults, contextSize, loadCheckpoint, createTokenizer, formatPrompt,
  encodePrompt, stopTokens }`.
- `LoadedChatModel`, `ChatModelSession`, `ChatTokenizer` — loaded checkpoint,
  prefill/decode session, and tokenizer with `decodeGenerated` filtering.
- `GemmaModel`, `runGemmaPrefill`, `runGemmaStep`; `LfmModel`, `runLfmPrefill`,
  `runLfmStep` — low-level weight types and passes.

## Tokenizer (`engine/tokenizer/tokenizer.ts`)

- `HuggingFaceBpeTokenizer` — `bosToken`, `eosToken`, `padToken`,
  `specialTokenIds`; `constructor(data)`, `static fromBinary(data)`,
  `encode(text)`, `decode(tokens)`.

## Training (`engine/runtime/training.ts`)

- `TrainingRunner` — `new TrainingRunner(definition, config)`; `init(params)`,
  `step(batch)`, `train(dataset)` (async generator of `{ index, loss }`),
  `getParams()`.
- `TrainingConfig` —
  `{ optimizer, optimizerOptions, epochs, batchSize,
  onEpochEnd?, onBatchEnd? }`;
  `TrainingBatch` — `{ inputIds, targets }`.
- `createOptimizer(type, opts)` — `OptimizerType` (`"sgd" | "adam" |
  "adamw"`)
  with `OptimizerOptions` (`learningRate`, `weightDecay?`, `maxGradNorm?`,
  `beta1?`, `beta2?`, `eps?`).
- `crossEntropyLoss(logits, targets)` — causal language-modeling loss.

## Database (`engine/database/`)

- `Database<K, V>` — `get(key)`, `add(key, value)`, `update(key, new_value)`.
- `DenoKVDB<K, V>` — Deno KV-backed implementation.

## Config (`engine/config.ts`)

- `Config` — `{ chat: { backend, modelId, maxTokens, modelOverrides? } }`.
- `DEFAULT_CONFIG` — webgpu / default model / 4096 tokens.
- `loadConfig()` — read `config.json` (or `JAX_JS_CONFIG_PATH`), merged over
  defaults. `createConfig(options)` — apply CLI-style overrides and alias
  resolution. Re-exports `ChatEngineOptions`.
