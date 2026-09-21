# Architecture

`@ricardo/jax-llm` uses a two-layer abstraction. The entry point is `mod.ts`,
which re-exports `engine/index.ts`.

## Layer 1 — Runtime (`engine/runtime/`)

Low-level JAX-JS model lifecycle, deliberately unaware of chat semantics:

- `ModelRuntime` (`runtime.ts`): `load()` runs device init → tokenizer →
  weights. `initDevice()` selects the `webgpu` (default) or `wasm` backend;
  `loadTokenizer()` tries multiple URL formats and tokenizer types (BPE first,
  SentencePiece fallback); `loadWeights()` fetches and hydrates the checkpoint;
  `createSession()` hands out inference sessions.
- Types (`types.ts`): `ModelDefinition`, `LoadedModel`, `InferenceSession`
  (`prefill(tokenIds)` → `step(token)` → `dispose()`), `TokenizerInterface`,
  `RuntimeConfig` / `DEFAULT_RUNTIME_CONFIG`, `SamplingDefaults`, `Backend`.
- Registry (`registry.ts`): `resolveModel()` maps a built-in ID, short alias, or
  Hugging Face repo ID to a `ModelDefinition`. See [models.md](models.md).
- Safetensors (`safetensors.ts`): `parseSafetensors()` wraps the upstream parser
  with BF16 → F32 conversion.
- Training (`training.ts`): `createOptimizer()`, `crossEntropyLoss()`,
  `TrainingRunner`. See [training.md](training.md).

## Layer 2 — Chat (`engine/chat/`)

High-level API most consumers use:

- `ChatEngine` (`chat_engine.ts`): `new ChatEngine(model, options?)` builds a
  `ModelRuntime` internally. `init()` loads and resolves sampling defaults;
  `chat(message)` is the one-shot path; `chatStream(history)` is the streaming
  generate loop (prefill prompt → sample → check stop tokens → step → yield
  cumulative text → dispose session). `getSystemInfo()`, `getRuntime()` (escape
  hatch to the runtime), and `dispose()`.
- Sampling (`sampler.ts`): `sampleLogits()` implements temperature (≤0 =
  greedy), top-K, top-p nucleus filtering, and repetition penalty;
  `resolveSamplingDefaults()` merges per-model defaults with caller overrides.
- Prompts (`prompt.ts`): `gemmaPrompt`, `lfmPrompt`, `qwenPrompt`,
  `bonsaiPrompt`, `gptPrompt`, `phiPrompt`, `maplePrompt`, `genericPrompt`.
- Types (`types.ts`): `ChatMessage`, `ChatEngineOptions`, `SystemInfo`.

## Models (`engine/llm/`)

- `model.ts`: `CHAT_MODELS` registry, `CHAT_MODEL_IDS`, `ChatModelId`,
  `DEFAULT_CHAT_MODEL_ID`, plus `ChatModel`, `LoadedChatModel`,
  `ChatModelSession`, `ChatTokenizer` types and the shared `defineChatModel`
  helper.
- `gemma.ts` / `lfm.ts`: `runGemmaPrefill` / `runGemmaStep`, `runLfmPrefill` /
  `runLfmStep` and the `GemmaModel` / `LfmModel` weight types. Other families
  (Qwen, Bonsai, GPT, Phi, Maple) are wired through `model.ts` with their own
  loaders under `llm/loaders/`, KV-cache state under `llm/state/`, configs under
  `llm/configs/`, and layers under `llm/layers/`.

## Tokenizer (`engine/tokenizer/`)

`HuggingFaceBpeTokenizer` supports two `tokenizer.json` styles: byte-level BPE
(LFM2.5, GPT-2, via `BpeEncoding`) and SentencePiece-style BPE (`▁` word
boundaries, `<0xNN>` byte fallback, via a custom Unicode encoder).
`fromBinary()` parses raw JSON bytes; `encode()` / `decode()` round-trip text;
`specialTokenIds` tracks special tokens filtered from generated output.

## Inference flow

1. `ChatEngine.init()` → `ModelRuntime.load()` (device, tokenizer, weights).
2. `chatStream(history)` → `definition.encodePrompt(tokenizer, history)` →
   `session.prefill(inputIds)` → loop: `sampleLogits` → stop-token check →
   `session.step(token)` → yield `decodeGenerated(tokens)`.
3. `session.dispose()` in a `finally` block; `engine.dispose()` / runtime
   `dispose()` releases weights via reference-counted `tree.dispose`.

## Supporting modules

- `engine/database/` (`Database`, `DenoKVDB`): generic key-value storage.
- `engine/config.ts` (`Config`, `DEFAULT_CONFIG`, `loadConfig()`,
  `createConfig()`): file-based configuration. See
  [configuration.md](configuration.md).
- `engine/cache/`, `engine/training/`, `engine/agents/`, `engine/evolver/`:
  auxiliary modules not re-exported from the package root.
