# @ricardo/jax-llm

A JAX-based LLM engine for JavaScript/TypeScript, designed for Deno.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deno](https://img.shields.io/badge/deno-2.x-brightgreen.svg)](https://deno.land/)
[![JSR](https://jsr.io/badges/@ricardo/jax-llm)](https://jsr.io/@ricardo/jax-llm)

## Requirements

- **Deno 2.x** with `--allow-net --allow-read --allow-write --allow-env` (plus
  the `webgpu` and `kv` unstable flags used by `deno task test`).
- A **WebGPU**-capable browser/runtime by default, or `"wasm"` fallback.
- Network access on first run to download weights and tokenizer files from
  Hugging Face.

## Installation

```bash
deno add jsr:@ricardo/jax-llm
```

```typescript
import { ChatEngine } from "@ricardo/jax-llm";
```

## Quick Start

```typescript
import { ChatEngine } from "@ricardo/jax-llm";

const engine = new ChatEngine("lfm2.5-350m");
await engine.init();
const response = await engine.chat("Hello, how are you?");
console.log(response);
engine.dispose();
```

With options:

```typescript
const engine = new ChatEngine("gemma", {
  backend: "webgpu",
  maxTokens: 2048,
  sampling: { temperature: 0.5 },
});
await engine.init();
```

Streaming responses (`chatStream` yields cumulative text, so diff against the
previous chunk for the delta):

```typescript
for await (
  const chunk of engine.chatStream([
    { role: "user", content: "Tell me a story" },
  ])
) {
  console.log(chunk);
}
```

Multi-turn conversation:

```typescript
const reply = await engine.chatStream([
  { role: "system", content: "You are a concise assistant." },
  { role: "user", content: "What is JAX?" },
  { role: "assistant", content: "JAX is a numerical computing library." },
  { role: "user", content: "And how is it used here?" },
]);
```

## Models

Built-in model IDs (`CHAT_MODELS` / `MODEL_IDS`):

| ID              | Label               | Context | Download       |
| --------------- | ------------------- | ------- | -------------- |
| `lfm2.5-350m`   | LFM2.5 350M         | 4096    | 676 MB         |
| `gemma-3-270m`  | Gemma 3 270M        | 8192    | 536 MB         |
| `qwen2.5-0.5b`  | Qwen2.5 0.5B        | 4096    | 1.0 GB         |
| `bonsai`        | Bonsai (Llama)      | 2048    | 1.0 GB         |
| `gpt2`          | GPT-2               | 1024    | 548 MB         |
| `phi-2`         | Phi-2               | 2048    | 5.2 GB         |
| `maple-preview` | Maple Preview (MoE) | 4096    | 20B (9 shards) |

The default model is `lfm2.5-350m` (`DEFAULT_MODEL_ID` /
`DEFAULT_CHAT_MODEL_ID`).

Short aliases are also accepted: `lfm`, `lfm2`, `lfm2.5`, `gemma`, `qwen`,
`qwen2`, `qwen2.5`, `bonsai`, `gpt`, `gpt2`, `phi`, `phi2`, `maple`.

Any other `"org/model-name"` string is treated as a Hugging Face repo ID and
resolved dynamically (`isHuggingFaceRepo`, `resolveModel`), using the LFM2.5
checkpoint loader as a generic fallback. See [docs/models.md](docs/models.md).

## API Overview

All public symbols are exported from the package root (`mod.ts` →
`engine/index.ts`) and documented with JSDoc (`deno doc --lint` reports 0
`missing-jsdoc` errors). See [docs/api.md](docs/api.md) for the full reference.

**Chat layer** (`engine/chat/`) — what most consumers need:

| Symbol                                                                                                             | Kind      | Description                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------ | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `ChatEngine`                                                                                                       | class     | `new ChatEngine(model, options?)` → `init()` → `chat()` / `chatStream()`; plus `getSystemInfo()`, `getRuntime()`, `dispose()` |
| `ChatMessage`                                                                                                      | type      | `{ role: "system" \| "user" \| "assistant", content: string }`                                                                |
| `ChatEngineOptions`                                                                                                | type      | `{ backend?, maxTokens?, sampling?, weightOverrides?, tokenizerOverrides? }`                                                  |
| `SystemInfo`                                                                                                       | type      | Runtime status from `getSystemInfo()`                                                                                         |
| `sampleLogits`                                                                                                     | function  | Temperature / top-K / top-p / repetition-penalty sampling                                                                     |
| `SamplingOptions` / `SamplingDefaults`                                                                             | types     | Sampling parameters; `resolveSamplingDefaults()` merges overrides                                                             |
| `gemmaPrompt`, `lfmPrompt`, `qwenPrompt`, `bonsaiPrompt`, `gptPrompt`, `phiPrompt`, `maplePrompt`, `genericPrompt` | functions | Per-family prompt formatters                                                                                                  |

**Runtime layer** (`engine/runtime/`) — advanced use:

| Symbol                                                                                                                      | Kind                | Description                                                           |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------- |
| `ModelRuntime`                                                                                                              | class               | Device init, tokenizer/weight loading, `createSession()`, `dispose()` |
| `ModelDefinition` / `LoadedModel` / `InferenceSession`                                                                      | types               | `prefill()` → `step()` → `dispose()` inference sessions               |
| `RuntimeConfig` / `DEFAULT_RUNTIME_CONFIG`                                                                                  | types               | `{ backend, dtype, weightOverrides?, tokenizerOverrides? }`           |
| `resolveModel`, `isBuiltInModel`, `isAlias`, `isHuggingFaceRepo`, `MODEL_IDS`, `DEFAULT_MODEL_ID`, `tokenizerUrlCandidates` | functions/constants | Model resolution                                                      |
| `parseSafetensors`                                                                                                          | function            | Safetensors parser with BF16 → F32 support                            |
| `TrainingRunner`, `TrainingConfig`, `TrainingBatch`                                                                         | class/types         | Optax fine-tuning loop                                                |
| `createOptimizer`, `OptimizerType`, `OptimizerOptions`, `crossEntropyLoss`                                                  | function/types      | SGD / Adam / AdamW + causal LM loss                                   |

**Models, tokenizer, database, config:**

| Symbol                                                                                                              | Kind            | Description                                                                    |
| ------------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------ |
| `CHAT_MODELS`, `CHAT_MODEL_IDS`, `ChatModel`, `ChatModelId`, `LoadedChatModel`, `ChatModelSession`, `ChatTokenizer` | constants/types | Model registry (`engine/llm/model.ts`)                                         |
| `GemmaModel`, `runGemmaPrefill`, `runGemmaStep`, `LfmModel`, `runLfmPrefill`, `runLfmStep`                          | types/functions | Low-level model passes                                                         |
| `HuggingFaceBpeTokenizer`                                                                                           | class           | `fromBinary()`, `encode()`, `decode()`; byte-level and SentencePiece-style BPE |
| `Database`, `DenoKVDB`                                                                                              | interface/class | `get()` / `add()` / `update()` key-value storage                               |
| `Config`, `DEFAULT_CONFIG`, `loadConfig()`, `createConfig()`                                                        | types/functions | `config.json` handling (see below)                                             |

## Architecture

Two-layer abstraction:

- **Runtime layer** (`engine/runtime/`): device init, tokenizer/weight loading,
  JIT-compiled inference sessions, memory management, safetensors parsing (with
  BF16 support), and optax-based training.
- **Chat layer** (`engine/chat/`): `ChatEngine` facade plus sampling and prompt
  formatting. Pass a model name and call `engine.chat(input)`.

Model implementations live in `engine/llm/` (registry in `model.ts`, passes in
`gemma.ts` / `lfm.ts`, loaders in `llm/loaders/`, KV-cache state in
`llm/state/`, configs in `llm/configs/`); tokenizers in `engine/tokenizer/`.

See [docs/architecture.md](docs/architecture.md) for the full guide.

## Configuration

`ChatEngine` options (`ChatEngineOptions`):

```typescript
{
  backend: "webgpu",      // or "wasm"
  maxTokens: 4096,
  sampling: { temperature: 0.8, topK: 64, topP: 0.95, repetitionPenalty: 1 },
  weightOverrides: { "lfm2.5-350m": "https://mirror.example/model.safetensors" },
  tokenizerOverrides: { "lfm2.5-350m": "https://mirror.example/tokenizer.json" },
}
```

`loadConfig()` reads `config.json` at the project root (or the path in
`JAX_JS_CONFIG_PATH`) and merges it over `DEFAULT_CONFIG`; `createConfig()`
resolves model aliases on top of that. All fields are optional. See
[docs/configuration.md](docs/configuration.md).

## Training

Fine-tuning utilities in `engine/runtime/training.ts`:

```typescript
import { TrainingRunner } from "@ricardo/jax-llm";

const runner = new TrainingRunner(runtime.definition, {
  optimizer: "adamw",
  optimizerOptions: { learningRate: 1e-4 },
  epochs: 3,
  batchSize: 4,
});
runner.init(modelWeights);
for await (const epoch of runner.train(dataset)) {
  console.log(`Epoch ${epoch.index} loss: ${epoch.loss}`);
}
```

See [docs/training.md](docs/training.md).

## Testing

```bash
# All tests (unit + integration; integration needs --allow-net)
deno task test

# Unit tests only (offline-safe)
deno task test:unit

# Integration tests (downloads weights; skipped without --allow-net)
deno task test:integration

# Coverage
deno task test:coverage

# Watch mode
deno task dev
```

Unit tests live next to their modules (`engine/*/*_test.ts`, e.g.
`tokenizer_test.ts`, `sampler_test.ts`, `prompt_test.ts`, `registry_test.ts`,
`runtime_test.ts`, `model_test.ts`, `database_test.ts`, `kv_test.ts`,
`config_test.ts`); end-to-end tests live in `tests/integration_test.ts`.
`deno task test:unit` currently passes 75 tests.

## Development

```bash
deno task lint        # deno lint
deno fmt              # format
deno task test:unit   # fast feedback loop
deno doc --lint mod.ts  # JSDoc coverage (0 missing-jsdoc errors required)
```

JSR requires ≥80% of exported symbols to have documentation; this package is at
100%. Keep every new export documented. See
[docs/development.md](docs/development.md) for the publishing checklist.

## Documentation

- [docs/](docs/) — architecture, API reference, models, configuration, training,
  and development guides.
- `deno doc mod.ts` — local API docs; the JSR package page renders the same
  JSDoc.
- Source JSDoc on every export is the source of truth; `docs/api.md` is a
  grouped overview.

## License

MIT — see [LICENSE](LICENSE).
