# Models

## Built-in models

Registered in `CHAT_MODELS` (`engine/llm/model.ts`), listed by `MODEL_IDS` and
`CHAT_MODEL_IDS`:

| ID              | Label               | Context | Download       |
| --------------- | ------------------- | ------- | -------------- |
| `lfm2.5-350m`   | LFM2.5 350M         | 4096    | 676 MB         |
| `gemma-3-270m`  | Gemma 3 270M        | 8192    | 536 MB         |
| `qwen2.5-0.5b`  | Qwen2.5 0.5B        | 4096    | 1.0 GB         |
| `bonsai`        | Bonsai (Llama)      | 2048    | 1.0 GB         |
| `gpt2`          | GPT-2               | 1024    | 548 MB         |
| `phi-2`         | Phi-2               | 2048    | 5.2 GB         |
| `maple-preview` | Maple Preview (MoE) | 4096    | 20B (9 shards) |

Defaults: `DEFAULT_MODEL_ID` and `DEFAULT_CHAT_MODEL_ID` are both `lfm2.5-350m`.
Per-model sampling defaults live on each definition (`definition.defaults`).

## Aliases

Short names resolved by `resolveModel()` (`engine/runtime/registry.ts`):

| Alias                      | Resolves to     |
| -------------------------- | --------------- |
| `lfm`, `lfm2`, `lfm2.5`    | `lfm2.5-350m`   |
| `gemma`                    | `gemma-3-270m`  |
| `qwen`, `qwen2`, `qwen2.5` | `qwen2.5-0.5b`  |
| `bonsai`                   | `bonsai`        |
| `gpt`, `gpt2`              | `gpt2`          |
| `phi`, `phi2`              | `phi-2`         |
| `maple`                    | `maple-preview` |

Use `isBuiltInModel()` / `isAlias()` as type guards.

## Hugging Face repos

Any other `"org/model-name"` string (`isHuggingFaceRepo()`) is resolved
dynamically: weights default to `<repo>/model.safetensors`, tokenizer to
`<repo>/tokenizer.json`, with a generic ChatML prompt and the LFM2.5 checkpoint
loader as fallback. Architectures incompatible with that loader raise an
explanatory error — they need a dedicated implementation under `engine/llm/`.

```typescript
const engine = new ChatEngine("deepgrove/Bonsai");
```

## URL overrides

Point any model at a mirror without code changes:

```typescript
new ChatEngine("lfm2.5-350m", {
  weightOverrides: {
    "lfm2.5-350m": "https://mirror.example/model.safetensors",
  },
  tokenizerOverrides: {
    "lfm2.5-350m": "https://mirror.example/tokenizer.json",
  },
});
```

`ModelRuntime` forwards these to `resolveModel()`; `tokenizerUrlCandidates()`
adds `tokenizer.json` / `tokenizer.model` / `tokenizer.spm` fallbacks next to
the primary tokenizer URL.
