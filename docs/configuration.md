# Configuration

## `ChatEngineOptions` (preferred)

```typescript
const engine = new ChatEngine("lfm2.5-350m", {
  backend: "webgpu", // or "wasm"
  maxTokens: 2048,
  sampling: { temperature: 0.5, topK: 40 },
  weightOverrides: {
    "lfm2.5-350m": "https://mirror.example/model.safetensors",
  },
  tokenizerOverrides: {
    "lfm2.5-350m": "https://mirror.example/tokenizer.json",
  },
});
```

All fields are optional. `sampling` is a `Partial<SamplingDefaults>` merged over
the model's own defaults via `resolveSamplingDefaults()`.

## `RuntimeConfig` (advanced)

`new ModelRuntime(modelId, config)` takes
`{ backend, dtype,
weightOverrides?, tokenizerOverrides? }`
(`engine/runtime/types.ts`). `DEFAULT_RUNTIME_CONFIG` is webgpu / float32.
`ChatEngine` builds this from `ChatEngineOptions` internally — reach for it only
when using the runtime directly.

## `config.json` file

`loadConfig()` (`engine/config.ts`) reads `config.json` at the project root, or
the path in `JAX_JS_CONFIG_PATH`, and merges it over `DEFAULT_CONFIG`:

```json
{
  "chat": {
    "backend": "webgpu",
    "modelId": "lfm2.5-350m",
    "maxTokens": 4096,
    "modelOverrides": {}
  }
}
```

Missing file → defaults. `createConfig({ model?, modelId?, backend? })` applies
alias resolution (`gemma` → `gemma-3-270m`) on top of the loaded config.

## Sampling defaults

Per-model defaults (`temperature`, `topK`, `topP`, `repetitionPenalty`) live on
each `ModelDefinition`/`ChatModel`. `temperature <= 0` means greedy decoding in
`sampleLogits()`.
