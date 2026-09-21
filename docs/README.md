# Documentation

Start here, or jump to a topic:

- [Architecture](architecture.md) — runtime vs. chat layers, inference flow,
  registry, tokenizer, memory management.
- [API reference](api.md) — all 60 exported symbols, grouped by layer.
- [Models](models.md) — built-in models, aliases, Hugging Face repos, URL
  overrides.
- [Configuration](configuration.md) — `ChatEngineOptions`, `RuntimeConfig`,
  `config.json`, environment variables.
- [Training](training.md) — optimizers, loss, `TrainingRunner`.
- [Development](development.md) — tasks, tests, doc coverage, JSR publishing.

Source JSDoc on every export is the source of truth (`deno doc mod.ts`); these
pages are narrative guides over the same API.
