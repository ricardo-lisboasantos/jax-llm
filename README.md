# @ricardo/jax-llm

A JAX‑based LLM engine for JavaScript/TypeScript, designed for Deno.

## Installation

```bash
# Using Deno
deno add jsr:@ricardo/jax-llm
```

## Quick Start

### Using the Library

The simplest API — just pass a model name and call `chat()`:

```typescript
import { ChatEngine } from "@ricardo/jax-llm";

const engine = new ChatEngine("lfm2.5-350m");
await engine.init();
const response = await engine.chat("Hello, how are you?");
console.log(response);
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

Streaming responses:

```typescript
for await (
  const chunk of engine.chatStream([
    { role: "user", content: "Tell me a story" },
  ])
) {
  process.stdout.write(chunk);
}
```

Model names can be:

- **Built-in IDs**: `"lfm2.5-350m"`, `"gemma-3-270m"`
- **Short aliases**: `"lfm"`, `"gemma"`
- **HuggingFace repos**: `"org/model-name"` (loaded dynamically)```

### Using the CLI

```bash
deno run -A cli/main.ts
```

The CLI includes an interactive chat interface with support for commands like
`/clear`, `/model`, and `/help`.

## Architecture

The codebase uses a **two-layer abstraction**:

- **Runtime layer** (`engine/runtime/`): Low-level JAX-JS concerns — device
  init, tokenizer/weight loading, JIT-compiled inference sessions, memory
  management, and optax-based training.
- **Chat layer** (`engine/chat/`): High-level API — `ChatEngine` is a thin
  facade that delegates to the runtime. Devs just pass a model name and call
  `engine.chat(input)`.

See the [Architecture Guide](docs/wiki/architecture.md) for details.

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
deno task test

# Run unit tests only
deno task test:unit

# Run integration tests (requires network access)
deno task test:integration

# Run tests with coverage report
deno task test:coverage

# Run tests in watch mode during development
deno task dev
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
  - `engine/tokenizer/tokenizer_test.ts` - Tokenizer encoding/decoding
  - `engine/llm/model_test.ts` - Model definitions and prompt formatting
  - `engine/llm/gemma_test.ts` - Gemma model configuration and state
  - `engine/llm/lfm_test.ts` - LFM model configuration and state
  - `engine/llm/chat_test.ts` - Chat engine API
  - `engine/chat/sampler_test.ts` - Logit sampling (temperature, topK, topP)
  - `engine/chat/prompt_test.ts` - Prompt formatting per model family
  - `engine/runtime/registry_test.ts` - Model registry & resolution
  - `engine/runtime/runtime_test.ts` - ModelRuntime lifecycle
  - `engine/database/database_test.ts` - Database interface
  - `engine/database/kv_test.ts` - Deno KV implementation
  - `engine/config_test.ts` - Configuration handling

- **Integration Tests**: Test end-to-end functionality
  - `integration_test.ts` - Full engine initialization and model downloads

> Integration tests require network access and download model weights. They skip
> automatically when `--allow-net` is not granted, so `deno test` and
> `deno task dev` run clean offline. Run them explicitly with
> `deno task test:integration`.

## Development

This project is managed with Deno tasks. The most useful commands are:

- **\`deno fmt\`** – Format all source files according to the shared Prettier
  configuration.
- **\`deno lint\`** – Run the built‑in Deno linter; fails on style warnings or
  potential bugs.
- **\`deno typecheck\`** – Run a full TypeScript type‑check (no compilation step
  required).
- **\`deno prepare\`** – Run formatter, linter, and unit tests in one step
  (ideal for pre‑commit checks).
- **\`deno test\`** – Execute the full test suite.
- **\`deno test:unit\`** – Run only unit tests.
- **\`deno test:integration\`** – Run integration tests (requires network
  access; disabled by default).
- **\`deno test:coverage\`** – Generate a code‑coverage report and upload it to
  Codecov.
- **\`deno dev\`** – Watch mode for rapid development; re‑runs tests on file
  changes.

### Configuration

The library optionally reads a `config.json` at the project root. You can use it
to override:

- \`chat.backend\` – Choose `"webgpu"` (default) or `"wasm"`.
- \`chat.modelId\` – The identifier of the model to load.
- \`chat.maxTokens\` – Maximum token count for generated responses.
- \`chat.modelOverrides\` – Custom URLs for weights or tokenizer files.

A sample `config.json` is included in the repository. All fields are optional;
missing values fall back to the defaults defined in \`engine/config.ts\`.

### Examples

A collection of ready‑to‑run examples lives in the \`examples/\` folder. The
simplest chat demo is \`examples/chat.ts\`.

## Documentation

Detailed API documentation can be generated with Doxygen or TypeDoc if desired,
but the source code is heavily commented and the TypeScript types provide
sufficient guidance for most use cases.

For a high‑level overview of architecture, see the
[Architecture Guide](docs/wiki/architecture.md).

## License

MIT

## Contributing

Thank you for considering contributing to this project! 🙌

- **Code of Conduct** – This project follows the
  [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
- **Security** – If you discover a security vulnerability, please email
  **security@ricardo.dev** (see [SECURITY.md](SECURITY.md) for details).
- **Contributing Guide** – Review the [CONTRIBUTING.md](CONTRIBUTING.md) for
  development setup, testing, and PR workflow.

All contributions are welcome: bug fixes, feature implementations, documentation
improvements, and new model integrations.

## Continuous Integration

The repository includes robust CI pipelines:

- **CI** – Runs on every push/PR: `deno fmt --check`, `deno lint`,
  type‑checking, unit + integration tests, coverage upload to Codecov, and
  dependency vulnerability scanning.
- **Release** – On pushes to `main`, a GitHub Release is automatically created
  with a generated changelog via `standard-version`.

CI status badges can be found in the GitHub Actions workflow files under
`.github/workflows/ci.yml` and `release.yml`.

## Badges

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deno Version](https://img.shields.io/badge/deno-1.38%2B-brightgreen.svg)](https://deno.land/)
[![CI Status](https://github.com/ricardo/jax-js-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardo/jax-js-llm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ricardo/jax-js-llm?include_prereleases&label=release)](https://github.com/ricardo/jax-js-llm/releases)
