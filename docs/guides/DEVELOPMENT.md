# Development

## Tasks

Defined in `deno.json`:

| Command                      | Purpose                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `deno task test`             | All tests (needs `--allow-net` for integration)               |
| `deno task test:unit`        | Unit tests only (`--filter unit`, offline-safe)               |
| `deno task test:integration` | Integration tests (`--filter integration`, downloads weights) |
| `deno task test:coverage`    | Coverage into `cov_profile.lcov`                              |
| `deno task dev`              | `deno test --watch`                                           |
| `deno task lint`             | `deno lint`                                                   |
| `deno fmt`                   | Format                                                        |

## Tests

- Unit tests live next to their modules: `tokenizer_test.ts`, `sampler_test.ts`,
  `prompt_test.ts`, `registry_test.ts`, `runtime_test.ts`, `model_test.ts`,
  `gemma_test.ts`, `lfm_test.ts`, `chat_test.ts`, `database_test.ts`,
  `kv_test.ts`, `config_test.ts`, `caches_test.ts`, `lora_test.ts`.
- End-to-end tests live in `tests/integration_test.ts` (network-gated).
- `deno task test:unit` passes 79 tests.
- Permission-gated tests (`config_test.ts` needs env, `lora_test.ts` needs
  write, integration needs net) skip gracefully when their permission is denied,
  so bare `deno test` stays green; `deno task test` (full flags) and CI
  (`deno test -A`) execute everything.

## CI/CD

Defined in `.github/workflows/ci.yml`. Triggers on every push, every pull
request, and manually via `workflow_dispatch`.

- `verify` job (Deno v2.x via `denoland/setup-deno@v2`), steps run in order and
  stop on first failure:
  1. `deno fmt --check` — formatting must already be clean (run `deno fmt`
     locally).
  2. `deno lint`.
  3. `deno check --all` — full type check.
  4. `deno test -A` — entire suite, including the network integration test.
- `publish` job (`needs: verify`, only for pushes to `main`, never PRs):
  `npx jsr publish` using OIDC trusted publishing (`id-token: write`), so no
  token is stored in repo secrets. Republishing an unchanged version is a
  harmless no-op (`jsr` reports "already published").

## Documentation coverage

JSR requires ≥80% of exported symbols to carry JSDoc. This package is at 100% —
`deno doc --lint mod.ts` reports 0 `missing-jsdoc` errors. Keep it that way:

```bash
deno doc --lint mod.ts
deno doc mod.ts  # rendered API preview
```

Rules: every `export` (including class members) needs a directly attached
`/** ... */` comment. The remaining `deno doc --lint` findings are
`private-type-ref` (slow-types) notices, which are tracked separately from doc
coverage.

## Publishing

- `deno.json` publishes `mod.ts`, `engine/`, `README.md`, `LICENSE` to
  `jsr:@ricardo/jax-llm`; tests, `cli`, and `docs/` are excluded.
- Before publishing: `deno fmt`, `deno task lint`, `deno task test:unit`,
  `deno doc --lint mod.ts`.
- `docs/` and this README are the narrative layer; per-symbol JSDoc is the
  source of truth rendered on the JSR page.
