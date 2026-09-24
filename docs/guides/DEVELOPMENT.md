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

Defined in `.github/workflows/`. Branch model: `dev` for development, `main` for
production only (branch-protected: pull request + green `CI gate` required, no
direct pushes, no deletions, enforced for admins). CI triggers on pushes to
`main`/`dev`, pull requests targeting either, and manually via
`workflow_dispatch`.

Pipeline layout (all `ubuntu-latest` per the cost guard):

- `ci.yml` — verification only, never publishes:
  1. `conventional` (PRs): PR title must be a conventional commit.
  2. `cost-guard`: every workflow must use stock `ubuntu-latest` runners.
  3. `verify`: the shared gate (`.github/actions/verify`) — `deno audit`,
     `fmt --check`, `lint`, `check --all`, JSDoc `missing-jsdoc` only, full
     `deno test -A` with coverage artifact, `npx jsr publish --dry-run`.
  4. `gate` (`CI gate`, the required check): single pass/fail summary.
- `release.yml` — the **Release button** (Actions → Release → Use workflow from
  `main` → Run workflow; refuses any other branch): `plan` (resolve version,
  gather commits since last tag, render notes) → `verify` (shared gate on
  `main`) → `propose` (stamp `deno.json` + `CHANGELOG.md` +
  `docs/releases/vX.Y.Z.md`, re-verify the stamp without the test suite, open a
  `release/vX.Y.Z` PR labeled `release`). Supports `dry_run` previews. Full
  process: `docs/guides/RELEASING.md`.
- `release-publish.yml` — runs when a `release`-labeled PR merges to `main`:
  shared gate on the merged tree (including the `deno.json` version match),
  pushes the `vX.Y.Z` tag, `npx jsr publish` via OIDC, creates the GitHub
  Release with notes + manifest artifacts. Forks excluded; no direct pushes
  required, so `main` stays fully branch-protected.
- `sync-dev.yml` — on every push to `main`, merges `main` back into `dev`
  (version stamps and hotfixes flow back automatically); opens a `sync` PR on
  conflict instead of failing silently.

The gate sequence is defined once in `.github/actions/verify` and reused by CI
and all release jobs — fix it there, not in each workflow. Least privilege:
top-level permissions are read-only; jobs grant only what they need
(`contents:write` for pushes and artifact uploads, `pull-requests:write` for PR
creation, `id-token:write` for JSR publishing).

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
coverage — CI enforces only `missing-jsdoc` (see `.github/workflows/ci.yml`).

## Publishing

- `deno.json` publishes `mod.ts`, `engine/`, `README.md`, `LICENSE` to
  `jsr:@ricardo/jax-llm`; tests, `cli`, and `docs/` are excluded.
- Before publishing: `deno fmt`, `deno task lint`, `deno task test:unit`,
  `deno doc --lint mod.ts`.
- `docs/` and this README are the narrative layer; per-symbol JSDoc is the
  source of truth rendered on the JSR page.
