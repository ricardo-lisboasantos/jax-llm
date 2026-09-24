# Releasing

Single source of truth for how this repo ships versions. Read this before
cutting a release.

## Versioning

- **Source of truth:** `version` in `deno.json` (published to
  `jsr:@ricardo/jax-llm`).
- **Tags:** `vX.Y.Z`, always equal to `deno.json` at the tagged commit.
- **Scheme:** SemVer — `major` (breaking), `minor` (features), `patch` (fixes).
  Commit types map to bumps the same way (`feat` → minor, `fix`/`perf` → patch).
- **Changelog:** `CHANGELOG.md` (machine-prepended per release) plus a
  per-release narrative in `docs/releases/vX.Y.Z.md`. Both are generated from
  conventional commits — do not hand-edit release sections.

## Branch model

- `dev` — day-to-day development. All feature/fix branches target `dev`.
- `main` — production only, branch-protected (pull request + green `CI gate`
  required, no direct pushes, no deletions, no force-pushes — enforced for
  admins too). `dev` reaches `main` via pull request; release stamps land via a
  `release/vX.Y.Z` pull request. Never commit directly to `main`.
- After any push to `main`, `sync-dev` merges it back into `dev` automatically
  (release stamps included), so the branches never diverge.

## The Release button (standard path)

Releasing is two stages — propose, then merge to publish:

1. Merge `dev` → `main` via pull request (CI must be green on both sides).
2. GitHub → **Actions** → **Release** → **Use workflow from: `main`** → **Run
   workflow** (the workflow refuses to run from any other branch).
3. Pick inputs:
   - `bump`: `patch` | `minor` | `major` | `explicit`.
   - `version`: required only for `explicit` (e.g. `0.5.0`).
   - `dry_run`: `true` first when unsure — runs plan + verify + stamping and
     stops before opening the release PR.
   - `prerelease`: adds the `pre-release` label so the eventual GitHub Release
     is marked pre-release.
4. The run opens a `release/vX.Y.Z` pull request to `main` (label: `release`)
   with the generated changelog as its body. CI runs the full gate on it.
5. Review and **merge** the release PR. Merging triggers `release-publish`,
   which verifies the merged tree, pushes the `vX.Y.Z` tag, publishes to JSR via
   OIDC (no stored token), and creates the GitHub Release with
   `docs/releases/vX.Y.Z.md`, `release-manifest.json`, and `CHANGELOG.md`
   attached.
6. After merge: `deno.json`, `CHANGELOG.md`, and `docs/releases/vX.Y.Z.md` on
   `main` already reflect the release. Nothing else to do manually.

Outputs:

| Artifact               | Where                                             |
| ---------------------- | ------------------------------------------------- |
| JSR package            | `https://jsr.io/@ricardo/jax-llm@X.Y.Z`           |
| GitHub Release + notes | `Releases → vX.Y.Z` (body = generated changelog)  |
| Manifest               | `release-manifest.json` (version, SHAs, subjects) |
| Workflow artifacts     | Actions run → Artifacts (90-day retention)        |

## Rules

- Never publish from a laptop — only `release-publish` (triggered by merging a
  `release/` PR) publishes to JSR. This keeps OIDC provenance, changelog, tag,
  and release notes consistent.
- Never push a `v*` tag manually — tags are created by `release-publish` after
  the stamp merge. A manually pushed tag publishes nothing.
- Never hand-edit a published `CHANGELOG.md` section or `docs/releases/v*` file
  — fix forward in the next release.
- Every PR title must be a conventional commit (`feat: …`, `fix(scope): …`); CI
  lints this and the changelog groups depend on it.
- `main` is always releasable: CI runs the full gate plus a JSR dry-run on every
  push and PR.

## Local preview (same generator CI uses)

```bash
# What would the next patch release contain?
deno task release:preview

# Explicit version preview:
deno run -A scripts/generate_changelog.ts --version 0.5.0

# Full stamp preview without publishing (writes CHANGELOG + docs file):
deno run -A scripts/generate_changelog.ts --version 0.5.0 --write-changelog --write-docs
git diff --stat  # inspect, then `git checkout --` to discard
```

## Rules

- Never publish from a laptop — only the `Release` workflow publishes to JSR.
  This keeps OIDC provenance, changelog, tag, and release notes consistent.
- Never hand-edit a published `CHANGELOG.md` section or `docs/releases/v*` file
  — fix forward in the next release.
- Every PR title must be a conventional commit (`feat: …`, `fix(scope): …`); CI
  lints this and the changelog groups depend on it.
- `main` is always releasable: CI runs the full gate plus a JSR dry-run on every
  push and PR.
