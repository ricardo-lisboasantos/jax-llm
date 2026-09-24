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
- `main` — production only. `dev` reaches `main` via pull request, and only the
  `Release` workflow writes to `main` otherwise (version-stamp commit + tag).
  Never commit directly to `main`.

## The Release button (standard path)

1. Merge `dev` → `main` via pull request (CI must be green on both sides).
2. GitHub → **Actions** → **Release** → **Use workflow from: `main`** → **Run
   workflow** (the workflow refuses to run from any other branch).
3. Pick inputs:
   - `bump`: `patch` | `minor` | `major` | `explicit`.
   - `version`: required only for `explicit` (e.g. `0.5.0`).
   - `dry_run`: `true` first when unsure — runs plan + verify + stamping,
     uploads the notes as artifacts, and stops before JSR/tag/Release.
   - `prerelease`: marks the GitHub Release as pre-release.
4. Watch the run **Summary**:
   - `Plan release` shows resolved version, previous tag, and full notes.
   - `Verify` must be green (fmt, lint, types, docs, tests, JSR dry-run).
   - `Publish` stamps files, publishes to JSR via OIDC (no stored token), pushes
     the `chore(release): vX.Y.Z` commit + `vX.Y.Z` tag, and creates the GitHub
     Release with `NOTES.md`, `release-manifest.json`, and `CHANGELOG.md`
     attached.
5. After merge-back: `deno.json`, `CHANGELOG.md`, and `docs/releases/vX.Y.Z.md`
   on `main` already reflect the release. Nothing else to do manually.

Outputs:

| Artifact               | Where                                             |
| ---------------------- | ------------------------------------------------- |
| JSR package            | `https://jsr.io/@ricardo/jax-llm@X.Y.Z`           |
| GitHub Release + notes | `Releases → vX.Y.Z` (body = generated changelog)  |
| Manifest               | `release-manifest.json` (version, SHAs, subjects) |
| Workflow artifacts     | Actions run → Artifacts (90-day retention)        |

## Tag-driven releases (alternative)

Pushing a `v*.*.*` tag runs the same pipeline pinned to that version:

```bash
git tag -a v0.5.0 -m "Release v0.5.0"
git push origin v0.5.0
```

No bump commit is pushed in this mode — make sure `deno.json` already carries
the tagged version, otherwise JSR and the tag disagree.

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
