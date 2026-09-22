# jax-llm Best Practices & Standards

A comprehensive guide to maintaining code quality, consistency, and reliability
across the jax-llm project.

## Table of Contents

1. [Code Style & Formatting](#code-style--formatting)
2. [Commit Standards](#commit-standards)
3. [Testing Requirements](#testing-requirements)
4. [Documentation Standards](#documentation-standards)
5. [Security & Secrets](#security--secrets)
6. [Performance & Optimization](#performance--optimization)
7. [Review Process](#review-process)
8. [Pre-Commit Setup](#pre-commit-setup)

---

## Code Style & Formatting

### TypeScript/JavaScript Standards

**Auto-enforcement via `deno fmt`:**

- Use 2-space indentation
- Single quotes for strings (unless interpolation needed)
- Semicolons required
- Line length: max 100 characters (prefer 80)

```typescript
// ✅ GOOD
const greeting = "Hello, World!";
const message = `Hello, ${name}!`;

// ❌ BAD
const greeting = "Hello, World!";
const message = `Hello, ${name}!`; // Mixed quotes
```

### Running Format Checks

```bash
# Format all files
deno fmt --ext ts,tsx,js,jsx,json,md

# Check without fixing
deno fmt --ext ts,tsx,js,jsx,json,md --check

# Format specific file
deno fmt engine/llm/model.ts
```

### Linting Rules

**Enforced via `deno lint`:**

- No unused variables
- No unused imports
- No console in production (use logger)
- No `any` types (use explicit types)
- No `var` declarations (use `const`/`let`)

```bash
# Run linter
deno lint

# Lint specific file
deno lint engine/llm/model.ts
```

### Type Safety

**Required via `deno check`:**

- All files must pass type checking
- No implicit `any` types
- Proper type annotations on public APIs

```bash
# Type check everything
deno check --all

# Type check specific file
deno check engine/llm/model.ts
```

---

## Commit Standards

### Conventional Commit Format

**Required format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type Categories

| Type       | Usage                                | Example                                      |
| ---------- | ------------------------------------ | -------------------------------------------- |
| `feat`     | New feature                          | `feat(cache): add paged cache support`       |
| `fix`      | Bug fix                              | `fix(profiler): resolve race condition`      |
| `perf`     | Performance improvement              | `perf(decode): optimize buffer pool`         |
| `docs`     | Documentation                        | `docs: update README`                        |
| `test`     | Test additions/fixes                 | `test(cache): add memory leak test`          |
| `refactor` | Code refactoring (no feature change) | `refactor(model): simplify state management` |
| `style`    | Code style (formatting, etc.)        | `style: fix linting issues`                  |
| `chore`    | Maintenance tasks                    | `chore: update dependencies`                 |
| `ci`       | CI/CD changes                        | `ci: add test coverage check`                |
| `build`    | Build system changes                 | `build: update deno.json`                    |

### Commit Message Rules

**✅ DO:**

- Use imperative mood: "Add feature" not "Added feature"
- Start with capital letter after colon
- Keep subject line ≤ 72 characters
- Leave blank line between subject and body
- Reference issues: "closes #123" or "relates to #456"
- Explain WHY, not just WHAT

**❌ DON'T:**

- Commit without issue reference
- Mix multiple features/fixes in one commit
- Write vague messages: "fix bugs", "update code"
- End subject line with period
- Use ALL CAPS

### Example Commit Messages

```
✅ GOOD:
feat(quantization): add INT8 weight loading with FP32 fallback

Automatically detects and loads INT8 quantized models when available,
with graceful fallback to FP32 if variants are missing. This reduces
model size by 50% with <3% latency overhead.

Closes #234
Relates to Phase 3.0 optimization roadmap

❌ BAD:
updated model loading

fix various issues with quantization

Adds support for int8
```

### Atomic Commits

**One feature per commit:**

- Each commit should be independently deployable
- Include related tests in same commit
- Don't mix refactoring with feature work
- Don't mix formatting with logic changes

---

## Testing Requirements

### Test Coverage Standards

**Minimum expectations:**

- Unit tests: 70%+ coverage for core engine
- Integration tests: Critical user paths
- All public APIs must have tests
- Bug fixes must include regression tests

### Test File Organization

```typescript
// test file naming: {module}_test.ts or _test.ts suffix
// Location: tests/ directory
// Naming: use kebab-case for file names

tests/
├── issue_1_memory_leak_test.ts      // Issue-specific tests
├── issue_2_buffer_uaf_test.ts
├── unit/
│   ├── cache_test.ts
│   ├── model_test.ts
│   └── profiler_test.ts
└── integration/
    └── end_to_end_test.ts
```

### Test Structure

```typescript
Deno.test("unit: feature name", async (t) => {
  // Arrange
  const input = setupTestData();

  // Act
  const result = functionUnderTest(input);

  // Assert
  assertEquals(result.value, expected);
});
```

### Running Tests

```bash
# All tests
deno test -A

# Unit tests only
deno test -A --filter "unit:"

# Integration tests
deno test -A --filter "integration:"

# Specific test file
deno test -A tests/issue_1_memory_leak_test.ts

# With coverage
deno test --coverage=cov_profile && deno coverage cov_profile
```

---

## Documentation Standards

### README Requirements

Every public module/feature must have:

- Purpose statement
- Usage examples
- API documentation
- Links to detailed docs

### Code Comments

**Good comments explain WHY:**

```typescript
// ✅ GOOD: Explains reasoning
// Use LIFO stack instead of single value to handle concurrent profiling sessions
// that may nest (session A calls start → session B calls start → B ends → A ends)
const timestamps: number[] = [];

// ❌ BAD: States the obvious
// Push timestamp to array
timestamps.push(Date.now());
```

### Documentation Files

```
docs/
├── phase-3-0/           # Feature documentation
│   ├── DEPLOYMENT_SUMMARY.md
│   ├── MEDIC_DIAGNOSIS_PHASE_3_0.md
│   └── ...
├── architecture/        # System design
├── api/                 # API reference
└── guides/              # How-to guides
```

### Markdown Standards

- Use descriptive headings (h2-h4, not h1)
- Include code examples
- Link related docs
- Keep lines ≤ 100 characters

---

## Security & Secrets

### What's Forbidden

❌ **Never commit:**

- API keys, tokens, credentials
- Database passwords
- Private encryption keys
- OAuth secrets
- AWS/GCP service account keys

### Secret Detection

Pre-commit hook automatically scans for:

- Patterns like `password=`, `apiKey=`, `secret=`
- Common secret formats
- Private key markers

```bash
# Manual secret scan
detect-secrets scan --baseline .secrets.baseline
```

### Safe Practices

✅ **DO:**

- Use environment variables for secrets
- Store secrets in `.env` (git-ignored)
- Use secure credential managers
- Rotate tokens regularly
- Use short-lived credentials

```typescript
// ✅ GOOD: From environment
const apiKey = Deno.env.get("API_KEY");
if (!apiKey) throw new Error("API_KEY not set");

// ❌ BAD: Hardcoded
const apiKey = "sk-1234567890abcdef";
```

---

## Performance & Optimization

### Performance Checklist

Before committing performance-critical code:

- ✅ Profile the code (benchmark before/after)
- ✅ Document expected improvements
- ✅ Avoid premature optimization
- ✅ Use big-O analysis for algorithms
- ✅ Consider memory vs speed tradeoffs

### Profiling Requirements

For Phase 3.0 and beyond:

```typescript
engine.enableProfiling();
const result = await engine.chat(messages);
const metrics = engine.getMetrics();

console.log({
  ttft: metrics.ttft, // Time to first token
  tokPerSec: metrics.tokPerSec, // Throughput
  p90: metrics.p90, // 90th percentile latency
});
```

### Benchmark Format

Include in commit message for performance work:

```
Performance impact:
- TTFT: 2.5s → 1.8s (-28%)
- Throughput: 3.6 tok/s → 5.2 tok/s (+44%)
- Memory: 32MB → 8MB (-75%)
```

---

## Review Process

### Code Review Checklist

**Reviewer should verify:**

- [ ] Code follows style guide (deno fmt, deno lint)
- [ ] All tests passing locally
- [ ] Test coverage maintained/improved
- [ ] No secrets or sensitive data
- [ ] Documentation updated
- [ ] Commit message follows convention
- [ ] No performance regressions
- [ ] No breaking changes (or marked with `!`)

### Pull Request Template

```markdown
## What

Brief description of changes

## Why

Motivation and context

## Testing

How to verify locally:

- Run: `deno test -A --filter "unit:"`
- Check: Performance metrics

## Changes

- [ ] Code changes
- [ ] Tests added/updated
- [ ] Docs updated
- [ ] No breaking changes
- [ ] Performance impact (if applicable)
```

---

## Pre-Commit Setup

### Installation

```bash
# 1. Install pre-commit framework
pip install pre-commit

# 2. Install hooks
cd /Users/ricardo/Projects/jax-llm
bash .githooks/setup-hooks.sh

# 3. Verify installation
git config --list | grep hook
```

### What Gets Validated

On every `git commit`:

1. **Code Formatting** — `deno fmt`
2. **Linting** — `deno lint`
3. **Type Checking** — `deno check --all`
4. **Unit Tests** — `deno test --filter "unit:"`
5. **Secret Scanning** — Prevents hardcoded secrets
6. **Merge Conflicts** — Catches unresolved conflicts
7. **Trailing Whitespace** — Normalizes line endings

On every `git commit --allow-empty` (commit-msg hook):

8. **Commit Message Format** — Conventional commits
9. **Message Length** — Subject ≤ 72 chars, body ≤ 100 chars
10. **Imperative Mood** — "Add" not "Added"
11. **Issue References** — Links to issues/tickets

### Bypass Hooks (Not Recommended)

```bash
# Skip all pre-commit hooks (use with caution!)
git commit --no-verify

# Skip specific hook during setup
# Edit .git/hooks/pre-commit and comment out sections
```

### Troubleshooting

**Hook fails but I know I'm right:**

```bash
# Check hook output
cat .git/hooks/pre-commit

# Debug individual checks
deno fmt --check
deno lint
deno check --all
deno test --filter "unit:"

# Fix issues
deno fmt  # Auto-fix formatting
deno lint # Manual fix required
```

**Hooks not running:**

```bash
# Reinstall hooks
bash .githooks/setup-hooks.sh

# Verify hooks exist and are executable
ls -la .git/hooks/
chmod +x .git/hooks/*
```

---

## Quick Reference

### Before You Commit

```bash
# 1. Auto-fix formatting
deno fmt --ext ts,tsx,js,jsx,json,md

# 2. Check for lint errors
deno lint

# 3. Type check
deno check --all

# 4. Run unit tests
deno test -A --filter "unit:"

# 5. Commit (hooks run automatically)
git add -A
git commit -m "type(scope): subject"
```

### Common Commands

```bash
# Format all code
deno fmt --ext ts,tsx,js,jsx,json,md

# Lint all code
deno lint

# Run all tests
deno test -A

# Run specific tests
deno test -A tests/issue_1_memory_leak_test.ts

# Check git hooks status
git config --list | grep hook
```

---

## Useful Links

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Deno Documentation](https://docs.deno.com/)
- [Pre-commit Framework](https://pre-commit.com/)
- [Git Hooks Guide](https://git-scm.com/docs/githooks)

---

**Last Updated:** September 22, 2026\
**Version:** 1.0\
**Owner:** jax-llm Maintainers
