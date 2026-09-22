# Contributing to jax-llm

Welcome! This document outlines standards, processes, and best practices for
contributing to jax-llm.

## Table of Contents

- [Setup](#setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Commit Standards](#commit-standards)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Documentation](#documentation)
- [Performance](#performance)
- [Security](#security)

---

## Setup

### Prerequisites

- Deno >= 1.40+
- Git 2.30+
- Python 3.9+ (for pre-commit hooks)

### First Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/ricardo/jax-llm.git
cd jax-llm

# 2. Install pre-commit hooks
pip install pre-commit
bash .githooks/setup-hooks.sh

# 3. Verify setup
deno test -A --filter "unit:" | head -20
```

### Verify Installation

```bash
# Check deno
deno --version

# Check git hooks
ls -la .git/hooks/pre-commit .git/hooks/commit-msg

# Run example test
deno test -A tests/issue_1_memory_leak_test.ts
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feat/short-description
# or bugfix
git checkout -b fix/issue-number-short-description
```

**Branch naming:**

- `feat/` for new features
- `fix/` for bug fixes
- `perf/` for performance improvements
- `docs/` for documentation
- `test/` for tests
- `refactor/` for code cleanup
- `chore/` for maintenance

### 2. Make Changes

Write code, tests, and docs following [BEST_PRACTICES.md](./BEST_PRACTICES.md).

### 3. Validate Locally

```bash
# Format code
deno fmt --ext ts,tsx,js,jsx,json,md

# Check formatting
deno fmt --check --ext ts,tsx,js,jsx,json,md

# Lint code
deno lint

# Type check
deno check --all

# Run tests
deno test -A --filter "unit:"
deno test -A tests/issue_*.ts

# Check coverage
deno test --coverage=cov_profile
deno coverage cov_profile --lcov > coverage.lcov
```

### 4. Commit Changes

```bash
# Stage files
git add .

# Commit (hooks run automatically)
git commit -m "type(scope): description"

# Hooks will validate:
# ✅ Code formatting (deno fmt)
# ✅ Linting (deno lint)
# ✅ Type checking (deno check)
# ✅ Unit tests (deno test --filter "unit:")
# ✅ Secrets scanning
# ✅ Commit message format
```

### 5. Push & Create PR

```bash
# Push to remote
git push origin feat/short-description

# Create pull request on GitHub
# Use template below
```

---

## Code Standards

### TypeScript/JavaScript

- **Formatting:** Enforced by `deno fmt`
- **Linting:** Enforced by `deno lint`
- **Type Safety:** Required by `deno check`

```typescript
// ✅ GOOD: Clear, typed, simple
async function loadModel(path: string): Promise<Model> {
  const data = await Deno.readTextFile(path);
  return JSON.parse(data);
}

// ❌ BAD: Unclear intent, implicit any
async function load(p: any) {
  const d = await Deno.readTextFile(p);
  return JSON.parse(d);
}
```

### File Organization

```
engine/
├── llm/                    # Core LLM logic
│   ├── model.ts
│   ├── cache/
│   │   ├── paged_cache.ts
│   │   └── kv_cache.ts
│   └── attention.ts
├── runtime/                # Runtime utilities
├── profiling/              # Profiling & metrics
└── state/                  # State management
```

### Naming Conventions

| Category            | Convention       | Example                             |
| ------------------- | ---------------- | ----------------------------------- |
| Files               | kebab-case       | `paged_cache.ts`, `kv_cache.ts`     |
| Types/Classes       | PascalCase       | `LfmModel`, `PagedCache`            |
| Functions/Variables | camelCase        | `loadModel`, `cacheBuffer`          |
| Constants           | UPPER_SNAKE_CASE | `MAX_BATCH_SIZE`, `DEFAULT_TIMEOUT` |
| Private             | `_prefix`        | `_internalState`                    |

### Comments

**Explain WHY, not WHAT:**

```typescript
// ✅ GOOD: Explains reasoning
// Use LIFO stack instead of single value to handle concurrent
// profiling sessions that may nest (A → B → B.end → A.end)
const timestamps: number[] = [];

// ❌ BAD: States obvious
// Push timestamp
timestamps.push(Date.now());
```

---

## Commit Standards

### Format

```
type(scope): subject

body

footer
```

**Type:** feat, fix, perf, docs, test, refactor, style, chore, ci, build\
**Scope:** Affected module (cache, model, runtime, etc.)\
**Subject:** Max 72 chars, imperative mood, capitalized

**Example:**

```
feat(cache): add paged cache with LRU eviction

Implements on-demand KV cache allocation with 512-token pages
instead of fixed buffers. Pages are evicted using LRU when memory
limit is reached.

Performance impact:
- 8K context memory: 32MB → 8MB (-75%)
- Long sequence handling: now supports 16K+ tokens

Closes #234
Relates to: Phase 3.0 optimization roadmap
```

### Rules

✅ **DO:**

- Use imperative mood: "Add" not "Added"
- Reference issues: "Closes #234"
- Explain WHY not just WHAT
- Keep it atomic (one concern per commit)
- Include performance impact if relevant

❌ **DON'T:**

- Commit without issue reference
- Mix multiple features in one commit
- Write vague messages: "fix bugs", "update code"
- Include debug/console output
- Hardcode credentials or secrets

### Pre-Commit Validation

Automatically runs on `git commit`:

```
✓ Code formatting (deno fmt)
✓ Linting (deno lint)
✓ Type checking (deno check --all)
✓ Unit tests (deno test --filter "unit:")
✓ Secrets scanning
✓ Merge conflict check
✓ Commit message format validation
```

---

## Testing

### Test Structure

```typescript
Deno.test("unit: feature description", async () => {
  // Arrange: Setup test data
  const input = { value: 42 };

  // Act: Execute function
  const result = await functionUnderTest(input);

  // Assert: Verify result
  assertEquals(result.value, 84);
});
```

### Test Coverage

**Minimum requirements:**

- All public APIs must have tests
- Core engine: 70%+ coverage
- Bug fixes must include regression tests
- Performance-critical code needs benchmarks

### Running Tests

```bash
# All tests
deno test -A

# Unit tests only
deno test -A --filter "unit:"

# Specific file
deno test -A tests/issue_1_memory_leak_test.ts

# With coverage
deno test --coverage=cov_profile
deno coverage cov_profile --lcov
```

### Adding Tests for Bugs

When fixing a bug:

```typescript
// 1. Create failing test that demonstrates bug
Deno.test("unit: profiler should handle concurrent sessions", () => {
  // This fails with old code, passes with fix
  const timestamps = new Map();
  // ... concurrency test ...
});

// 2. Fix the bug in source code

// 3. Test passes, commit includes both
git add engine/profiling/profiler.ts tests/issue_4_profiler_race_test.ts
git commit -m "fix(profiler): resolve race condition in concurrent access"
```

---

## Pull Requests

### PR Title Format

Follow conventional commits:

```
type(scope): brief description
```

Examples:

- `feat(cache): add paged cache support`
- `fix(profiler): resolve concurrent timestamp collision`
- `docs: update README with new features`

### PR Template

```markdown
## What

Brief description of changes

## Why

Motivation and context. What problem does this solve?

## Testing

How to verify locally:

- Run: `deno test -A --filter "unit:"`
- Check: Performance metrics
- Verify: No regressions

## Checklist

- [ ] Code follows style guide (deno fmt, deno lint)
- [ ] All tests passing locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or marked with `!`)
- [ ] Commit messages follow conventional format
- [ ] No hardcoded secrets or credentials

## Performance Impact

If applicable:

- TTFT: 2.5s → 1.8s (-28%)
- Memory: 32MB → 8MB (-75%)

## Related Issues

Closes #234 Relates to: Phase 3.0 roadmap
```

### Review Process

1. **Automated checks** run on push (CI/CD)
2. **Code review** by maintainers
3. **Approval** required before merge
4. **Squash merge** to keep history clean

---

## Documentation

### README Requirements

Every feature must have:

- Purpose statement
- Usage example
- Link to detailed docs

```markdown
## Paged Cache

On-demand KV cache allocation with 512-token pages instead of fixed buffers.
Reduces memory usage for long-context inference.

### Usage

\`\`\`typescript const state = createLfmState({ usePagedCache: true, }); \`\`\`

See [Paged Cache Guide](./docs/phase-3-0/PAGED_CACHE.md) for details.
```

### Code Comments

- Explain WHY, not WHAT
- Document complex algorithms
- Link to external references
- Update when logic changes

### Documentation Structure

```
docs/
├── README.md              # Entry point
├── phase-3-0/             # Feature documentation
│   ├── DEPLOYMENT_SUMMARY.md
│   ├── MEDIC_DIAGNOSIS_PHASE_3_0.md
│   └── ...
├── architecture/          # System design docs
├── api/                   # API reference
└── guides/                # How-to guides
```

---

## Performance

### Before Committing Performance Code

- [ ] Profile before and after
- [ ] Document expected improvements
- [ ] Include benchmarks in commit message
- [ ] Avoid premature optimization
- [ ] Consider memory vs speed tradeoff

### Benchmark Format

```
Performance improvements:
- TTFT: 2.5s → 1.8s (-28%)
- Throughput: 3.6 tok/s → 5.2 tok/s (+44%)
- Memory (8K context): 32MB → 8MB (-75%)
```

### Profiling Tools

```typescript
// Enable profiling
engine.enableProfiling();
const result = await engine.chat(messages);
const metrics = engine.getMetrics();

console.log({
  ttft: metrics.ttft, // Time to first token
  tokPerSec: metrics.tokPerSec, // Throughput
  p50: metrics.p50, // Median latency
  p90: metrics.p90, // 90th percentile
  p99: metrics.p99, // 99th percentile
});
```

---

## Security

### Secrets & Credentials

❌ **Never commit:**

- API keys or tokens
- Database passwords
- Private encryption keys
- OAuth secrets
- AWS/GCP credentials

✅ **Instead:**

- Use environment variables
- Store in `.env` (git-ignored)
- Use credential managers
- Rotate tokens regularly

### Secret Scanning

Pre-commit hooks automatically detect and prevent secrets:

```typescript
// ❌ BLOCKED by pre-commit hook
const apiKey = "sk-1234567890abcdef";

// ✅ ALLOWED
const apiKey = Deno.env.get("API_KEY");
```

### Vulnerability Scanning

```bash
# Manual security audit
deno audit

# Check for known vulnerabilities
deno cache --check-all
```

---

## Common Tasks

### Update Dependencies

```bash
# Check for updates
deno cache --check-all

# Update deno.json manually
# Then verify
deno check --all
```

### Fix Formatting Issues

```bash
# Auto-fix formatting
deno fmt --ext ts,tsx,js,jsx,json,md

# Auto-fix linting issues (if applicable)
deno lint  # Review output manually
```

### Debug Tests

```bash
# Run with output
deno test -A --trace-ops tests/file.ts

# Run single test
deno test -A --filter "exact test name" tests/file.ts

# Run with debugging
deno test -A --inspect-brk tests/file.ts
```

---

## Getting Help

- **Questions?** Open a GitHub Discussion
- **Bug?** Open a GitHub Issue with reproduction steps
- **Ideas?** Start a Discussion first, then open an Issue

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intentions
- Focus on the code, not the person
- Help others succeed

---

**Last Updated:** September 22, 2026\
**Version:** 1.0\
**Questions?** See [BEST_PRACTICES.md](./BEST_PRACTICES.md)
