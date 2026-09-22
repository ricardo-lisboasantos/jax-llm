# jax-llm Standardization & Quality Gates

Complete reference for project standards, pre-commit validation, and
contribution guidelines.

## 🎯 Quick Start (New Contributors)

```bash
# 1. Clone and setup
git clone https://github.com/ricardo/jax-llm.git
cd jax-llm

# 2. Install pre-commit hooks
pip install pre-commit
bash .githooks/setup-hooks.sh

# 3. Verify it works
git diff --cached
git commit -m "test: verify hooks"

# 4. Read the standards
cat BEST_PRACTICES.md
cat CONTRIBUTING.md
```

---

## 📋 Core Standards

### Code Quality

| Tool           | Standard                       | Enforcement        |
| -------------- | ------------------------------ | ------------------ |
| **deno fmt**   | 2-space indent, single quotes  | ✅ Pre-commit hook |
| **deno lint**  | No unused vars, no `any` types | ✅ Pre-commit hook |
| **deno check** | All types must resolve         | ✅ Pre-commit hook |
| **deno test**  | 70%+ coverage, unit tests pass | ✅ Pre-commit hook |

### Commit Messages

**Format:** `type(scope): subject`

**Types:**

- `feat` → New feature
- `fix` → Bug fix
- `perf` → Performance improvement
- `docs` → Documentation
- `test` → Test additions
- `chore` → Maintenance
- `ci` → CI/CD changes

**Rules:**

- ✅ Subject ≤ 72 characters
- ✅ Imperative mood ("Add" not "Added")
- ✅ Reference issues: "closes #123"
- ❌ No vague messages: "fix bugs", "update code"

**Example:**

```
feat(cache): add paged cache with LRU eviction

Implements on-demand KV cache allocation with 512-token pages.
Pages are evicted using LRU when memory limit is reached.

Performance impact:
- Memory: 32MB → 8MB (-75%)
- Context support: 8K → 16K+ tokens

Closes #234
```

---

## 🔧 Pre-Commit Validation System

### What Gets Checked

```
On every 'git commit':

CODE QUALITY (auto-fail if broken):
  ✅ deno fmt          → Format check (auto-fixes possible)
  ✅ deno lint         → Linting (0 errors required)
  ✅ deno check --all  → Type safety (all types resolved)
  ✅ deno test         → Unit tests (must pass)

SECURITY & INTEGRITY:
  ✅ detect-secrets    → Blocks hardcoded credentials
  ✅ check-merge-conflict → Prevents unresolved conflicts
  ✅ trailing-whitespace → Normalizes line endings
  ✅ end-of-file-fixer → Adds final newlines
  ✅ check-yaml/json   → Validates config files

COMMIT MESSAGE (auto-fail if invalid):
  ✅ Conventional commit format (type(scope): subject)
  ✅ Subject length ≤ 72 characters
  ✅ Imperative mood ("Add" not "Added")
  ✅ Issue reference (closes #123 or relates to #456)
  ✅ No TODO/FIXME in message
```

### Files & Scripts

```
.githooks/
├── setup-hooks.sh            # Install into .git/hooks/
├── pre-commit-validator.sh   # Main validation (10 checks)
├── commit-msg-validator.sh   # Message format validation (10 checks)
└── README.md                 # Quick reference

.pre-commit-config.yaml       # Pre-commit framework configuration

BEST_PRACTICES.md             # Comprehensive standards guide (400+ lines)
CONTRIBUTING.md               # Contributor workflow & guidelines
```

---

## 🚀 Workflow

### For Every Commit

```bash
# 1. Create branch
git checkout -b feat/your-feature

# 2. Make changes & add tests
# ... code, tests, docs ...

# 3. Format & validate locally
deno fmt
deno lint
deno check --all
deno test -A --filter "unit:"

# 4. Commit (hooks run automatically)
git add .
git commit -m "feat(scope): your feature"

# Hooks validate:
# - Code formatting ✅
# - Linting ✅
# - Type checking ✅
# - Unit tests ✅
# - Secrets scanning ✅
# - Commit message format ✅

# 5. Push & create PR
git push origin feat/your-feature
# Open PR on GitHub with template
```

### Pre-Commit Hook Output

**Success (all green):**

```
╔════════════════════════════════════════════════════════════╗
║            PRE-COMMIT VALIDATION SUITE                     ║
╚════════════════════════════════════════════════════════════╝

→ Checking code formatting (deno fmt)...
  ✅ All files properly formatted

→ Linting code (deno lint)...
  ✅ No lint errors

→ Type checking (deno check)...
  ✅ All types valid

→ Running unit tests...
  ✅ All unit tests passing

...

╔════════════════════════════════════════════════════════════╗
║ ✅ ALL VALIDATIONS PASSED - READY TO COMMIT               ║
╚════════════════════════════════════════════════════════════╝
```

**Failure (must fix):**

```
→ Checking code formatting (deno fmt)...
  ❌ Code formatting issues found
     Run: deno fmt

→ Linting code (deno lint)...
  ❌ Lint errors found
     Run: deno lint

❌ SOME VALIDATIONS FAILED - FIX BEFORE COMMITTING
```

### Bypassing Hooks (Emergency Only)

```bash
# Skip all validation hooks (NOT RECOMMENDED)
git commit --no-verify

# Use only for:
# - Critical production hotfixes
# - CI/CD failures
# - Emergency reverts
# Plan to fix immediately afterward
```

---

## 📚 Complete Guides

### Quick Reference

- **[BEST_PRACTICES.md](./BEST_PRACTICES.md)** (400+ lines)
  - Code style standards
  - Commit message format
  - Testing requirements
  - Documentation standards
  - Security & secrets protection
  - Performance guidelines
  - Review process

- **[CONTRIBUTING.md](./CONTRIBUTING.md)** (500+ lines)
  - Setup & prerequisites
  - Development workflow
  - Code standards & conventions
  - Testing structure
  - Pull request template
  - Documentation requirements
  - Common tasks & debugging

- **[.githooks/README.md](./.githooks/README.md)**
  - Quick start
  - What gets checked
  - Common issues & fixes
  - Troubleshooting

---

## ✅ Validation Checklist

Before pushing code:

```
Code Quality:
☐ deno fmt --check (formatting OK)
☐ deno lint (0 errors)
☐ deno check --all (all types resolved)
☐ deno test -A --filter "unit:" (all pass)

Testing:
☐ Unit tests added for new code
☐ Bug fixes include regression tests
☐ 70%+ coverage maintained/improved

Documentation:
☐ README updated (if new feature)
☐ Code comments explain WHY
☐ Complex logic documented
☐ Examples provided

Commit:
☐ One feature per commit (atomic)
☐ Conventional format: type(scope): subject
☐ Subject ≤ 72 characters
☐ Issue reference (closes #123)
☐ Body explains WHY, not WHAT

Security:
☐ No hardcoded secrets
☐ No debug code (console.log, debugger)
☐ No TODO/FIXME in commit message
☐ No private keys or credentials

Performance:
☐ Profile before/after (if perf-critical)
☐ Document performance impact
☐ Benchmark included in message
```

---

## 🔒 Security Gates

### Secrets Detection

Pre-commit automatically blocks:

- API keys (`apiKey=`, `api_key=`)
- Passwords (`password=`, `passwd=`)
- Tokens (`token=`, `secret=`)
- Private keys (`-----BEGIN PRIVATE KEY-----`)

```typescript
// ❌ BLOCKED by pre-commit
const apiKey = "sk-1234567890abcdef";

// ✅ ALLOWED
const apiKey = Deno.env.get("API_KEY");
```

---

## 📊 Project Statistics

**Version:** 0.4.0\
**Last Updated:** September 22, 2026

### Code Quality

- ✅ deno fmt: 206 files
- ✅ deno lint: 0 errors
- ✅ deno check: All types valid
- ✅ Unit tests: 138 tests passing
- ✅ Coverage: 70%+ (core engine)

### Standards Coverage

- ✅ Best practices guide: 400+ lines
- ✅ Contributing guide: 500+ lines
- ✅ Pre-commit validation: 10+ checks
- ✅ Commit message validation: 10+ rules
- ✅ Documentation: Complete

---

## 🆘 Troubleshooting

### Hooks not running?

```bash
# Reinstall
bash .githooks/setup-hooks.sh

# Verify
ls -la .git/hooks/pre-commit .git/hooks/commit-msg

# Make executable
chmod +x .git/hooks/*
```

### Failed validation?

```bash
# See what failed
git commit -m "message"  # Shows error output

# Fix formatting
deno fmt

# Fix linting
deno lint  # Review output, fix manually

# Fix type errors
deno check --all

# Run failing tests
deno test -A tests/issue_*.ts
```

### Commit message validation?

```bash
# Message must follow format:
git commit -m "type(scope): Subject

Body text...

Closes #123"

# Common mistakes:
❌ "feat: subject"              # lowercase s
❌ "Feature name goes here"     # missing type
❌ "Fix various bugs"           # vague message
✅ "feat(cache): Add paged cache support"
```

---

## 📞 Getting Help

1. Read [BEST_PRACTICES.md](./BEST_PRACTICES.md) for detailed standards
2. Read [CONTRIBUTING.md](./CONTRIBUTING.md) for workflow
3. Read [.githooks/README.md](./.githooks/README.md) for hook issues
4. Run hooks manually to debug: `bash .githooks/pre-commit-validator.sh`

---

## 🎓 Key Takeaways

✅ **Setup once:** `bash .githooks/setup-hooks.sh`\
✅ **Auto-validated:** All commits validated automatically\
✅ **Educational:** Hooks show what failed and why\
✅ **Reversible:** Each validation is isolated and can be fixed\
✅ **Team aligned:** Everyone follows same standards\
✅ **Production ready:** Prevents bad code from shipping

---

**Next:** Read [CONTRIBUTING.md](./CONTRIBUTING.md) to start contributing!

---

**Documentation Version:** 1.0\
**Last Updated:** September 22, 2026\
**Owner:** jax-llm Maintainers
