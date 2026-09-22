# Git Hooks & Pre-Commit Validation

Automated quality gates to ensure code, commits, and standards consistency.

## Quick Start

```bash
# 1. Install pre-commit framework
pip install pre-commit

# 2. Install project hooks
bash .githooks/setup-hooks.sh

# 3. Test hooks are working
git diff --cached  # Stage some files
git commit -m "test: verify hooks"  # Should run validations
```

## What Gets Checked?

### Pre-Commit Hook (On `git commit`)

Runs automatically before creating a commit:

✅ **Code Quality**

- Deno formatting (`deno fmt`)
- Linting (`deno lint`)
- Type checking (`deno check --all`)

✅ **Testing**

- Unit test suite (`deno test --filter "unit:"`)

✅ **Security**

- Secrets detection
- Merge conflict checks
- File integrity

✅ **Maintenance**

- Trailing whitespace removal
- Line ending normalization
- YAML/JSON validation

### Commit-Msg Hook (On commit-msg)

Validates commit message format:

✅ **Format**

- Conventional commits: `type(scope): subject`
- Subject ≤ 72 characters
- Body ≤ 100 characters per line

✅ **Content**

- Imperative mood ("Add" not "Added")
- Issue references (closes #123)
- No TODO/FIXME in messages

## Files & Scripts

| File                      | Purpose                            |
| ------------------------- | ---------------------------------- |
| `.pre-commit-config.yaml` | Pre-commit framework configuration |
| `setup-hooks.sh`          | Install hooks into `.git/hooks/`   |
| `pre-commit-validator.sh` | Main pre-commit validation script  |
| `commit-msg-validator.sh` | Commit message validation          |

## Common Issues

### Hooks not running?

```bash
# Reinstall
bash .githooks/setup-hooks.sh

# Verify
ls -la .git/hooks/
```

### Failed validation but I know it's correct?

```bash
# View what failed
git commit -m "message"  # See the error output

# Fix specific issues
deno fmt  # Auto-fix formatting
deno check --all  # Review type errors
deno test -A --filter "unit:"  # Run failing tests

# Skip hooks (not recommended)
git commit --no-verify
```

### Want to debug a hook?

```bash
# Run manually
bash .githooks/pre-commit-validator.sh
bash .githooks/commit-msg-validator.sh .git/COMMIT_EDITMSG
```

## Standards & Guidelines

See [BEST_PRACTICES.md](../BEST_PRACTICES.md) for comprehensive standards on:

- Code style
- Commit format
- Testing requirements
- Documentation
- Security
- Performance

## Bypass (Use with Extreme Caution)

```bash
# Skip all hooks
git commit --no-verify

# This is NOT recommended unless:
# - Emergency hotfix to production
# - All validation issues are tracked
# - You plan to fix them immediately
```

---

**Next:** Read [BEST_PRACTICES.md](../BEST_PRACTICES.md) for detailed standards.
