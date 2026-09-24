## What

Brief description of changes.

## Why

Motivation and context. What problem does this solve?

## Testing

How to verify locally:

- Run: `deno test -A`
- Check: performance metrics (if applicable)
- Verify: no regressions

## Checklist

- [ ] Code follows style guide (`deno fmt`, `deno lint`, `deno check --all`)
- [ ] All tests passing locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or marked with `!` and noted below)
- [ ] Commit messages follow conventional format (`type(scope): subject`)
- [ ] No hardcoded secrets or credentials

## Performance impact

If applicable:

- TTFT: 2.5s → 1.8s (-28%)
- Memory: 32MB → 8MB (-75%)

## Release notes

Should this appear in the changelog? If yes, summarize in one line using
conventional-commit style (e.g. `feat(cache): add paged cache support`).

## Related issues

Closes #___ / Relates to: ___
