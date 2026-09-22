# AI-Assisted Code Disclaimer

Parts of `jax-llm` (`@ricardo/jax-llm`) were written with AI coding assistance,
under human direction and review.

**What this means:**

- Human maintainers own all final decisions — design, merges, releases.
- AI suggestions may contain mistakes, insecure patterns, or outdated APIs.
  Treat them like any third-party contribution: review before use.
- No additional warranty beyond the MIT `LICENSE`. See `LICENSE` for the full
  "AS IS" terms.
- Training-data overlap / IP: if you believe any portion infringes your rights,
  open an issue with file + lines and we will promptly review / replace it.
- Security-sensitive use (prod inference, untrusted input, weights from Hugging
  Face): audit `engine/` and pin versions. Do not rely on AI-generated comments
  as spec — `deno doc mod.ts` + JSDoc + tests are source of truth.

**For contributors:** you may use AI tools, but you are responsible for what you
submit — test it (`deno task test:unit`), lint it (`deno lint`), and disclose
significant AI-generated blocks in your PR description.

Last updated: 2026-09-22 / v0.4.1
