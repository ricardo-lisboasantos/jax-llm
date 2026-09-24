/**
 * Generate release notes from conventional commits since the last release tag.
 *
 * Usage (local preview — same command the release workflow runs):
 * ```bash
 * deno run -A scripts/generate_changelog.ts --from v0.4.0 --to HEAD --version 0.5.0
 * deno run -A scripts/generate_changelog.ts --write-changelog --version 0.5.0
 * ```
 *
 * Behavior:
 * - When `--from` is omitted, the latest `v*` git tag is used. When no tag
 *   exists, all history is included (first release bootstrap).
 * - Commits are grouped by conventional-commit type (feat, fix, perf, …).
 *   Non-conforming subjects fall under "Other changes".
 * - Output is deterministic markdown suitable for CHANGELOG.md, GitHub
 *   Releases, and `docs/releases/vX.Y.Z.md`.
 */

const GROUPS: Array<{ key: string; title: string; types: string[] }> = [
  { key: "feat", title: "✨ Features", types: ["feat"] },
  { key: "fix", title: "🐛 Fixes", types: ["fix"] },
  { key: "perf", title: "⚡ Performance", types: ["perf"] },
  { key: "docs", title: "📚 Documentation", types: ["docs"] },
  { key: "refactor", title: "♻️ Refactors", types: ["refactor"] },
  { key: "test", title: "🧪 Tests", types: ["test"] },
  { key: "build", title: "📦 Build & Deps", types: ["build"] },
  { key: "ci", title: "🤖 CI/CD", types: ["ci"] },
  { key: "chore", title: "🧹 Maintenance", types: ["chore", "style"] },
];

interface Commit {
  sha: string;
  subject: string;
  body: string;
  type: string | null;
  scope: string | null;
  description: string;
  breaking: boolean;
}

function parseArgs(args: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (!a.startsWith("--")) continue;
    const eq = a.indexOf("=");
    if (eq !== -1) {
      out[a.slice(2, eq)] = a.slice(eq + 1);
    } else if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
      out[a.slice(2)] = args[++i];
    } else {
      out[a.slice(2)] = true;
    }
  }
  return out;
}

async function git(...args: string[]): Promise<string> {
  const cmd = new Deno.Command("git", {
    args,
    stdout: "piped",
    stderr: "piped",
  });
  const { code, stdout, stderr } = await cmd.output();
  if (code !== 0) {
    const err = new TextDecoder().decode(stderr).trim();
    throw new Error(`git ${args.join(" ")} failed: ${err}`);
  }
  return new TextDecoder().decode(stdout).trim();
}

async function latestTag(): Promise<string | null> {
  try {
    const tags = await git("tag", "--list", "v*", "--sort=-v:refname");
    const first = tags.split("\n").map((t) => t.trim()).filter(Boolean)[0];
    return first ?? null;
  } catch {
    return null;
  }
}

async function currentVersion(): Promise<string> {
  const raw = await Deno.readTextFile("deno.json");
  const json = JSON.parse(raw) as { version?: string };
  if (!json.version) throw new Error("deno.json has no version field");
  return json.version;
}

export function bumpVersion(current: string, bump: string): string {
  const m = current.replace(/^v/, "").match(/^(\d+)\.(\d+)\.(\d+)(.*)$/);
  if (!m) throw new Error(`Cannot parse semver: ${current}`);
  let [major, minor, patch] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const suffix = m[4] ?? "";
  if (/^\d+\.\d+\.\d+/.test(bump)) return bump.replace(/^v/, "");
  switch (bump) {
    case "major":
      major += 1;
      minor = 0;
      patch = 0;
      break;
    case "minor":
      minor += 1;
      patch = 0;
      break;
    case "patch":
      patch += 1;
      break;
    default:
      throw new Error(
        `Unknown bump "${bump}" (expected major|minor|patch|X.Y.Z)`,
      );
  }
  void suffix;
  return `${major}.${minor}.${patch}`;
}

function parseCommit(sha: string, subject: string, body: string): Commit {
  const clean = subject.trim();
  const match = clean.match(/^(\w+)(?:\(([^)]+)\))?(!)?:\s+(.+)$/);
  const breaking = Boolean(match?.[3]) || /BREAKING CHANGE:/m.test(body);
  return {
    sha,
    subject: clean,
    body: body.trim(),
    type: match ? match[1].toLowerCase() : null,
    scope: match?.[2] ?? null,
    description: match ? match[4] : clean,
    breaking,
  };
}

async function listCommits(from: string | null, to: string): Promise<Commit[]> {
  const range = from ? `${from}..${to}` : to;
  let raw: string;
  try {
    raw = await git(
      "log",
      range,
      "--pretty=format:%H%x1f%s%x1f%b%x1e",
      "--no-merges",
    );
  } catch {
    return [];
  }
  if (!raw.trim()) return [];
  return raw.split("\x1e").filter((e) => e.trim()).map((entry) => {
    const [sha = "", subject = "", body = ""] = entry.split("\x1f");
    return parseCommit(sha.trim(), subject, body);
  });
}

export function renderNotes(opts: {
  version: string;
  date: string;
  from: string | null;
  to: string;
  commits: Commit[];
  repo: string;
}): string {
  const { version, date, from, commits, repo } = opts;
  const lines: string[] = [];
  lines.push(`## v${version} — ${date}`, "");
  if (from) {
    lines.push(`**Full diff:** \`${from}...v${version}\``, "");
  } else {
    lines.push(`**Initial versioned release.**`, "");
  }
  if (commits.length === 0) {
    lines.push("_No changes since last release._", "");
    return lines.join("\n");
  }

  const breaking = commits.filter((c) => c.breaking);
  if (breaking.length > 0) {
    lines.push("### ⚠️ Breaking changes", "");
    for (const c of breaking) {
      lines.push(`- ${c.subject} (${c.sha.slice(0, 7)})`);
    }
    lines.push("");
  }

  const byType = new Map<string, Commit[]>();
  const other: Commit[] = [];
  for (const c of commits) {
    const group = GROUPS.find((g) => g.types.includes(c.type ?? ""));
    if (group) {
      const arr = byType.get(group.key) ?? [];
      arr.push(c);
      byType.set(group.key, arr);
    } else {
      other.push(c);
    }
  }
  for (const g of GROUPS) {
    const arr = byType.get(g.key) ?? [];
    if (arr.length === 0) continue;
    lines.push(`### ${g.title}`, "");
    for (const c of arr) {
      const scope = c.scope ? `**${c.scope}:** ` : "";
      lines.push(`- ${scope}${c.description} (${c.sha.slice(0, 7)})`);
    }
    lines.push("");
  }
  if (other.length > 0) {
    lines.push("### Other changes", "");
    for (const c of other) {
      lines.push(`- ${c.subject} (${c.sha.slice(0, 7)})`);
    }
    lines.push("");
  }

  lines.push("### Contributors", "");
  lines.push(
    `_Generated from ${commits.length} commit(s) in range \`${
      from ?? "(beginning)"
    }..${opts.to}\`._`,
  );
  if (repo) lines.push(`_Repo: ${repo}_`);
  lines.push("");
  return lines.join("\n");
}

async function repoSlug(): Promise<string> {
  try {
    const url = await git("config", "--get", "remote.origin.url");
    const m = url.match(/github\.com[:/](.+?)(?:\.git)?$/);
    return m ? m[1] : "";
  } catch {
    return "";
  }
}

if (import.meta.main) {
  const args = parseArgs(Deno.args);
  const to = String(args["to"] ?? "HEAD");
  let from = args["from"] ? String(args["from"]) : await latestTag();
  if (from === "") from = null;
  const explicitVersion = args["version"] ? String(args["version"]) : null;
  const bump = args["bump"] ? String(args["bump"]) : null;
  const current = await currentVersion();
  const version = explicitVersion
    ? explicitVersion.replace(/^v/, "")
    : bump
    ? bumpVersion(current, bump)
    : current;
  const date = new Date().toISOString().slice(0, 10);
  if (args["print-version"]) {
    console.log(version);
    Deno.exit(0);
  }
  const commits = await listCommits(from, to);
  const repo = await repoSlug();
  const notes = renderNotes({ version, date, from, to, commits, repo });

  const out = args["out"] ? String(args["out"]) : null;
  if (out) await Deno.writeTextFile(out, notes);

  if (args["write-changelog"]) {
    const path = "CHANGELOG.md";
    let existing = "";
    try {
      existing = await Deno.readTextFile(path);
    } catch {
      existing = "# Changelog\n\nAll notable changes to this project.\n";
    }
    if (existing.includes(`## v${version}`)) {
      console.log(`CHANGELOG.md already contains v${version}, skipping.`);
    } else {
      const header = "# Changelog\n\nAll notable changes to this project.\n";
      const rest = existing.startsWith(header)
        ? existing.slice(header.length)
        : existing;
      await Deno.writeTextFile(path, `${header}\n${notes}\n${rest.trim()}\n`);
      console.log(`Prepended v${version} to ${path}`);
    }
  }

  if (args["write-docs"]) {
    const path = `docs/releases/v${version}.md`;
    const doc = `# Release v${version}\n\n${notes}\n`;
    await Deno.writeTextFile(path, doc);
    console.log(`Wrote ${path}`);
  }

  if (!out) console.log(notes);

  if (args["manifest"]) {
    const manifest = {
      version,
      previousTag: from,
      tag: `v${version}`,
      date,
      commitCount: commits.length,
      commits: commits.map((c) => ({
        sha: c.sha,
        subject: c.subject,
        type: c.type,
        scope: c.scope,
        breaking: c.breaking,
      })),
    };
    await Deno.writeTextFile(
      String(args["manifest"]),
      JSON.stringify(manifest, null, 2) + "\n",
    );
    console.log(`Wrote ${args["manifest"]}`);
  }

  if (Deno.env.get("GITHUB_OUTPUT")) {
    const ghOut = Deno.env.get("GITHUB_OUTPUT")!;
    const append = await Deno.open(GhOutPath(ghOut), {
      append: true,
      create: true,
      write: true,
    });
    const enc = new TextEncoder();
    const write = async (s: string) => await append.write(enc.encode(s));
    await write(`version=${version}\n`);
    await write(`previous_tag=${from ?? ""}\n`);
    await write(`commit_count=${commits.length}\n`);
    append.close();
  }
}

function GhOutPath(p: string): string {
  return p;
}
