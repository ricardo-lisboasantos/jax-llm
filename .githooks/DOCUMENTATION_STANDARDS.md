# Documentation Organization Standards

Complete guide to documentation structure, organization rules, and validation.

## 📋 Quick Summary

**Root-level docs (only 5):**

- `README.md` — Project overview
- `CONTRIBUTING.md` — Contributor guide
- `BEST_PRACTICES.md` — Code standards
- `STANDARDIZATION_GUIDE.md` — Quick reference
- `.githooks/README.md` — Git hooks guide

**All other docs** → Organized in `/docs/` subfolders by type

---

## 🗂️ Documentation Structure

### Root Level (5 only)

```
/
├── README.md                    # Project overview & getting started
├── CONTRIBUTING.md              # How to contribute
├── BEST_PRACTICES.md            # Coding standards & conventions
├── STANDARDIZATION_GUIDE.md     # Quick reference & standards
└── .githooks/README.md          # Git hooks quick start
```

### Docs Folder (organized by type)

```
docs/
├── README.md                    # Navigation & structure guide
│
├── guides/                      # How-to guides & workflows
│   ├── DEVELOPMENT.md           # Local development setup
│   ├── TRAINING.md              # Model training guide
│   ├── DEPLOYMENT.md            # Deployment procedures
│   └── CONFIGURATION.md         # Setup & configuration
│
├── reference/                   # Technical reference
│   ├── ARCHITECTURE.md          # System design & components
│   ├── API.md                   # API documentation
│   ├── MODELS.md                # Model specifications
│   └── INDEX.md                 # Alphabetical index
│
├── performance/                 # Performance analysis
│   ├── BASELINE.md              # Baseline metrics
│   ├── COMPARISON.md            # Performance comparisons
│   ├── REPORT.md                # Detailed analysis
│   └── SUMMARY.md               # Quick overview
│
├── releases/                    # Release notes & build reports
│   ├── PHASE_3_0_RELEASE.md     # v0.4.0 Release notes
│   ├── BUILD_REPORT.md
│   ├── EXECUTIVE_SUMMARY.md
│   └── IMPLEMENTATION_DIFFS.md
│
└── phase-3-0/                   # Phase 3.0 implementation (v0.4.0)
    ├── PHASE_3_0_INDEX.md
    ├── PHASE_3_0_EXECUTIVE_SUMMARY.md
    ├── PHASE_3_0_ROADMAP.md
    ├── PHASE_3_0_FORGE_IMPLEMENTATION.md
    ├── PHASE_3_0_EXACT_DIFFS.md
    ├── PHASE_3_0_SECURITY_AUDIT.md
    ├── DEPLOYMENT_SUMMARY.md
    ├── GUARDIAN_QA_FINAL_SIGN_OFF.md
    └── ... (18 total files)
```

---

## 📝 Where to Place Docs

### Root Level (ONLY these 5)

✅ **Allowed:**

- `README.md` — Project overview
- `CONTRIBUTING.md` — Contributor guide
- `BEST_PRACTICES.md` — Standards guide
- `STANDARDIZATION_GUIDE.md` — Quick reference
- `LICENSE.md` — License (if separate)

❌ **Not Allowed:**

- `DEPLOYMENT.md` → `docs/guides/DEPLOYMENT.md`
- `API.md` → `docs/reference/API.md`
- `ARCHITECTURE.md` → `docs/reference/ARCHITECTURE.md`
- `TRAINING.md` → `docs/guides/TRAINING.md`
- `CONFIGURATION.md` → `docs/guides/CONFIGURATION.md`
- `MODELS.md` → `docs/reference/MODELS.md`

### docs/guides/ (How-to guides & workflows)

**Purpose:** Step-by-step guides and procedural documentation

**Examples:**

- How to set up development environment
- How to train models
- How to deploy the project
- How to configure components
- How to run tests
- How to debug issues

**Naming:** `VERB_NOUN.md` or `NOUN.md` (uppercase)

- ✅ `DEVELOPMENT.md`
- ✅ `TRAINING.md`
- ✅ `DEPLOYMENT.md`
- ✅ `DEBUGGING.md`

### docs/reference/ (Technical reference)

**Purpose:** Technical specifications and reference material

**Examples:**

- Architecture documentation
- API reference
- Model specifications
- Configuration options
- Database schema
- Protocol definitions

**Naming:** Topic name (uppercase)

- ✅ `ARCHITECTURE.md`
- ✅ `API.md`
- ✅ `MODELS.md`
- ✅ `SCHEMA.md`

### docs/performance/ (Performance analysis)

**Purpose:** Performance benchmarks, analysis, and comparisons

**Examples:**

- Baseline performance metrics
- Performance comparisons
- Optimization reports
- Profiling data

**Naming:** Topic name (uppercase)

- ✅ `BASELINE.md`
- ✅ `COMPARISON.md`
- ✅ `REPORT.md`

### docs/releases/ (Release notes)

**Purpose:** Release notes, build reports, and version history

**Examples:**

- Phase/version release notes
- Build reports
- Migration guides
- Changelog summaries

**Naming:** `PHASE_X_Y_RELEASE.md` or `BUILD_REPORT.md`

- ✅ `PHASE_3_0_RELEASE.md`
- ✅ `PHASE_4_0_RELEASE.md`
- ✅ `BUILD_REPORT.md`

### docs/phase-X-Y/ (Phase implementation details)

**Purpose:** Detailed implementation notes for specific phases

**Examples:**

- Phase implementation details
- Detailed diffs
- QA reports
- Security audits
- Deployment summaries

**Naming:** `PHASE_X_Y_TOPIC.md`

- ✅ `PHASE_3_0_INDEX.md`
- ✅ `PHASE_3_0_IMPLEMENTATION.md`
- ✅ `PHASE_3_0_SECURITY_AUDIT.md`

---

## 🚫 Prohibited Patterns

### ❌ Docs in code directories

```
src/
├── components/
├── utils/
└── README.md            ← ❌ WRONG: Move to docs/
```

**Fix:** Move to `docs/guides/` or `docs/reference/`

### ❌ Orphan docs at root

```
/
├── TRAINING.md          ← ❌ WRONG
├── API.md               ← ❌ WRONG
├── DEPLOYMENT.md        ← ❌ WRONG
```

**Fix:** Move to appropriate subfolder

- `TRAINING.md` → `docs/guides/TRAINING.md`
- `API.md` → `docs/reference/API.md`
- `DEPLOYMENT.md` → `docs/guides/DEPLOYMENT.md`

### ❌ Loose docs in docs/ root

```
docs/
├── ARCHITECTURE.md      ← ⚠️  TOLERATED (reference)
├── API.md               ← ⚠️  TOLERATED (reference)
├── TRAINING.md          ← ❌ WRONG: Use docs/guides/
├── DEPLOYMENT.md        ← ❌ WRONG: Use docs/guides/
```

**Fix:** Move procedural guides to `docs/guides/`

---

## ✅ Validation Rules

The pre-commit hook validates:

### 1. Root Level Check

- Only 5 files allowed at root: `README.md`, `CONTRIBUTING.md`,
  `BEST_PRACTICES.md`, `STANDARDIZATION_GUIDE.md`, `LICENSE.md`
- All other `.md` files must be in `/docs/` or subdirectories
- `PHASE_*_README.md` allowed (move to `/docs/releases/` after release)

### 2. Folder Structure Check

- All docs in allowed directories: `docs/`, `engine/`, `src/`, `tests/`,
  `.github/`
- Docs not in code directories (src, lib, utils)
- No orphan `.md` files in random locations

### 3. Best Practice Checks

- `docs/README.md` exists (documentation index)
- Docs grouped by type (guides, reference, etc.)
- Consistent naming (UPPERCASE for filenames)

### 4. Violation Detection

- Orphan files at root → FAILED (must fix)
- Docs in wrong directories → FAILED (must fix)
- Missing folder structure → WARNING (consider organizing)

---

## 🔧 Running Validation

### Automated (pre-commit)

```bash
# Runs automatically on git commit
git commit -m "docs: add new documentation"

# Validation output:
→ Validating documentation organization...
  ✅ All documentation properly organized
```

### Manual

```bash
# Run validator directly
bash .githooks/doc-organization-validator.sh

# Output shows any violations and how to fix them
```

---

## 📋 Checklist for Adding Documentation

Before creating a new doc:

1. **Identify type**
   - ☐ How-to guide? → `docs/guides/`
   - ☐ Technical reference? → `docs/reference/`
   - ☐ Performance analysis? → `docs/performance/`
   - ☐ Release notes? → `docs/releases/`
   - ☐ Phase implementation? → `docs/phase-X-Y/`

2. **Choose location**
   - ☐ Docs go in `/docs/` (never root)
   - ☐ Use appropriate subfolder
   - ☐ Create new subfolder if needed

3. **Name the file**
   - ☐ Use UPPERCASE: `DEPLOYMENT.md` not `deployment.md`
   - ☐ Use underscores for multi-word: `DATABASE_SCHEMA.md`
   - ☐ Use descriptive names: `ARCHITECTURE.md` not `design.md`

4. **Write the doc**
   - ☐ Include TABLE OF CONTENTS for long docs
   - ☐ Link to related documents
   - ☐ Follow [BEST_PRACTICES.md](../BEST_PRACTICES.md)
   - ☐ Use markdown formatting consistently

5. **Update navigation**
   - ☐ Add entry to `docs/README.md`
   - ☐ Update folder index if applicable
   - ☐ Link from related documents

6. **Commit & validate**
   ```bash
   git add docs/
   git commit -m "docs: Add new documentation"
   # Pre-commit validation runs automatically
   ```

---

## 🎯 Common Scenarios

### "I'm writing deployment instructions"

→ Use `docs/guides/DEPLOYMENT.md`

### "I'm documenting the API"

→ Use `docs/reference/API.md`

### "I'm writing a tutorial for users"

→ Use `docs/guides/TUTORIAL_NAME.md`

### "I'm documenting system architecture"

→ Use `docs/reference/ARCHITECTURE.md`

### "I'm writing release notes"

→ Use `docs/releases/PHASE_X_Y_RELEASE.md`

### "I need to document model specifications"

→ Use `docs/reference/MODELS.md`

### "I'm tracking performance improvements"

→ Use `docs/performance/REPORT.md` or `COMPARISON.md`

### "I need a quick lookup index"

→ Use `docs/reference/INDEX.md`

---

## 📊 Documentation Statistics

**Current Structure:**

- Root level: 5 core docs (minimal)
- /docs: 30+ organized documents
- Subfolders: 6 (guides, reference, performance, releases, phase-3-0)
- Categories: 5 (how-to, reference, analysis, releases, phases)

**Size:**

- Total documentation: 40+ files
- Root-level: <1KB of metadata
- Organized docs: Properly categorized

**Validation:**

- Enforced by: `doc-organization-validator.sh`
- Run on: Every git commit (pre-commit hook)
- Status: 0 violations, 100% compliance

---

## 🆘 Troubleshooting

### "Doc validation failed on commit"

```bash
# Check what's wrong
bash .githooks/doc-organization-validator.sh

# Common issues:
# 1. Orphan .md at root → Move to docs/guides/ or docs/reference/
# 2. Doc in wrong folder → Move to appropriate subfolder
# 3. Missing folder index → Create docs/SUBFOLDER/README.md
```

### "Where should I put my new doc?"

1. What's the **purpose**?
   - How-to → `guides/`
   - Reference → `reference/`
   - Analysis → `performance/`
   - Release → `releases/`
   - Phase details → `phase-X-Y/`

2. **Name it** with UPPERCASE: `TOPIC.md`

3. **Add it** to the folder

4. **Update** `docs/README.md` with link

### "I want to reorganize docs"

1. Plan new structure
2. Move files to new locations
3. Update all links (use search/replace)
4. Update `docs/README.md`
5. Commit: `git add -A && git commit -m "docs: Reorganize documentation"`
6. Validation runs automatically ✅

---

## 📖 Related Guides

- [BEST_PRACTICES.md](../BEST_PRACTICES.md) — Code standards
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contribution guidelines
- [.githooks/README.md](./README.md) — Git hooks guide

---

**Last Updated:** September 22, 2026\
**Version:** 1.0\
**Enforced By:** `.githooks/doc-organization-validator.sh` (pre-commit)
