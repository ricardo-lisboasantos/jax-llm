# jax-llm Documentation

Complete reference documentation for the jax-llm project.

## 📚 Quick Navigation

### Getting Started

- **[Contributing Guide](./guides/CONTRIBUTING.md)** — How to contribute to
  jax-llm
- **[Best Practices](./guides/BEST_PRACTICES.md)** — Code standards and
  conventions
- **[Standardization Guide](./guides/STANDARDIZATION_GUIDE.md)** — Project
  standards overview

### Development Guides

- **[Getting Started](./guides/DEVELOPMENT.md)** — Local development setup
- **[Training Guide](./guides/TRAINING.md)** — Model training instructions
- **[Deployment Guide](./guides/DEPLOYMENT.md)** — Deployment procedures
- **[Configuration](./guides/CONFIGURATION.md)** — Setup and configuration
  options

### Reference Documentation

- **[Architecture](./reference/ARCHITECTURE.md)** — System design and components
- **[API Reference](./reference/API.md)** — Complete API documentation
- **[Models](./reference/MODELS.md)** — Model documentation and specs
- **[Documentation Organization](./reference/DOCUMENTATION_ORGANIZATION_SUMMARY.md)**
  — How docs are organized
- **[Full Index](./reference/INDEX.md)** — Alphabetical reference

### Performance & Optimization

- **[Performance Index](./performance/)** — Performance benchmarks and reports
  - [Baseline](./performance/BASELINE.md) — Baseline performance metrics
  - [Comparison](./performance/COMPARISON.md) — Performance comparisons
  - [Report](./performance/REPORT.md) — Detailed performance analysis
  - [Summary](./performance/SUMMARY.md) — Quick performance overview

### Releases & Phases

- **[Releases Index](./releases/)** — Release notes and version history
  - [Phase 3.0 Release](./releases/PHASE_3_0_RELEASE.md) — v0.4.0 Performance
    Optimization
  - [Build Report](./releases/BUILD_REPORT.md) — Build and deployment details
  - [Executive Summary](./releases/EXECUTIVE_SUMMARY.md) — High-level release
    summary
  - [Implementation Diffs](./releases/IMPLEMENTATION_DIFFS.md) — Code changes
    per phase

### Phase 3.0 Details (v0.4.0)

- **[Phase 3.0 Index](./phase-3-0/PHASE_3_0_INDEX.md)** — Complete phase
  documentation
- **[Executive Summary](./phase-3-0/PHASE_3_0_EXECUTIVE_SUMMARY.md)** —
  High-level overview
- **[Roadmap](./phase-3-0/PHASE_3_0_ROADMAP.md)** — Development roadmap
- **[Implementation](./phase-3-0/PHASE_3_0_FORGE_IMPLEMENTATION.md)** —
  Technical details
- **[Exact Diffs](./phase-3-0/PHASE_3_0_EXACT_DIFFS.md)** — Line-by-line changes
- **[QA Sign-Off](./phase-3-0/GUARDIAN_QA_FINAL_SIGN_OFF.md)** — Quality
  assurance
- **[Security Audit](./phase-3-0/PHASE_3_0_SECURITY_AUDIT.md)** — Security
  review

---

## 📖 Documentation Structure

```
docs/
├── README.md                      ← You are here (overview & navigation)
│
├── guides/                        ← How-to guides and workflows
│   ├── CONTRIBUTING.md            ← Contributor guide
│   ├── BEST_PRACTICES.md          ← Standards & best practices
│   ├── STANDARDIZATION_GUIDE.md   ← Quick reference
│   ├── DEVELOPMENT.md             ← Local development setup
│   ├── TRAINING.md                ← Model training instructions
│   ├── DEPLOYMENT.md              ← Deployment procedures
│   └── CONFIGURATION.md           ← Setup & configuration
│
├── reference/                     ← Reference documentation
│   ├── ARCHITECTURE.md            ← System design & components
│   ├── API.md                     ← API documentation
│   ├── MODELS.md                  ← Model specs
│   ├── DOCUMENTATION_ORGANIZATION_SUMMARY.md ← Doc organization guide
│   └── INDEX.md                   ← Alphabetical index
│
├── performance/                   ← Performance analysis
│   ├── BASELINE.md                ← Baseline metrics
│   ├── COMPARISON.md              ← Performance comparisons
│   ├── REPORT.md                  ← Detailed analysis
│   └── SUMMARY.md                 ← Quick overview
│
├── releases/                      ← Release notes & build reports
│   ├── PHASE_3_0_RELEASE.md       ← Latest: v0.4.0
│   ├── BUILD_REPORT.md
│   ├── EXECUTIVE_SUMMARY.md
│   └── IMPLEMENTATION_DIFFS.md
│
└── phase-3-0/                     ← Phase 3.0 implementation details
    ├── PHASE_3_0_INDEX.md
    ├── PHASE_3_0_EXECUTIVE_SUMMARY.md
    ├── PHASE_3_0_ROADMAP.md
    ├── PHASE_3_0_FORGE_IMPLEMENTATION.md
    ├── PHASE_3_0_EXACT_DIFFS.md
    ├── PHASE_3_0_SECURITY_AUDIT.md
    └── ... (18 files total)

Root-Level (Minimal):
├── README.md                      ← Project overview only
└── .githooks/README.md            ← Git hooks guide
```

---

## 🎯 Common Tasks

### I want to...

**Understand the project** → Read [README.md](../README.md) then
[Architecture](./reference/ARCHITECTURE.md)

**Set up for development** → Follow [Development Guide](./guides/DEVELOPMENT.md)

**Deploy the project** → Follow [Deployment Guide](./guides/DEPLOYMENT.md)

**Train a model** → See [Training Guide](./guides/TRAINING.md)

**Learn the code standards** → Read [Best Practices](./guides/BEST_PRACTICES.md)

**Understand performance optimizations** → Check [Performance](./performance/)
folder

**Review Phase 3.0 changes** → Start with
[Phase 3.0 Index](./phase-3-0/PHASE_3_0_INDEX.md)

**See all API endpoints** → Read [API Reference](./reference/API.md)

**Configure the project** → Follow
[Configuration Guide](./guides/CONFIGURATION.md)

---

## 📊 Project Status

**Version:** 0.4.0\
**Status:** 🟢 Production Ready\
**Latest Release:**
[Phase 3.0 - Performance Optimization](./releases/PHASE_3_0_RELEASE.md)

**Key Metrics:**

- ✅ 138 tests passing
- ✅ 70%+ code coverage
- ✅ 0 lint errors
- ✅ All types validated
- ✅ Performance optimized (-75% memory, +100% throughput)

---

## 🔍 Finding Information

### By Category

| Category      | Folder         | Contents                               |
| ------------- | -------------- | -------------------------------------- |
| How-to guides | `guides/`      | Workflows, setup, training, deployment |
| Reference     | `reference/`   | Architecture, API, models, index       |
| Performance   | `performance/` | Benchmarks, analysis, comparisons      |
| Releases      | `releases/`    | Release notes, build reports           |
| Phase 3.0     | `phase-3-0/`   | Implementation details, diffs, QA      |

### By Topic

- **Architecture** → [reference/ARCHITECTURE.md](./reference/ARCHITECTURE.md)
- **API** → [reference/API.md](./reference/API.md)
- **Deployment** → [guides/DEPLOYMENT.md](./guides/DEPLOYMENT.md)
- **Performance** → [performance/](./performance/)
- **Training** → [guides/TRAINING.md](./guides/TRAINING.md)
- **Models** → [reference/MODELS.md](./reference/MODELS.md)
- **Development** → [guides/DEVELOPMENT.md](./guides/DEVELOPMENT.md)

### Quick Lookup

- **Alphabetical** → [reference/INDEX.md](./reference/INDEX.md)
- **Latest Release** →
  [releases/PHASE_3_0_RELEASE.md](./releases/PHASE_3_0_RELEASE.md)

---

## 📝 Adding Documentation

Documentation files should be organized by type:

**Guides (how-to, workflows):**

- Place in `docs/guides/`
- Example: `TRAINING.md`, `DEPLOYMENT.md`

**Reference (specs, technical):**

- Place in `docs/reference/`
- Example: `API.md`, `ARCHITECTURE.md`

**Performance Analysis:**

- Place in `docs/performance/`
- Keep folder flat or organize by metric

**Release Notes:**

- Place in `docs/releases/`
- Example: `PHASE_3_0_RELEASE.md`

**Phase/Version Details:**

- Create folder: `docs/phase-X-Y/`
- Example: `docs/phase-3-0/`

**Guidelines:**

- Use UPPERCASE for file names: `ARCHITECTURE.md` not `architecture.md`
- Include TABLE OF CONTENTS in long docs
- Link related documents together
- Update this README when adding new folders
- Follow [Best Practices](../BEST_PRACTICES.md)

---

## 🆘 Need Help?

1. **Setting up?** → [Development Guide](./guides/DEVELOPMENT.md)
2. **Using the API?** → [API Reference](./reference/API.md)
3. **Deploying?** → [Deployment Guide](./guides/DEPLOYMENT.md)
4. **Training models?** → [Training Guide](./guides/TRAINING.md)
5. **Understanding code?** → [Architecture](./reference/ARCHITECTURE.md)
6. **Need a quick lookup?** → [Full Index](./reference/INDEX.md)

---

**Last Updated:** September 22, 2026\
**Version:** 0.4.0\
**Organized Under:** Documentation standards enforced by pre-commit hooks
