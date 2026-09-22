# JAX-LLM Documentation Index

## Getting Started

- **[README](../README.md)** — Quick start, API overview, installation
- **[Configuration](configuration.md)** — Engine options, model selection,
  backends
- **[Models](models.md)** — Built-in model registry, Hugging Face support

## Core Guides

- **[Architecture](architecture.md)** — System design, layer abstraction, module
  organization
- **[API Reference](api.md)** — Complete symbol reference with examples
- **[Development](development.md)** — Testing, CI/CD, publishing to JSR
- **[Training](training.md)** — Fine-tuning with Optax, loss functions,
  optimization

## Performance & Optimization (v0.3.3)

### Performance Reports

- **[Performance Report](performance/REPORT.md)** — Detailed metrics,
  methodology, analysis
- **[Baseline Comparison](performance/BASELINE.md)** — Baseline metrics and
  configuration
- **[Optimization Summary](performance/SUMMARY.md)** — 15 optimizations
  implemented, trade-offs
- **[Comparison Table](performance/COMPARISON.md)** — Before/after side-by-side
  metrics

### Deployment

- **[Deployment Guide](DEPLOYMENT.md)** — Production rollout, monitoring,
  rollback plan

## Release History

- **[Executive Summary](releases/EXECUTIVE_SUMMARY.md)** — High-level overview
- **[Build Report](releases/BUILD_REPORT.md)** — Implementation progress, test
  results
- **[Implementation Details](releases/IMPLEMENTATION_DIFFS.md)** — Code changes,
  file-by-file breakdown
- **[Test Reference](releases/TEST_REFERENCE.md)** — Test suite structure,
  coverage

## Quick Navigation

### By Task

**I want to...**

- Get started → [README](../README.md) + [Quick Start](../README.md#quick-start)
- Understand architecture → [Architecture](architecture.md)
- Build an integration → [API Reference](api.md)
- Fine-tune a model → [Training](training.md)
- Configure the engine → [Configuration](configuration.md)
- Set up CI/CD → [Development](development.md#cicd)
- Deploy to production → [Deployment Guide](DEPLOYMENT.md)
- Check performance improvements → [Performance Report](performance/REPORT.md)
- Review implementation → [Build Report](releases/BUILD_REPORT.md)

### By Component

- **Chat Layer**: `engine/chat/`, see [API Reference](api.md#chat-layer)
- **Runtime**: `engine/runtime/`, see
  [Architecture](architecture.md#runtime-layer)
- **Models**: `engine/llm/`, see [Models](models.md)
- **Tokenizers**: `engine/tokenizer/`, see [API Reference](api.md)
- **Profiling**: `engine/llm/profiling/`, see
  [Performance Report](performance/REPORT.md)
- **Quantization**: `engine/runtime/quantization/`, see
  [Optimization Summary](performance/SUMMARY.md)

## Version History

### v0.3.3 (Current)

- ✅ Performance optimization: -17.5% TTFT, +21.1% prefill throughput
- ✅ GPU buffer pool with LRU eviction
- ✅ WebGPU profiler for kernel-level diagnostics
- ✅ INT8 quantization foundation
- ✅ WASM SIMD kernels for CPU fallback
- ✅ Fixed UseAfterFreeError on long prompts (>8K tokens)
- ✅ All 127 unit tests passing

See [Performance Report](performance/REPORT.md) for detailed metrics.

### v0.3.2

- Initial stable release with Gemma, LFM, Qwen2.5 support

## Checklist for Common Tasks

### Production Deployment

- [ ] Review [Deployment Guide](DEPLOYMENT.md)
- [ ] Check [Performance Report](performance/REPORT.md) for baseline
      expectations
- [ ] Set up monitoring per [Deployment Guide](DEPLOYMENT.md#monitoring-plan)
- [ ] Test with staging models per [Configuration](configuration.md)

### New Feature Development

- [ ] Understand architecture per [Architecture](architecture.md)
- [ ] Review model implementations per [Models](models.md)
- [ ] Add tests per [Development](development.md#testing)
- [ ] Update API docs per [Development](development.md#documentation)

### Performance Tuning

- [ ] Review [Optimization Summary](performance/SUMMARY.md) for existing
      improvements
- [ ] Check [Performance Report](performance/REPORT.md) for profiler setup
- [ ] Measure baselines per [Baseline Comparison](performance/BASELINE.md)
- [ ] Profile with [WebGPU Profiler](../engine/llm/profiling/webgpu_profiler.ts)

---

**Last updated**: 2026-09-22\
**Current version**: 0.3.3
