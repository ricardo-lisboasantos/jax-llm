/**
 * @module engine
 *
 * Re-exports all public symbols from the engine's layered architecture.
 *
 * Layer 1 — Runtime (`./runtime/`): device init, tokenizer/weight loading,
 *           JIT inference, optax training.
 * Layer 2 — Chat (`./chat/`): ChatEngine, sampling, prompt formatting.
 *
 * Model implementations (`./llm/`) and tokenizers (`./tokenizer/`) are
 * also re-exported for advanced use cases.
 */

// Low-level runtime layer
export * from "./runtime/index.ts";

// High-level chat layer
export * from "./chat/index.ts";

// Model implementations (Gemma, LFM, model registry)
export * from "./llm/model.ts";
export * from "./llm/gemma.ts";
export * from "./llm/lfm.ts";

// Tokenizer
export * from "./tokenizer/tokenizer.ts";

// Database
export * from "./database/database.ts";
export * from "./database/kv.ts";

// Config
export * from "./config.ts";
