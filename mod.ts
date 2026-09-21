// @module @ricardo/jax-llm
//
// Main entry point for the `@ricardo/jax-llm` library.
//
// Architecture:
//  - **Runtime layer** (`engine/runtime/`): low-level JAX-JS model lifecycle —
//    device init, tokenizer/weight loading, JIT inference, optax training.
//  - **Chat layer** (`engine/chat/`): high-level `ChatEngine` — `chat()` for
//    conversations, `generate()` for single prompts.
//
// @example
// ```ts
// import { ChatEngine } from "@ricardo/jax-llm";
//
// const engine = new ChatEngine("lfm2.5-350m");
// await engine.init();
// const reply = await engine.generate("Hello!");
// ```

export * from "./engine/index.ts";
