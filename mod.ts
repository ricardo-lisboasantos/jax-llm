// @module @ricardo/jax-llm
//
// Main entry point for the `@ricardo/jax-llm` library.
//
// Architecture:
//  - **Runtime layer** (`engine/runtime/`): low-level JAX-JS model lifecycle —
//    device init, tokenizer/weight loading, JIT inference, optax training.
//  - **Chat layer** (`engine/chat/`): high-level `ChatEngine` — just pass a
//    model name and call `engine.chat(input)`.
//
// @example
// ```ts
// import { ChatEngine } from "@ricardo/jax-llm";
//
// const engine = new ChatEngine("lfm2.5-350m");
// await engine.init();
// const reply = await engine.chat("Hello!");
// ```

export * from "./engine/index.ts";
