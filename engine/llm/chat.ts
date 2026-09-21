/**
 * @module llm/chat
 *
 * Backward-compatible re-export. The ChatEngine now lives in
 * `engine/chat/chat_engine.ts` as part of the high-level chat layer.
 *
 * @deprecated Import from `../chat/chat_engine.ts` or the package root instead.
 */

export { ChatEngine } from "../chat/chat_engine.ts";

/** @deprecated Use `Backend` from `../runtime/types.ts` instead. */
export type ChatBackend = "webgpu" | "wasm";
