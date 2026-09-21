/**
 * @module chat/prompt
 *
 * Per-model-family prompt formatting functions.
 * Each function converts a `ChatMessage[]` history into the raw text
 * string that the model's tokenizer will encode.
 *
 * The BOS token is NOT included here — `ModelDefinition.encodePrompt`
 * prepends it as a token ID so SentencePiece doesn't treat it as text.
 */

import type { ChatMessage } from "./types.ts";

/**
 * Gemma chat template using `<start_of_turn>` / `<end_of_turn>` markers.
 * The assistant role is mapped to "model" as per Gemma's convention.
 */
export function gemmaPrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    const role = message.role === "assistant" ? "model" : "user";
    text += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
  }
  return `${text}<start_of_turn>model\n`;
}

/**
 * LFM (Liquid Foundation Model) chat template using ChatML-style
 * `<|im_start|>` / `<|im_end|>` markers.
 */
export function lfmPrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    text += `<|im_start|>${message.role}\n${content}<|im_end|>\n`;
  }
  return `${text}<|im_start|>assistant\n`;
}

/**
 * Generic ChatML-style prompt for dynamically loaded HuggingFace models
 * that don't have a registered template. Uses the same format as LFM.
 */
export function genericPrompt(history: ChatMessage[]): string {
  return lfmPrompt(history);
}

/**
 * Qwen2 chat template using ChatML-style `<|im_start|>` / `<|im_end|>`
 * markers (same structure as LFM, Qwen's native format).
 */
export function qwenPrompt(history: ChatMessage[]): string {
  return lfmPrompt(history);
}

/**
 * Bonsai (Llama/Mistral) chat template using `[INST]` / `[/INST]`
 * markers, following the Mistral chat format that Bonsai adopts.
 */
export function bonsaiPrompt(history: ChatMessage[]): string {
  let systemText = "";
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    if (message.role === "system") {
      systemText = content;
      continue;
    }
    if (message.role === "user") {
      const inst = systemText ? `${systemText}\n\n${content}` : content;
      text += `[INST] ${inst} [/INST]`;
      systemText = "";
    } else {
      text += ` ${content}</s>`;
    }
  }
  return `<s>${text}`;
}

/**
 * GPT-2 chat template. GPT-2 has no native chat format — this uses a
 * simple `User:` / `Assistant:` convention for multi-turn dialogue.
 */
export function gptPrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    const role = message.role === "assistant" ? "Assistant" : "User";
    text += `${role}: ${content}\n`;
  }
  return `${text}Assistant:`;
}

/**
 * Phi-2 chat template using `Instruct:` / `Output:` markers,
 * following Phi-2's recommended prompting convention.
 */
export function phiPrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    if (message.role === "assistant") {
      text += `Output: ${content}\n`;
    } else {
      text += `Instruct: ${content}\n`;
    }
  }
  return `${text}Output:`;
}

/**
 * Maple chat template using ChatML-style `<|im_start|>` / `<|im_end|>`
 * markers (Maple uses a Qwen-derived tokenizer).
 */
export function maplePrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    text += `<|im_start|>${message.role}\n${content}<|im_end|>\n`;
  }
  return `${text}<|im_start|>assistant\n`;
}
