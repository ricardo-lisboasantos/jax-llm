/**
 * @module training/qlora
 *
 * Quantized LoRA (QLoRA): LoRA on top of a quantized frozen base model
 * (typically 4-bit NF4) to enable fine-tuning on a single GPU.
 */

import { DEFAULT_LORA_CONFIG, type LoraConfig } from "./lora.ts";

export type QloraConfig = LoraConfig & {
  /** Base-model quantization: 4-bit NF4 is the QLoRA default. */
  bits: 4 | 8;
  /** Quantization scheme for the frozen base weights. */
  quantType: "nf4" | "fp4" | "int8";
  /** Use double quantization for the quantization constants. */
  doubleQuant?: boolean;
};

export const DEFAULT_QLORA_CONFIG: QloraConfig = {
  ...DEFAULT_LORA_CONFIG,
  bits: 4,
  quantType: "nf4",
  doubleQuant: true,
};

/** Validate a QLoRA config; throws on unsupported combinations. */
export function validateQloraConfig(config: QloraConfig): void {
  if (config.bits !== 4 && config.bits !== 8) {
    throw new Error(`Unsupported QLoRA bits: ${config.bits}`);
  }
  if (config.rank <= 0) throw new Error("QLoRA rank must be > 0");
}
