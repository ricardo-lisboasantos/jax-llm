/**
 * @module training/lora
 *
 * Low-Rank Adaptation (LoRA) configuration and helpers.
 *
 * LoRA freezes the base weights and trains small rank-decomposition
 * matrices (A @ B) that are added to selected projection layers.
 */

export type LoraConfig = {
  /** Rank of the adaptation matrices. */
  rank: number;
  /** Scaling factor (alpha / rank applied at merge time). */
  alpha: number;
  /** Dropout applied to the LoRA branch during training. */
  dropout?: number;
  /** Name substrings of target modules (e.g. ["qProj", "vProj"]). */
  targetModules: string[];
};

export const DEFAULT_LORA_CONFIG: LoraConfig = {
  rank: 8,
  alpha: 16,
  dropout: 0,
  targetModules: ["qProj", "vProj"],
};

/** Scaling factor applied to the LoRA branch (alpha / rank). */
export function loraScale(config: LoraConfig): number {
  if (config.rank <= 0) throw new Error("LoRA rank must be > 0");
  return config.alpha / config.rank;
}

/** Number of trainable params for one adapted matrix of shape [in, out]. */
export function loraParamCount(
  inDim: number,
  outDim: number,
  config: LoraConfig = DEFAULT_LORA_CONFIG,
): number {
  return config.rank * (inDim + outDim);
}
