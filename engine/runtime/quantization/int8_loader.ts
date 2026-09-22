/**
 * @module runtime/quantization/int8_loader
 *
 * INT8 quantization weight loader with on-the-fly dequantization.
 * Reduces model memory footprint by 4x (float32 → int8) to 2x (float16 → int8).
 *
 * Format: Quantized weights stored as:
 *  - int8 values: [-128, 127] representing quantized activations
 *  - scale: float32 per-channel scaling factor
 *  - zero_point: int8 per-channel zero offset
 *
 * Dequantization: original_value = (int8_value - zero_point) * scale
 *
 * Storage: Packed in safetensors with metadata:
 *  - "model.layer.0.linear.weight_q8": int8 quantized values
 *  - "model.layer.0.linear.weight_scales": float32 scales
 *  - "model.layer.0.linear.weight_zero_point": int8 zero points
 *
 * Goals:
 *  - Load quantized weights directly from safetensors
 *  - Dequantize on-the-fly during forward pass (minimal latency)
 *  - 50% memory reduction vs float32 baseline
 *  - <5% latency overhead from dequantization
 */

import type { numpy as np } from "npm:@jax-js/jax@^0.1.25";

export interface QuantizationMetadata {
  /** Quantization bit depth (8, 4, 2, etc.). */
  bits: number;
  /** Per-channel scales for dequantization. */
  scales: number[];
  /** Per-channel zero points for dequantization. */
  zeroPoints: number[];
  /** Original dtype before quantization. */
  originalDtype: string;
  /** Whether to use symmetric quantization (range: [-127, 127]). */
  symmetric: boolean;
}

export interface QuantizedWeight {
  /** Quantized values (int8 or smaller). */
  data: Int8Array;
  /** Shape of the original (dequantized) tensor. */
  shape: number[];
  /** Quantization metadata. */
  metadata: QuantizationMetadata;
}

/**
 * Dequantize an int8 weight tensor using per-channel scaling.
 *
 * Formula: original = (quantized - zero_point) * scale
 *
 * @param quantized Quantized int8 data
 * @param scales Per-channel scaling factors
 * @param zeroPoints Per-channel zero offsets
 * @param shape Target output shape
 * @returns Float32 dequantized array
 */
export function dequantizeInt8(
  quantized: Int8Array,
  scales: number[],
  zeroPoints: number[],
  shape: number[],
): Float32Array {
  const result = new Float32Array(quantized.length);

  // Simple per-element dequantization with per-channel scaling
  const channelSize = quantized.length / scales.length;
  for (let i = 0; i < quantized.length; i++) {
    const channelIdx = Math.floor(i / channelSize);
    const scale = scales[channelIdx];
    const zeroPoint = zeroPoints[channelIdx];
    result[i] = (quantized[i] - zeroPoint) * scale;
  }

  return result;
}

/**
 * Quantize a float32 array to int8 with per-channel scaling.
 * Used for saving quantized checkpoints.
 *
 * @param data Float32 values to quantize
 * @param scales Per-channel scales (should be computed from data statistics)
 * @param zeroPoints Per-channel zero points
 * @returns Quantized int8 array
 */
export function quantizeToInt8(
  data: Float32Array,
  scales: number[],
  zeroPoints: number[],
): Int8Array {
  const result = new Int8Array(data.length);
  const channelSize = data.length / scales.length;

  for (let i = 0; i < data.length; i++) {
    const channelIdx = Math.floor(i / channelSize);
    const scale = scales[channelIdx];
    const zeroPoint = zeroPoints[channelIdx];

    // Quantize: int8 = round(value / scale) + zero_point
    const quantized = Math.round(data[i] / scale + zeroPoint);
    result[i] = Math.max(-128, Math.min(127, quantized));
  }

  return result;
}

/**
 * Compute optimal scales and zero points from float data.
 * Uses min-max quantization per channel.
 *
 * @param data Float32 values
 * @param numChannels Number of channels for per-channel quantization
 * @returns { scales, zeroPoints } for use in quantization
 */
export function computeQuantizationParams(
  data: Float32Array,
  numChannels: number,
): { scales: number[]; zeroPoints: number[] } {
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  const channelSize = data.length / numChannels;

  for (let ch = 0; ch < numChannels; ch++) {
    const start = ch * channelSize;
    const end = start + channelSize;
    const channelData = data.slice(start, end);

    // Find min/max for this channel
    let minVal = channelData[0];
    let maxVal = channelData[0];
    for (let i = 0; i < channelData.length; i++) {
      minVal = Math.min(minVal, channelData[i]);
      maxVal = Math.max(maxVal, channelData[i]);
    }

    // Compute scale: map [minVal, maxVal] to [-128, 127]
    const range = maxVal - minVal || 1e-8; // Avoid division by zero
    const scale = range / 255; // 256 values for int8 range
    const zeroPoint = Math.round(-minVal / scale);

    scales.push(scale);
    zeroPoints.push(Math.max(-128, Math.min(127, zeroPoint)));
  }

  return { scales, zeroPoints };
}

/**
 * Parser for INT8-quantized safetensors weights.
 * Extracts quantized data and metadata from a checkpoint.
 */
export class Int8QuantizationLoader {
  /**
   * Load an INT8 quantized weight from safetensors-like format.
   * Expected keys in checkpoint:
   *  - "{name}_q8": int8 quantized data
   *  - "{name}_scales": float32 scales
   *  - "{name}_zp": int8 zero points
   *
   * @param checkpointData Raw safetensors bytes
   * @param weightName Name of the weight (e.g., "embedding.weight")
   * @returns Quantized weight object, or null if not quantized
   */
  static parseQuantizedWeight(
    checkpointData: Uint8Array,
    weightName: string,
  ): QuantizedWeight | null {
    // This is a simplified parser. In practice, you'd use a safetensors library.
    // For now, assume the checkpoint includes metadata indicating quantization.

    // Placeholder: would require actual safetensors parsing
    // For demonstration, return null (indicating weight not quantized)
    return null;
  }

  /**
   * Check if a weight is quantized in the checkpoint.
   * @param weightName Weight name to check
   * @param checkpointKeys All keys in the checkpoint
   * @returns True if quantized variant exists
   */
  static isQuantized(weightName: string, checkpointKeys: string[]): boolean {
    return checkpointKeys.includes(`${weightName}_q8`);
  }
}

/**
 * Quantization-aware weight cache for mixed quantized/float weights.
 * Dequantizes on-demand with transparent API.
 */
export class QuantizationCache {
  private quantizedWeights = new Map<string, QuantizedWeight>();
  private dequantizedCache = new Map<string, Float32Array>();

  /**
   * Register a quantized weight.
   */
  registerQuantized(name: string, weight: QuantizedWeight): void {
    this.quantizedWeights.set(name, weight);
  }

  /**
   * Get a weight, dequantizing if necessary.
   * Results are cached to avoid repeated dequantization.
   *
   * @param name Weight name
   * @returns Dequantized float32 array, or undefined if not found
   */
  getWeight(name: string): Float32Array | undefined {
    // Check cache first
    if (this.dequantizedCache.has(name)) {
      return this.dequantizedCache.get(name)!;
    }

    // Check quantized weights
    const quantized = this.quantizedWeights.get(name);
    if (!quantized) return undefined;

    // Dequantize and cache
    const meta = quantized.metadata;
    const dequantized = dequantizeInt8(
      quantized.data,
      meta.scales,
      meta.zeroPoints,
      quantized.shape,
    );

    this.dequantizedCache.set(name, dequantized);
    return dequantized;
  }

  /**
   * Clear dequantized cache to free memory.
   * Quantized weights are retained.
   */
  clearDequantizedCache(): void {
    this.dequantizedCache.clear();
  }

  /**
   * Get cache statistics (for monitoring memory usage).
   */
  getStats(): { quantized: number; dequantized: number; totalBytes: number } {
    let quantizedCount = 0;
    let dequantizedCount = 0;
    let totalBytes = 0;

    for (const weight of this.quantizedWeights.values()) {
      quantizedCount++;
      totalBytes += weight.data.byteLength;
    }

    for (const dequantized of this.dequantizedCache.values()) {
      dequantizedCount++;
      totalBytes += dequantized.byteLength;
    }

    return { quantized: quantizedCount, dequantized: dequantizedCount, totalBytes };
  }
}

/**
 * Global quantization cache instance.
 */
export const globalQuantizationCache = new QuantizationCache();
