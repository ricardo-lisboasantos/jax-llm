/**
 * unit: INT8 quantization loader tests
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import {
  computeQuantizationParams,
  dequantizeInt8,
  QuantizationCache,
  quantizeToInt8,
} from "./int8_loader.ts";

Deno.test("unit: INT8 dequantization math", () => {
  // Create simple quantized data: [0, 1, 2, ...]
  const quantized = new Int8Array([0, 1, 2, 3, 4, 5]);
  const scales = [0.1]; // One channel
  const zeroPoints = [0];
  const shape = [6];

  const dequantized = dequantizeInt8(quantized, scales, zeroPoints, shape);

  // Verify formula: original = (quantized - zero_point) * scale
  assertAlmostEquals(dequantized[0], 0.0, 1e-5); // (0 - 0) * 0.1
  assertAlmostEquals(dequantized[1], 0.1, 1e-5); // (1 - 0) * 0.1
  assertAlmostEquals(dequantized[2], 0.2, 1e-5); // (2 - 0) * 0.1
  assertAlmostEquals(dequantized[5], 0.5, 1e-5); // (5 - 0) * 0.1
});

Deno.test("unit: INT8 dequantization with zero point", () => {
  const quantized = new Int8Array([0, 1, 2, 3]);
  const scales = [0.1];
  const zeroPoints = [1]; // Non-zero offset
  const shape = [4];

  const dequantized = dequantizeInt8(quantized, scales, zeroPoints, shape);

  // Formula: (quantized - zero_point) * scale
  assertAlmostEquals(dequantized[0], -0.1, 1e-5); // (0 - 1) * 0.1
  assertAlmostEquals(dequantized[1], 0.0, 1e-5); // (1 - 1) * 0.1
  assertAlmostEquals(dequantized[2], 0.1, 1e-5); // (2 - 1) * 0.1
  assertAlmostEquals(dequantized[3], 0.2, 1e-5); // (3 - 1) * 0.1
});

Deno.test("unit: INT8 dequantization per-channel", () => {
  // Two channels: channel 0 uses scale 0.1, channel 1 uses scale 0.2
  const quantized = new Int8Array([0, 1, 2, 3]); // [ch0[0], ch0[1], ch1[0], ch1[1]]
  const scales = [0.1, 0.2];
  const zeroPoints = [0, 0];
  const shape = [2, 2];

  const dequantized = dequantizeInt8(quantized, scales, zeroPoints, shape);

  // First channel uses scale 0.1
  assertAlmostEquals(dequantized[0], 0.0, 1e-5); // 0 * 0.1
  assertAlmostEquals(dequantized[1], 0.1, 1e-5); // 1 * 0.1
  // Second channel uses scale 0.2
  assertAlmostEquals(dequantized[2], 0.4, 1e-5); // 2 * 0.2
  assertAlmostEquals(dequantized[3], 0.6, 1e-5); // 3 * 0.2
});

Deno.test("unit: INT8 quantization roundtrip", () => {
  const original = new Float32Array([0.5, 1.5, -0.5, 2.5]);
  const scales = [0.1]; // Quantize to -0.1..0.1 range
  const zeroPoints = [0];

  // Quantize
  const quantized = quantizeToInt8(original, scales, zeroPoints);

  // Dequantize
  const restored = dequantizeInt8(quantized, scales, zeroPoints, [4]);

  // Check approximate equality (some precision loss expected)
  for (let i = 0; i < original.length; i++) {
    assertAlmostEquals(
      restored[i],
      original[i],
      0.15,
      `Index ${i}: expected ${original[i]}, got ${restored[i]}`,
    );
  }
});

Deno.test("unit: computeQuantizationParams single channel", () => {
  const data = new Float32Array([0.0, 1.0, 2.0, 3.0, 4.0]);
  const params = computeQuantizationParams(data, 1);

  assertEquals(params.scales.length, 1);
  assertEquals(params.zeroPoints.length, 1);

  const scale = params.scales[0];
  const zp = params.zeroPoints[0];

  // Range is [0, 4], quantized to [-128, 127]
  assert(scale > 0, "Scale should be positive");
  assert(zp >= -128 && zp <= 127, "Zero point should be in int8 range");
});

Deno.test("unit: computeQuantizationParams multi-channel", () => {
  // Two channels with different ranges: [0, 1, 2] (range=2) and [0, 10, 20] (range=20)
  const data = new Float32Array([0, 1, 2, 0, 10, 20]);
  const params = computeQuantizationParams(data, 2);

  assertEquals(params.scales.length, 2);
  assertEquals(params.zeroPoints.length, 2);

  // First channel has smaller range, should have smaller scale
  assert(
    params.scales[0] < params.scales[1],
    "First channel should have smaller scale",
  );
});

Deno.test("unit: QuantizationCache stores and retrieves weights", () => {
  const cache = new QuantizationCache();

  const weight = {
    data: new Int8Array([1, 2, 3]),
    shape: [3],
    metadata: {
      bits: 8,
      scales: [0.1],
      zeroPoints: [0],
      originalDtype: "float32",
      symmetric: true,
    },
  };

  cache.registerQuantized("layer.0.weight", weight);

  const retrieved = cache.getWeight("layer.0.weight");
  assert(retrieved !== undefined);
  assertEquals(retrieved.length, 3);
});

Deno.test("unit: QuantizationCache caches dequantized results", () => {
  const cache = new QuantizationCache();

  const weight = {
    data: new Int8Array([0, 1, 2]),
    shape: [3],
    metadata: {
      bits: 8,
      scales: [0.1],
      zeroPoints: [0],
      originalDtype: "float32",
      symmetric: true,
    },
  };

  cache.registerQuantized("test.weight", weight);

  // First call dequantizes
  const result1 = cache.getWeight("test.weight");

  // Second call should return cached result
  const result2 = cache.getWeight("test.weight");

  assert(result1 === result2, "Should return cached dequantized result");
});

Deno.test("unit: QuantizationCache reports statistics", () => {
  const cache = new QuantizationCache();

  const weight = {
    data: new Int8Array([1, 2, 3, 4, 5]),
    shape: [5],
    metadata: {
      bits: 8,
      scales: [0.1],
      zeroPoints: [0],
      originalDtype: "float32",
      symmetric: true,
    },
  };

  cache.registerQuantized("weight1", weight);
  cache.registerQuantized("weight2", weight);

  const stats = cache.getStats();
  assertEquals(stats.quantized, 2);
  assertEquals(stats.dequantized, 0); // Not yet dequantized

  // Trigger dequantization
  cache.getWeight("weight1");
  const stats2 = cache.getStats();
  assertEquals(stats2.dequantized, 1);

  // Clear cache
  cache.clearDequantizedCache();
  const stats3 = cache.getStats();
  assertEquals(stats3.dequantized, 0);
  assertEquals(stats3.quantized, 2); // Quantized still there
});

Deno.test("unit: INT8 quantization clamps values to int8 range", () => {
  const data = new Float32Array([0, 1000, -1000]); // Way outside int8 range
  const scales = [1.0]; // No scaling
  const zeroPoints = [0];

  const quantized = quantizeToInt8(data, scales, zeroPoints);

  // All values should be clamped to [-128, 127]
  for (let i = 0; i < quantized.length; i++) {
    assert(quantized[i] >= -128 && quantized[i] <= 127);
  }
});

Deno.test("unit: INT8 handles negative values correctly", () => {
  const quantized = new Int8Array([-128, -64, 0, 64, 127]);
  const scales = [0.1];
  const zeroPoints = [0];

  const dequantized = dequantizeInt8(quantized, scales, zeroPoints, [5]);

  assertAlmostEquals(dequantized[0], -12.8, 1e-5); // -128 * 0.1
  assertAlmostEquals(dequantized[2], 0.0, 1e-5); // 0 * 0.1
  assertAlmostEquals(dequantized[4], 12.7, 1e-5); // 127 * 0.1
});
