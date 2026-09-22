/**
 * unit: WASM SIMD kernel tests
 */
import { assertEquals, assert, assertAlmostEquals } from "@std/assert";
import {
  isWasmSimdAvailable,
  simdRelu,
  simdGelu,
  simdSoftmax,
  simdLayernorm,
  simdMatvec,
  getSimdBackend,
} from "./wasm_simd.ts";

Deno.test("unit: WASM SIMD availability check", () => {
  // Should not throw
  const available = isWasmSimdAvailable();
  assertEquals(typeof available, "boolean");
});

Deno.test("unit: SIMD ReLU activation", () => {
  const input = new Float32Array([-2, -1, 0, 1, 2, 3]);
  const output = simdRelu(input);

  assertEquals(output[0], 0); // max(0, -2) = 0
  assertEquals(output[1], 0); // max(0, -1) = 0
  assertEquals(output[2], 0); // max(0, 0) = 0
  assertEquals(output[3], 1); // max(0, 1) = 1
  assertEquals(output[4], 2); // max(0, 2) = 2
  assertEquals(output[5], 3); // max(0, 3) = 3
});

Deno.test("unit: SIMD ReLU handles odd-length arrays", () => {
  const input = new Float32Array([-1, 0, 1, 2, 3]);
  const output = simdRelu(input);

  assertEquals(output.length, 5);
  assertEquals(output[0], 0);
  assertEquals(output[4], 3);
});

Deno.test("unit: SIMD GeLU activation is smooth", () => {
  // Use positive range where GeLU is strictly monotonic
  const input = new Float32Array([0, 0.5, 1, 1.5, 2]);
  const output = simdGelu(input);

  // GeLU should be monotonically increasing in positive range
  assert(output[0] < output[1]);
  assert(output[1] < output[2]);
  assert(output[2] < output[3]);
  assert(output[3] < output[4]);

  // GeLU(0) should be close to 0
  assertAlmostEquals(output[0], 0, 0.05);
});

Deno.test("unit: SIMD Softmax sums to 1", () => {
  const input = new Float32Array([1, 2, 3, 4]);
  const output = simdSoftmax(input, 4);

  let sum = 0;
  for (const val of output) {
    assert(val >= 0 && val <= 1);
    sum += val;
  }

  assertAlmostEquals(sum, 1.0, 0.0001, "Softmax should sum to 1");
});

Deno.test("unit: SIMD Softmax multi-row", () => {
  // Two rows: [1,2,3] and [4,5,6]
  const input = new Float32Array([1, 2, 3, 4, 5, 6]);
  const output = simdSoftmax(input, 3);

  // First row should sum to 1
  const sum1 = output[0] + output[1] + output[2];
  assertAlmostEquals(sum1, 1.0, 0.0001);

  // Second row should sum to 1
  const sum2 = output[3] + output[4] + output[5];
  assertAlmostEquals(sum2, 1.0, 0.0001);

  // All values should be in (0, 1)
  for (const val of output) {
    assert(val > 0 && val < 1);
  }
});

Deno.test("unit: SIMD Softmax is invariant to additive shift", () => {
  const input1 = new Float32Array([1, 2, 3]);
  const input2 = new Float32Array([11, 12, 13]); // Add 10 to each element

  const output1 = simdSoftmax(input1, 3);
  const output2 = simdSoftmax(input2, 3);

  // Adding a constant to all inputs should not change softmax output
  for (let i = 0; i < 3; i++) {
    assertAlmostEquals(output1[i], output2[i], 0.0001);
  }
});

Deno.test("unit: SIMD LayerNorm normalizes to unit variance", () => {
  const input = new Float32Array([1, 2, 3, 4, 5, 6]);
  const weight = new Float32Array([1, 1, 1, 1, 1, 1]);
  const bias = new Float32Array([0, 0, 0, 0, 0, 0]);

  const output = simdLayernorm(input, 3, weight, bias);

  // First row: [1, 2, 3]
  // mean = 2, std = sqrt(0.67) ≈ 0.816
  // normalized: [-1.22, 0, 1.22]
  const mean1 = (output[0] + output[1] + output[2]) / 3;
  assertAlmostEquals(mean1, 0, 0.01, "First row mean should be ~0");

  // Compute variance
  let var1 = 0;
  for (let i = 0; i < 3; i++) {
    var1 += output[i] * output[i];
  }
  var1 /= 3;
  assertAlmostEquals(var1, 1, 0.1, "First row variance should be ~1");
});

Deno.test("unit: SIMD LayerNorm applies scale and bias", () => {
  const input = new Float32Array([1, 2, 3]);
  const weight = new Float32Array([2, 2, 2]);
  const bias = new Float32Array([1, 1, 1]);

  const output = simdLayernorm(input, 3, weight, bias);

  // Output should be: normalized * weight + bias
  // normalized values have mean 0, variance 1
  // So output should have mean = bias = 1
  const mean = (output[0] + output[1] + output[2]) / 3;
  assertAlmostEquals(mean, 1, 0.1, "Mean should be close to bias");
});

Deno.test("unit: SIMD Matvec computes correct output", () => {
  // Matrix: [[1, 2], [3, 4]]
  // Vector: [5, 6]
  // Expected: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
  const matrix = new Float32Array([1, 2, 3, 4]);
  const vector = new Float32Array([5, 6]);

  const output = simdMatvec(matrix, 2, vector);

  assertEquals(output.length, 2);
  assertEquals(output[0], 17);
  assertEquals(output[1], 39);
});

Deno.test("unit: SIMD Matvec with larger matrix", () => {
  // 3x2 matrix
  const matrix = new Float32Array([1, 2, 3, 4, 5, 6]);
  const vector = new Float32Array([1, 1]);

  const output = simdMatvec(matrix, 2, vector);

  assertEquals(output[0], 3); // 1 + 2
  assertEquals(output[1], 7); // 3 + 4
  assertEquals(output[2], 11); // 5 + 6
});

Deno.test("unit: getSimdBackend returns valid option", () => {
  const backend = getSimdBackend();
  assertEquals(backend === "simd" || backend === "scalar", true);
});
