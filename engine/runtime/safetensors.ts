/**
 * @module runtime/safetensors
 *
 * Safetensors parser with BF16 support.
 *
 * The `@jax-js/loaders` safetensors parser does not support BF16 (bfloat16)
 * tensors. This module wraps the existing parser and adds BF16 → F32
 * conversion, since BF16 is just the upper 16 bits of a float32.
 *
 * The returned object is structurally compatible with `safetensors.File`
 * from `@jax-js/loaders`, so it can be used as a drop-in replacement.
 */

import { safetensors } from "npm:@jax-js/loaders@^0.1.3";

/**
 * Parse a safetensors file, with support for BF16 tensors.
 *
 * BF16 tensors are converted to Float32Array (with dtype "F32") since
 * BF16 is just the upper 16 bits of float32 — the conversion is a simple
 * bit shift with no precision loss.
 *
 * All other dtypes are handled by the upstream `safetensors.parse()`.
 *
 * Returns a `safetensors.File` — structurally identical to what
 * `safetensors.parse()` returns, so it can be used as a drop-in.
 */
export function parseSafetensors(
  data: Uint8Array<ArrayBuffer>,
): safetensors.File {
  // First, check if the file contains any BF16 tensors by reading the header.
  const buffer = data.buffer;
  const ptr = data.byteOffset;
  const headerSize = new DataView(buffer, ptr, 8).getBigUint64(0, true);
  const headerText = new TextDecoder().decode(
    new Uint8Array(buffer, ptr + 8, Number(headerSize)),
  );
  const header = JSON.parse(headerText);

  // Check if any tensor is BF16.
  let hasBf16 = false;
  for (const [, value] of Object.entries(header)) {
    if (
      typeof value === "object" && value !== null &&
      "dtype" in value && (value as { dtype: string }).dtype === "BF16"
    ) {
      hasBf16 = true;
      break;
    }
  }

  if (!hasBf16) {
    // No BF16 tensors — use the upstream parser directly.
    return safetensors.parse(data);
  }

  // BF16 tensors present — parse manually with BF16 support.
  return parseWithBf16Support(data, header);
}

/**
 * Full safetensors parser with BF16 → F32 conversion.
 */
function parseWithBf16Support(
  data: Uint8Array<ArrayBuffer>,
  header: Record<string, unknown>,
): safetensors.File {
  const buffer = data.buffer;
  const ptr = data.byteOffset;
  const headerSize = new DataView(buffer, ptr, 8).getBigUint64(0, true);
  const dataStart = ptr + Number(headerSize) + 8;

  const file: safetensors.File = {
    tensors: {},
    totalSize: data.byteLength,
  };

  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      file.metadata = value as Record<string, string>;
      continue;
    }

    const { dtype, shape, data_offsets } = value as {
      dtype: string;
      shape: number[];
      data_offsets: [number, number];
    };

    const byteOffset = dataStart + data_offsets[0];
    const byteLength = data_offsets[1] - data_offsets[0];

    let tensorData: safetensors.TensorData;

    switch (dtype) {
      case "BF16":
        // BF16 → F32: each 2-byte BF16 value becomes a 4-byte F32 value
        // by placing the BF16 bits in the upper 16 bits of the F32.
        tensorData = bf16ToF32(new Uint8Array(buffer, byteOffset, byteLength));
        break;
      case "F16":
        tensorData = alignedData(
          Float16Array,
          buffer,
          byteOffset,
          byteLength / 2,
        );
        break;
      case "F32":
        tensorData = alignedData(
          Float32Array,
          buffer,
          byteOffset,
          byteLength / 4,
        );
        break;
      case "F64":
        tensorData = alignedData(
          Float64Array,
          buffer,
          byteOffset,
          byteLength / 8,
        );
        break;
      case "I8":
        tensorData = new Int8Array(buffer, byteOffset, byteLength);
        break;
      case "I16":
        tensorData = alignedData(
          Int16Array,
          buffer,
          byteOffset,
          byteLength / 2,
        );
        break;
      case "I32":
        tensorData = alignedData(
          Int32Array,
          buffer,
          byteOffset,
          byteLength / 4,
        );
        break;
      case "I64":
        tensorData = alignedData(
          BigInt64Array,
          buffer,
          byteOffset,
          byteLength / 8,
        );
        break;
      case "U8":
        tensorData = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      case "U16":
        tensorData = alignedData(
          Uint16Array,
          buffer,
          byteOffset,
          byteLength / 2,
        );
        break;
      case "U32":
        tensorData = alignedData(
          Uint32Array,
          buffer,
          byteOffset,
          byteLength / 4,
        );
        break;
      case "U64":
        tensorData = alignedData(
          BigUint64Array,
          buffer,
          byteOffset,
          byteLength / 8,
        );
        break;
      case "BOOL":
        tensorData = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }

    // For BF16, report as F32 since we converted the data.
    file.tensors[key] = {
      dtype: (dtype === "BF16" ? "F32" : dtype) as safetensors.DType,
      shape,
      data: tensorData,
    };
  }

  return file;
}

/**
 * Convert BF16 byte data to a Float32Array.
 *
 * BF16 has the same 8-bit exponent as F32 but only 7 mantissa bits
 * (vs F32's 23). Converting BF16 → F32 is lossless: just place the
 * 16 BF16 bits into the upper 16 bits of a 32-bit float and zero
 * the lower 16 bits.
 */
function bf16ToF32(bytes: Uint8Array): Float32Array<ArrayBuffer> {
  const count = bytes.length / 2;
  const result = new Float32Array(count);
  // View the result bytes as uint32 for bit manipulation.
  const resultU32 = new Uint32Array(result.buffer);
  // Bulk Uint16 view avoids per-element DataView overhead (~3-5x faster).
  // Copy only when source is unaligned for Uint16 access.
  let u16: Uint16Array;
  if (bytes.byteOffset % 2 === 0) {
    u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, count);
  } else {
    u16 = new Uint16Array(bytes.slice().buffer, 0, count);
  }
  for (let i = 0; i < count; i++) {
    // BF16 → F32: place 16-bit BF16 in upper 16 bits of F32.
    resultU32[i] = u16[i] << 16;
  }
  return result;
}

/**
 * Replicate the alignedData helper from @jax-js/loaders.
 *
 * `ctor` is typed as a generic buffer-backed constructor (instead of a union
 * of `typeof` array types) so `new ctor(buffer, byteOffset, length)` stays
 * well-typed across TS lib versions — unions of constructor types only
 * expose their common (0-1 argument) overloads to `new`, which newer
 * checkers (e.g. `jsr publish`) reject.
 */
function alignedData<T extends ArrayBufferView>(
  ctor: {
    new (
      buffer: ArrayBufferLike,
      byteOffset?: number,
      length?: number,
    ): T;
    readonly BYTES_PER_ELEMENT: number;
  },
  buffer: ArrayBuffer,
  byteOffset: number,
  length: number,
): T {
  if (byteOffset % ctor.BYTES_PER_ELEMENT === 0) {
    return new ctor(buffer, byteOffset, length);
  }
  const byteLength = length * ctor.BYTES_PER_ELEMENT;
  return new ctor(
    new Uint8Array(buffer, byteOffset, byteLength).slice().buffer,
    0,
    length,
  );
}
