/**
 * @module runtime/safetensors_remote
 *
 * Range-lazy safetensors access (B2): read a checkpoint's header without
 * downloading weights, then fetch individual tensors with HTTP `Range`
 * requests. Servers without `Range` support fall back to full-body GET
 * plus slicing (correct, just not lazy).
 *
 * Layout: `u64LE header_len | JSON header | tensor data`. Header maps
 * tensor names to `{ dtype, shape, data_offsets }` relative to the data
 * section start (`8 + header_len`).
 */

/** One tensor's manifest entry: dtype, shape, and data-section byte range. */
export type TensorManifestEntry = {
  /** Safetensors dtype string (e.g. `"F32"`, `"BF16"`). */
  dtype: string;
  /** Tensor shape. */
  shape: number[];
  /** `[start, end)` byte offsets relative to the data section. */
  dataOffsets: [number, number];
};

/** Parsed safetensors header: tensor index plus section layout. */
export type SafetensorsManifest = {
  /** Tensor entries keyed by checkpoint tensor name. */
  tensors: Record<string, TensorManifestEntry>;
  /** Optional `__metadata__` header block. */
  metadata?: Record<string, string>;
  /** Byte offset where the data section starts (`8 + header_len`). */
  dataStart: number;
  /** Total header footprint (length prefix + JSON). */
  headerSize: number;
};

/** Options for ranged checkpoint fetches. */
export type RangeFetchOptions = {
  /** Abort signal for the underlying fetches. */
  signal?: AbortSignal;
  /** Attempts per request (default 2). */
  retries?: number;
};

type HeaderJson = Record<
  string,
  {
    dtype?: string;
    shape?: number[];
    data_offsets?: [number, number];
  } | Record<string, string>
>;

function parseHeaderBytes(bytes: Uint8Array): SafetensorsManifest {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerText = new TextDecoder().decode(
    bytes.subarray(8, 8 + headerLen),
  );
  const header = JSON.parse(headerText) as HeaderJson;
  const tensors: Record<string, TensorManifestEntry> = {};
  let metadata: Record<string, string> | undefined;
  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      metadata = value as Record<string, string>;
      continue;
    }
    const v = value as {
      dtype: string;
      shape: number[];
      data_offsets: [number, number];
    };
    tensors[key] = {
      dtype: v.dtype,
      shape: v.shape,
      dataOffsets: v.data_offsets,
    };
  }
  return {
    tensors,
    metadata,
    dataStart: 8 + headerLen,
    headerSize: 8 + headerLen,
  };
}

/** GET with byte range; tolerates servers that ignore `Range` (200). */
async function getRange(
  url: string,
  start: number,
  endInclusive: number,
  opts: RangeFetchOptions = {},
): Promise<Uint8Array> {
  const retries = opts.retries ?? 2;
  let lastError: unknown = null;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { Range: `bytes=${start}-${endInclusive}` },
        signal: opts.signal,
      });
      if (resp.status !== 206 && resp.status !== 200) {
        throw new Error(
          `Range request failed: ${resp.status} ${resp.statusText}`,
        );
      }
      const body = new Uint8Array(await resp.arrayBuffer());
      if (resp.status === 206) return body;
      // Server ignored Range — slice the needed window from the full body.
      return body.slice(start, endInclusive + 1);
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(`Range fetch failed for ${url}`);
}

/**
 * Fetch only a checkpoint's header and parse the tensor manifest.
 * Two small requests (8B length + JSON) instead of a multi-GB download.
 */
export async function fetchSafetensorsManifest(
  url: string,
  opts: RangeFetchOptions = {},
): Promise<SafetensorsManifest> {
  const prefix = await getRange(url, 0, 7, opts);
  if (prefix.length < 8) throw new Error("Checkpoint too small for header");
  const headerLen = Number(
    new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true),
  );
  const headerBytes = await getRange(url, 0, 8 + headerLen - 1, opts);
  return parseHeaderBytes(headerBytes);
}

/**
 * Fetch one tensor's payload bytes by manifest entry.
 * Returns a copy sized exactly to the tensor's byte range.
 */
export async function fetchTensorRange(
  url: string,
  entry: TensorManifestEntry,
  dataStart: number,
  opts: RangeFetchOptions = {},
): Promise<Uint8Array> {
  const [start, end] = entry.dataOffsets;
  const bytes = await getRange(
    url,
    dataStart + start,
    dataStart + end - 1,
    opts,
  );
  return bytes.slice(0, end - start);
}
