/**
 * unit: Range-lazy safetensors remote tests (B2/T1).
 * Served from a local `Deno.serve` with synthetic checkpoint bytes —
 * no network, no weights.
 */
import { assertEquals } from "@std/assert";
import {
  fetchSafetensorsManifest,
  fetchTensorRange,
} from "./safetensors_remote.ts";

function buildCheckpoint(): {
  bytes: Uint8Array;
  a: Float32Array;
  b: Float32Array;
} {
  const a = new Float32Array([1, 2, 3, 4]);
  const b = new Float32Array([5, 6, 7]);
  const header = JSON.stringify({
    a: { dtype: "F32", shape: [2, 2], data_offsets: [0, 16] },
    b: { dtype: "F32", shape: [3], data_offsets: [16, 28] },
  });
  const headerBytes = new TextEncoder().encode(header);
  const out = new Uint8Array(8 + headerBytes.length + 28);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  out.set(new Uint8Array(a.buffer), 8 + headerBytes.length);
  out.set(new Uint8Array(b.buffer), 8 + headerBytes.length + 16);
  return { bytes: out, a, b };
}

function rangeSlice(
  bytes: Uint8Array,
  range: string | null,
  honor: boolean,
): { status: number; body: Uint8Array; headers: Record<string, string> } {
  if (!honor || !range) {
    return { status: 200, body: bytes, headers: {} };
  }
  const m = /bytes=(\d+)-(\d+)/.exec(range);
  if (!m) return { status: 416, body: new Uint8Array(), headers: {} };
  const start = Number(m[1]);
  const end = Math.min(Number(m[2]), bytes.length - 1);
  return {
    status: 206,
    body: bytes.slice(start, end + 1),
    headers: { "Content-Range": `bytes ${start}-${end}/${bytes.length}` },
  };
}

async function withServer(
  fn: (base: string) => Promise<void>,
): Promise<void> {
  const { bytes } = buildCheckpoint();
  const server = Deno.serve({ port: 0, onListen: () => {} }, (req) => {
    const url = new URL(req.url);
    const honor = url.pathname !== "/full";
    const r = rangeSlice(bytes, req.headers.get("range"), honor);
    return new Response(r.body.buffer as ArrayBuffer, {
      status: r.status,
      headers: r.headers,
    });
  });
  try {
    const base = `http://127.0.0.1:${(server.addr as Deno.NetAddr).port}`;
    await fn(base);
  } finally {
    await server.shutdown();
  }
}

Deno.test("unit: remote manifest parses header only", async () => {
  await withServer(async (base) => {
    const manifest = await fetchSafetensorsManifest(
      `${base}/model.safetensors`,
    );
    assertEquals(Object.keys(manifest.tensors).sort(), ["a", "b"]);
    assertEquals(manifest.tensors["a"], {
      dtype: "F32",
      shape: [2, 2],
      dataOffsets: [0, 16],
    });
    assertEquals(manifest.tensors["b"].shape, [3]);
  });
});

Deno.test("unit: remote tensor Range fetch returns exact bytes", async () => {
  await withServer(async (base) => {
    const url = `${base}/model.safetensors`;
    const manifest = await fetchSafetensorsManifest(url);
    const { bytes } = buildCheckpoint();
    for (const name of ["a", "b"] as const) {
      const entry = manifest.tensors[name];
      const got = await fetchTensorRange(url, entry, manifest.dataStart);
      const [start, end] = entry.dataOffsets;
      const want = bytes.slice(
        manifest.dataStart + start,
        manifest.dataStart + end,
      );
      assertEquals(got, want);
    }
  });
});

Deno.test("unit: remote fetch falls back when server ignores Range", async () => {
  await withServer(async (base) => {
    const url = `${base}/full/model.safetensors`;
    const manifest = await fetchSafetensorsManifest(url);
    assertEquals(Object.keys(manifest.tensors).sort(), ["a", "b"]);
    const entry = manifest.tensors["a"];
    const got = await fetchTensorRange(url, entry, manifest.dataStart);
    assertEquals(got.length, 16);
    assertEquals(new Float32Array(got.buffer)[0], 1);
  });
});
