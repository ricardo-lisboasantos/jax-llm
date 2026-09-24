/**
 * unit: paged MoE expert store tests (B1.3)
 *
 * Covers L0/L1/L2 roundtrip, device LRU eviction with rehydration,
 * and paged-vs-dense `runMoE` numerical equality on a tiny model.
 */
import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertRejects,
} from "@std/assert";
import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import {
  PagedExpertStore,
  type RawExpert,
  type RawTensor,
} from "./paged_experts.ts";
import { type MapleMoE, runMoE } from "./maple_moe.ts";
import { RamOffload } from "../../../runtime/offload/ram.ts";
import { NvmeOffload } from "../../../runtime/offload/nvme.ts";

const H = 4; // hidden
const I = 2; // intermediate

function f16Raw(values: number[], shape: number[]): RawTensor {
  const f16 = new Float16Array(values);
  const bytes = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
  return { dtype: "F16", shape, bytes };
}

function f32Raw(values: number[], shape: number[]): RawTensor {
  const f32 = new Float32Array(values);
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  return { dtype: "F32", shape, bytes };
}

function makeExpert(seed: number): { raw: RawExpert; vals: number[] } {
  // Layout is [out, in] (HF safetensors order): gate/up [I, H], down [H, I].
  const vals = Array.from(
    { length: I * H },
    (_, i) => ((seed * 7 + i * 13) % 9) - 4 + 0.5,
  );
  return {
    vals,
    raw: {
      gateProj: f16Raw(vals, [I, H]),
      upProj: f16Raw(vals.map((v) => -v), [I, H]),
      downProj: f16Raw(vals.slice(), [H, I]),
    },
  };
}

async function makeStore(
  deviceCap = 8,
  ramMaxBytes?: number,
): Promise<{ store: PagedExpertStore; dir: string }> {
  const dir = await Deno.makeTempDir();
  const store = new PagedExpertStore({
    modelId: "test-maple",
    layer: 0,
    deviceCap,
    ram: new RamOffload({ maxBytes: ramMaxBytes ?? 1 << 20 }),
    nvme: new NvmeOffload({ dir }),
  });
  return { store, dir };
}

function toArray(a: np.Array): number[] {
  return Array.from(a.dataSync() as ArrayLike<number>);
}

Deno.test("unit: paged experts stage + get roundtrip", async () => {
  const { store } = await makeStore();
  try {
    const { raw, vals } = makeExpert(1);
    await store.stage(0, raw);
    const mlp = await store.get(0);
    assertEquals(store.size, 1);
    const got = toArray(mlp.gateProj.weight);
    assertEquals(got.length, vals.length);
    for (let i = 0; i < vals.length; i++) {
      assertAlmostEquals(got[i], vals[i], 1e-3);
    }
  } finally {
    store.dispose();
  }
});

Deno.test("unit: paged experts evict LRU and rehydrate exact", async () => {
  const { store } = await makeStore(2);
  try {
    const e0 = makeExpert(1);
    const e1 = makeExpert(2);
    const e2 = makeExpert(3);
    await store.stage(0, e0.raw);
    await store.stage(1, e1.raw);
    await store.stage(2, e2.raw);
    await store.get(0);
    await store.get(1);
    assertEquals(store.size, 2);
    // Expert 2 evicts expert 0 (coldest).
    await store.get(2);
    assertEquals(store.size, 2);
    assertEquals(store.peek(0), undefined);
    assert(store.peek(1) !== undefined);
    assert(store.peek(2) !== undefined);
    // Rehydrated values are exact (backed by L1 bytes).
    const got = toArray((await store.get(0)).upProj.weight);
    const want = e0.vals.map((v) => -v);
    for (let i = 0; i < want.length; i++) {
      assertAlmostEquals(got[i], want[i], 1e-3);
    }
  } finally {
    store.dispose();
  }
});

Deno.test("unit: paged experts spill to disk tier past RAM budget", async () => {
  // 48B/expert raw; budget fits exactly one → rest spill to L2.
  const { store } = await makeStore(8, 48);
  try {
    const e0 = makeExpert(1);
    const e1 = makeExpert(5);
    await store.stage(0, e0.raw);
    await store.stage(1, e1.raw);
    // Expert 1 served from the disk tier.
    const got = toArray((await store.get(1)).gateProj.weight);
    for (let i = 0; i < e1.vals.length; i++) {
      assertAlmostEquals(got[i], e1.vals[i], 1e-3);
    }
  } finally {
    store.dispose();
  }
});

Deno.test("unit: paged experts missing key throws", async () => {
  const { store } = await makeStore();
  try {
    await assertRejects(() => store.get(7), Error, "L1+L2");
  } finally {
    store.dispose();
  }
});

Deno.test("unit: paged experts fetchMissing fills L1+L2 on miss", async () => {
  const dir = await Deno.makeTempDir();
  let calls = 0;
  const seed = makeExpert(9);
  const store = new PagedExpertStore({
    modelId: "test-maple",
    layer: 0,
    deviceCap: 8,
    ram: new RamOffload({ maxBytes: 1 << 20 }),
    nvme: new NvmeOffload({ dir }),
    fetchMissing: (_expert, proj) => {
      calls++;
      const t = seed.raw[proj];
      return Promise.resolve({
        dtype: t.dtype,
        shape: [...t.shape],
        bytes: t.bytes.slice(),
      });
    },
  });
  try {
    const got = toArray((await store.get(9)).gateProj.weight);
    for (let i = 0; i < seed.vals.length; i++) {
      assertAlmostEquals(got[i], seed.vals[i], 1e-3);
    }
    assertEquals(calls, 3);
    // Repeat hit stays local — no further remote calls (device tier).
    await store.get(9);
    assertEquals(calls, 3);
    // After device eviction, L1/L2 still serve locally.
    store.dispose();
    await store.get(9);
    assertEquals(calls, 3);
  } finally {
    store.dispose();
  }
});

Deno.test("unit: runMoE paged matches dense", async () => {
  // f32 throughout: the wasm unit-test backend cannot gather f16
  // (production runs f16 on WebGPU — shapes/ref flow is what we verify).
  const dir = await Deno.makeTempDir();
  const store = new PagedExpertStore({
    modelId: "test-maple",
    layer: 0,
    deviceCap: 256,
    ram: new RamOffload({ maxBytes: 1 << 20 }),
    nvme: new NvmeOffload({ dir }),
    dtype: np.float32,
  });
  const expertVals = (seed: number) =>
    Array.from(
      { length: I * H },
      (_, i) => ((seed * 7 + i * 13) % 9) - 4 + 0.5,
    );
  const f32Expert = (seed: number): RawExpert => {
    const vals = expertVals(seed);
    return {
      gateProj: f32Raw(vals, [I, H]),
      upProj: f32Raw(vals.map((v) => -v), [I, H]),
      downProj: f32Raw(vals.slice(), [H, I]),
    };
  };
  try {
    // Full 256-expert tiny model to satisfy MAPLE_CONFIG routing space.
    // Gate layout is [out, in] (HF safetensors order) — runLinear transposes.
    const gateVals = Array.from(
      { length: 256 * H },
      (_, i) => (((i * 11) % 23) - 11) * 0.25,
    );
    // One gate array per path: runLinear consumes the weight ref.
    const mkGate = () =>
      np.array(new Float32Array(gateVals), {
        shape: [256, H],
        dtype: np.float32,
      });
    const denseExperts = [];
    for (let e = 0; e < 256; e++) {
      const raw = f32Expert(e);
      await store.stage(e, raw);
      const w = (t: RawTensor, shape: number[]) =>
        np.array(new Float32Array(t.bytes.buffer as ArrayBuffer), {
          shape,
          dtype: np.float32,
        });
      denseExperts.push({
        gateProj: { weight: w(raw.gateProj, [I, H]) },
        upProj: { weight: w(raw.upProj, [I, H]) },
        downProj: { weight: w(raw.downProj, [H, I]) },
      });
    }
    const mkX = () =>
      np.array(new Float32Array([0.5, -1, 1.5, 0.25, 1, 0, -0.5, 2]), {
        shape: [2, H],
        dtype: np.float32,
      });
    const dense: MapleMoE = {
      gate: { weight: mkGate() },
      experts: denseExperts,
    };
    const paged: MapleMoE = {
      gate: { weight: mkGate() },
      experts: [],
      expertStore: store,
    };
    const want = toArray(await runMoE(dense, mkX()));
    const got = toArray(await runMoE(paged, mkX()));
    assertEquals(got.length, want.length);
    for (let i = 0; i < want.length; i++) {
      assertAlmostEquals(got[i], want[i], 1e-2);
    }
  } finally {
    store.dispose();
  }
});
