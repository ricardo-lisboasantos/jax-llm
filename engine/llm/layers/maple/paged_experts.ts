/**
 * @module llm/layers/maple/paged_experts
 *
 * Three-tier MoE expert paging for Maple (256 experts/layer, top-8 routed).
 *
 * Tiers:
 * - L0 device: per-layer LRU of hydrated `ExpertMLP` (default 16 slots —
 *   decode touches 8, short prefill rarely exceeds 16 unique).
 * - L1 RAM: `RamOffload` raw-bytes LRU (default 1 GiB) — sync hits.
 * - L2 disk: `NvmeOffload` per-tensor files under `.opfs/offload` — async.
 *
 * Raw bytes are the safetensors payloads (F16/F32 + shape); hydration is
 * `np.array(view, { shape, dtype })`, matching `loaders/maple.ts`.
 * Disposal uses `tree.dispose` so jax-js refcounts stay balanced.
 */

import { numpy as np, tree } from "npm:@jax-js/jax@^0.1.25";
import { RamOffload } from "../../../runtime/offload/ram.ts";
import { NvmeOffload } from "../../../runtime/offload/nvme.ts";
import type { ExpertMLP } from "./maple_mlp.ts";

/** Raw expert projection weights as stored in L1/L2 (safetensors payload). */
export type RawTensor = {
  dtype: "F16" | "F32";
  shape: number[];
  bytes: Uint8Array;
};

/** Raw expert = 3 SwiGLU projections (gate/up/down). */
export type RawExpert = {
  gateProj: RawTensor;
  upProj: RawTensor;
  downProj: RawTensor;
};

export type PagedExpertsConfig = {
  /** Model id prefix for L1/L2 keys (e.g. `"maple-preview"`). */
  modelId: string;
  /** Decoder layer index (scoped into cache keys). */
  layer: number;
  /** Device-resident expert slots per layer (default 16). */
  deviceCap?: number;
  /** Shared RAM tier (created per store when omitted). */
  ram?: RamOffload;
  /** Shared disk tier (created per store when omitted). */
  nvme?: NvmeOffload;
  /** Target device dtype for hydration. */
  dtype?: np.DType;
  /**
   * L1+L2 miss fallback (B2): fetch one projection's raw bytes remotely
   * (e.g. HF `Range` request) instead of throwing. Fetched bytes are
   * staged into L1/L2 so repeat hits stay local.
   */
  fetchMissing?: (
    expert: number,
    proj: "gateProj" | "upProj" | "downProj",
  ) => Promise<RawTensor>;
};

function rawByteSize(t: RawTensor): number {
  return t.bytes.byteLength;
}

function expertByteSize(e: RawExpert): number {
  return rawByteSize(e.gateProj) + rawByteSize(e.upProj) +
    rawByteSize(e.downProj);
}

/** View raw bytes as the typed array the device loader expects. */
function viewRaw(
  t: RawTensor,
): Float16Array<ArrayBuffer> | Float32Array<ArrayBuffer> {
  // Copy on unaligned boundaries so `np.array` never sees a torn view.
  if (t.dtype === "F16") {
    if (t.bytes.byteOffset % 2 === 0) {
      return new Float16Array(
        t.bytes.buffer as ArrayBuffer,
        t.bytes.byteOffset,
        t.bytes.byteLength / 2,
      );
    }
    return new Float16Array(t.bytes.slice().buffer as ArrayBuffer);
  }
  if (t.bytes.byteOffset % 4 === 0) {
    return new Float32Array(
      t.bytes.buffer as ArrayBuffer,
      t.bytes.byteOffset,
      t.bytes.byteLength / 4,
    );
  }
  return new Float32Array(t.bytes.slice().buffer as ArrayBuffer);
}

/**
 * Device-resident LRU of hydrated experts with RAM + disk backing.
 * All hydration funnels through `get()`; eviction disposes device arrays.
 */
export class PagedExpertStore {
  readonly layer: number;
  readonly deviceCap: number;
  readonly ram: RamOffload;
  readonly nvme: NvmeOffload;
  private readonly fetchMissing:
    | ((
      expert: number,
      proj: "gateProj" | "upProj" | "downProj",
    ) => Promise<RawTensor>)
    | undefined;
  private readonly modelId: string;
  private readonly dtype: np.DType;
  private slots = new Map<number, ExpertMLP>();
  private lru: number[] = [];
  private ramBytes = 0;

  constructor(config: PagedExpertsConfig) {
    this.modelId = config.modelId;
    this.layer = config.layer;
    this.deviceCap = config.deviceCap ?? 16;
    this.ram = config.ram ?? new RamOffload();
    this.nvme = config.nvme ?? new NvmeOffload();
    this.dtype = config.dtype ?? np.float16;
    this.fetchMissing = config.fetchMissing;
  }

  /** Cache key for one expert projection. */
  key(expert: number, proj: "gateProj" | "upProj" | "downProj"): string {
    return `${this.modelId}/l${this.layer}e${expert}${proj}`;
  }

  /** Number of device-resident experts. */
  get size(): number {
    return this.slots.size;
  }

  /** Stage raw expert bytes into L1 (+L2 when RAM is over budget). */
  async stage(expert: number, raw: RawExpert): Promise<void> {
    const size = expertByteSize(raw);
    const projections = Object.entries(raw) as [keyof RawExpert, RawTensor][];
    // Spill to disk first when the new expert would overflow RAM, so L1
    // stays within budget without a second pass.
    const spill = this.ramBytes + size > this.ram.maxBytes;
    for (const [proj, tensor] of projections) {
      const bytes = tensor.bytes.slice();
      const stored: RawTensor = {
        dtype: tensor.dtype,
        shape: [...tensor.shape],
        bytes,
      };
      if (!spill) {
        this.ram.put(this.key(expert, proj), bytes);
        this.ramBytes += bytes.byteLength;
        // Persist shape/dtype sidecar alongside; raw map holds bytes only.
        await this.nvme.put(
          `${this.key(expert, proj)}.meta`,
          encodeSidecar(stored),
        );
      } else {
        await this.nvme.put(this.key(expert, proj), bytes);
        await this.nvme.put(
          `${this.key(expert, proj)}.meta`,
          encodeSidecar(stored),
        );
      }
    }
  }

  /**
   * Get a device-resident expert, hydrating through L1 → L2 on miss.
   * Touches LRU on hit; evicts the coldest slot when over capacity.
   */
  async get(expert: number): Promise<ExpertMLP> {
    const hit = this.slots.get(expert);
    if (hit) {
      this.touch(expert);
      return hit;
    }
    const raw = await this.loadRaw(expert);
    const mlp: ExpertMLP = {
      gateProj: { weight: this.hydrate(raw.gateProj) },
      upProj: { weight: this.hydrate(raw.upProj) },
      downProj: { weight: this.hydrate(raw.downProj) },
    };
    this.insert(expert, mlp);
    return mlp;
  }

  /** Device-resident without hydration (undefined on miss). */
  peek(expert: number): ExpertMLP | undefined {
    return this.slots.get(expert);
  }

  /** Release all device slots (backing bytes in L1/L2 are retained). */
  dispose(): void {
    for (const mlp of this.slots.values()) {
      // Workaround: jax-js refcount quirk can throw UseAfterFreeError on
      // fresh arrays — slots are dropped regardless (repo-wide convention).
      try {
        tree.dispose(mlp);
      } catch {
        // Slot dropped regardless.
      }
    }
    this.slots.clear();
    this.lru = [];
  }

  // ── Internal ─────────────────────────────────────────────────────────

  private touch(expert: number): void {
    const i = this.lru.indexOf(expert);
    if (i >= 0) this.lru.splice(i, 1);
    this.lru.push(expert);
  }

  private insert(expert: number, mlp: ExpertMLP): void {
    this.slots.set(expert, mlp);
    this.touch(expert);
    while (this.slots.size > this.deviceCap) {
      const cold = this.lru.shift()!;
      const evicted = this.slots.get(cold);
      this.slots.delete(cold);
      if (evicted) {
        try {
          tree.dispose(evicted);
        } catch {
          // Upstream refcount quirk — slot is dropped regardless.
        }
      }
    }
  }

  private hydrate(t: RawTensor): np.Array {
    const data = viewRaw(t);
    if (this.dtype === np.float16) {
      return np.array(data as Float16Array<ArrayBuffer>, {
        shape: t.shape,
        dtype: np.float16,
      });
    }
    const f32 = t.dtype === "F32"
      ? (data as Float32Array<ArrayBuffer>)
      : new Float32Array(data as Float16Array<ArrayBuffer>);
    return np.array(f32, { shape: t.shape, dtype: np.float32 });
  }

  private async loadRaw(expert: number): Promise<RawExpert> {
    const projs: ("gateProj" | "upProj" | "downProj")[] = [
      "gateProj",
      "upProj",
      "downProj",
    ];
    const out = {} as RawExpert;
    for (const proj of projs) {
      const ramHit = this.ram.get(this.key(expert, proj));
      if (ramHit) {
        const meta = await this.nvme.get(`${this.key(expert, proj)}.meta`);
        out[proj] = decodeSidecar(meta!, ramHit);
        continue;
      }
      const bytes = await this.nvme.get(this.key(expert, proj));
      if (bytes) {
        const meta = await this.nvme.get(`${this.key(expert, proj)}.meta`);
        out[proj] = decodeSidecar(meta!, bytes);
        continue;
      }
      // B2 miss path: remote fetch, then stage locally for repeat hits.
      if (this.fetchMissing) {
        const raw = await this.fetchMissing(expert, proj);
        const staged = raw.bytes.slice();
        this.ram.put(this.key(expert, proj), staged);
        this.ramBytes += staged.byteLength;
        await this.nvme.put(
          `${this.key(expert, proj)}.meta`,
          encodeSidecar({ dtype: raw.dtype, shape: raw.shape, bytes: staged }),
        );
        out[proj] = { dtype: raw.dtype, shape: [...raw.shape], bytes: staged };
        continue;
      }
      throw new Error(
        `Paged expert missing from L1+L2: ${this.key(expert, proj)}`,
      );
    }
    return out;
  }
}

/** Encode RawTensor shape/dtype sidecar (bytes live in the sibling key). */
function encodeSidecar(t: RawTensor): Uint8Array {
  return new TextEncoder().encode(JSON.stringify({
    dtype: t.dtype,
    shape: t.shape,
    byteLength: t.bytes.byteLength,
  }));
}

/** Reattach sidecar metadata to fetched bytes. */
function decodeSidecar(
  meta: Uint8Array | undefined,
  bytes: Uint8Array,
): RawTensor {
  if (!meta) throw new Error("Paged expert sidecar missing");
  const parsed = JSON.parse(new TextDecoder().decode(meta)) as {
    dtype: "F16" | "F32";
    shape: number[];
  };
  return { dtype: parsed.dtype, shape: parsed.shape, bytes };
}
