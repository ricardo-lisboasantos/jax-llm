import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { MAPLE_CONFIG } from "../configs/maple_config.ts";
import type { MapleModel } from "../maple.ts";
import {
  PagedExpertStore,
  type RawTensor,
} from "../layers/maple/paged_experts.ts";
import { RamOffload } from "../../runtime/offload/ram.ts";
import { NvmeOffload } from "../../runtime/offload/nvme.ts";

const mapper = new WeightMapper({
  prefix: { "model.": "" },
  substring: {
    word_embeddings: "wordEmbeddings",
    input_layernorm: "inputLayernorm",
    post_attention_layernorm: "postAttentionLayernorm",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    o_proj: "oProj",
    q_norm: "qNorm",
    k_norm: "kNorm",
    gate_proj: "gateProj",
    up_proj: "upProj",
    down_proj: "downProj",
    lm_head: "lmHead",
  },
});

function tensorToArray(
  tensor: safetensors.Tensor,
  dtype: np.DType = np.float16,
): np.Array {
  if (tensor.dtype !== "F16" && tensor.dtype !== "F32") {
    throw new Error(
      `Expected F16 or F32 (BF16-converted) weights, but tensor has dtype ${tensor.dtype}.`,
    );
  }
  switch (dtype) {
    case np.float16:
      return tensor.dtype === "F16"
        ? np.array(tensor.data as Float16Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float16,
        })
        : np.array(tensor.data as Float32Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float16,
        });
    case np.float32:
      return tensor.dtype === "F32"
        ? np.array(tensor.data as Float32Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float32,
        })
        : np.array(new Float32Array(tensor.data as Float16Array<ArrayBuffer>), {
          shape: tensor.shape,
          dtype: np.float32,
        });
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

export function mapleFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<MapleModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (key.endsWith(".scales")) continue;
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }
  const model = safetensors.toNested(hydrated) as MapleModel;
  if (model.layers.length !== MAPLE_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${MAPLE_CONFIG.numHiddenLayers} Maple layers, found ${model.layers.length}`,
    );
  }
  const firstExperts = model.layers[0].mlp.experts;
  if (firstExperts.length !== MAPLE_CONFIG.numExperts) {
    throw new Error(
      `Expected ${MAPLE_CONFIG.numExperts} experts per layer, found ${firstExperts.length}`,
    );
  }
  return blockUntilReady(model);
}

/** Raw HF projection names as they appear in sharded checkpoints. */
const EXPERT_KEY_RE =
  /\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)(?:\.weight)?$/;

/**
 * Classify a raw checkpoint key: MoE expert projection (with layer /
 * expert indices) or dense. Real checkpoints suffix weights with
 * `.weight` (e.g. `model.layers.0.mlp.experts.3.gate_proj.weight`).
 */
export function classifyMapleKey(key: string): {
  layer: number;
  expert: number;
  proj: "gateProj" | "upProj" | "downProj";
} | null {
  const m = EXPERT_KEY_RE.exec(key);
  if (!m) return null;
  return {
    layer: Number(m[1]),
    expert: Number(m[2]),
    proj: PROJ_CAMEL[m[3]],
  };
}

const PROJ_CAMEL: Record<string, "gateProj" | "upProj" | "downProj"> = {
  gate_proj: "gateProj",
  up_proj: "upProj",
  down_proj: "downProj",
};

export type PagedMapleOptions = {
  modelId?: string;
  deviceCap?: number;
  ram?: RamOffload;
  nvme?: NvmeOffload;
};

/**
 * Staging format version. Bump when the Nvme spill layout or sidecar
 * schema changes — old spills are then restaged instead of misread.
 */
const STAGING_VERSION = 1;

function stagedMarkerKey(modelId: string): string {
  return `${modelId}/staged.v${STAGING_VERSION}.ready`;
}

/**
 * Incremental Maple staging (B2): consume one parsed shard at a time.
 * Experts spill to L1/L2 as they arrive; dense weights accumulate in
 * `hydrated` (attention/norms/routers ≈ 1–2GB total, not 20B).
 * Shard buffers are releasable after each `stageShard` returns.
 */
export async function beginMaplePagedLoad(
  dtype: np.DType = np.float16,
  opts: PagedMapleOptions = {},
): Promise<{
  stageShard(file: safetensors.File): Promise<void>;
  finish(device: string): Promise<MapleModel>;
}> {
  const modelId = opts.modelId ?? "maple-preview";
  const ram = opts.ram ?? new RamOffload();
  const nvme = opts.nvme ?? new NvmeOffload();
  const ready = await nvme.get(stagedMarkerKey(modelId)).catch(
    () => undefined,
  );
  const stores: PagedExpertStore[] = [];
  for (let l = 0; l < MAPLE_CONFIG.numHiddenLayers; l++) {
    stores.push(
      new PagedExpertStore({
        modelId,
        layer: l,
        deviceCap: opts.deviceCap,
        ram,
        nvme,
        dtype,
      }),
    );
  }
  const hydrated: Record<string, np.Array> = {};
  const stagedExperts = new Set<string>();
  // Per-expert projection accumulator across shards (closure-local so
  // concurrent loads never share state).
  const pendingExpertParts = new Map<string, RawTensor>();

  return {
    async stageShard(file: safetensors.File): Promise<void> {
      const jobs: Promise<void>[] = [];
      for (const [key, tensor] of Object.entries(file.tensors)) {
        if (key.endsWith(".scales")) continue;
        const cls = classifyMapleKey(key);
        if (cls) {
          if (ready) continue;
          const tag = `${cls.layer}/${cls.expert}`;
          const view = tensor.data as
            | Float16Array<ArrayBuffer>
            | Float32Array<ArrayBuffer>;
          const bytes = new Uint8Array(
            view.buffer,
            view.byteOffset,
            view.byteLength,
          ).slice();
          const raw: RawTensor = {
            dtype: tensor.dtype as "F16" | "F32",
            shape: [...tensor.shape],
            bytes,
          };
          // Accumulate per-expert across shards; stage once complete.
          // Shards group a full expert's 3 projections together in
          // practice, so stage when this shard completes the triple.
          const accKey = `${tag}/${cls.proj}`;
          pendingExpertParts.set(accKey, raw);
          const triple: Record<string, RawTensor | undefined> = {
            gateProj: pendingExpertParts.get(`${tag}/gateProj`),
            upProj: pendingExpertParts.get(`${tag}/upProj`),
            downProj: pendingExpertParts.get(`${tag}/downProj`),
          };
          if (triple.gateProj && triple.upProj && triple.downProj) {
            if (!stagedExperts.has(tag)) {
              stagedExperts.add(tag);
              const parts = {
                gateProj: triple.gateProj,
                upProj: triple.upProj,
                downProj: triple.downProj,
              };
              pendingExpertParts.delete(`${tag}/gateProj`);
              pendingExpertParts.delete(`${tag}/upProj`);
              pendingExpertParts.delete(`${tag}/downProj`);
              jobs.push(stores[cls.layer].stage(cls.expert, parts));
            }
          }
          continue;
        }
        const mapped = mapper.mapKey(key);
        if (mapped in hydrated) {
          throw new Error(`Duplicate tensor "${key}" across shards`);
        }
        hydrated[mapped] = tensorToArray(tensor, dtype);
      }
      await Promise.all(jobs);
    },
    async finish(_device: string): Promise<MapleModel> {
      if (!ready) {
        const want = MAPLE_CONFIG.numHiddenLayers * MAPLE_CONFIG.numExperts;
        if (stagedExperts.size !== want) {
          throw new Error(
            `Incomplete Maple checkpoint: staged ${stagedExperts.size}/${want} experts`,
          );
        }
      }
      const model = safetensors.toNested(hydrated) as MapleModel;
      if (model.layers.length !== MAPLE_CONFIG.numHiddenLayers) {
        throw new Error(
          `Expected ${MAPLE_CONFIG.numHiddenLayers} Maple layers, found ${model.layers.length}`,
        );
      }
      await blockUntilReady(model);
      for (let l = 0; l < model.layers.length; l++) {
        model.layers[l].mlp.experts = [];
        model.layers[l].mlp.expertStore = stores[l];
      }
      if (!ready) {
        await nvme.put(
          stagedMarkerKey(modelId),
          new TextEncoder().encode(
            JSON.stringify({
              version: STAGING_VERSION,
              layers: model.layers.length,
              dtype,
            }),
          ),
        );
      }
      return model;
    },
  };
}

/**
 * Batch variant (compat for `loadModelPaged`): stage all shards, then finish.
 */
export async function mapleFromSafetensorsPaged(
  files: safetensors.File[],
  dtype: np.DType = np.float16,
  opts: PagedMapleOptions = {},
): Promise<MapleModel> {
  const staging = await beginMaplePagedLoad(dtype, opts);
  for (const file of files) await staging.stageShard(file);
  return await staging.finish("staged");
}
