import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import type { LfmModel } from "../lfm.ts";
import { LFM_CONFIG } from "../configs/lfm_config.ts";

const mapper = new WeightMapper({
  prefix: {
    "model.": "",
  },
  substring: {
    embed_tokens: "embedTokens",
    embedding_norm: "embeddingNorm",
    operator_norm: "operatorNorm",
    ffn_norm: "ffnNorm",
    feed_forward: "feedForward",
    in_proj: "inProj",
    out_proj: "outProj",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    q_layernorm: "qLayernorm",
    k_layernorm: "kLayernorm",
  },
});

function tensorToArray(
  tensor: safetensors.Tensor,
  dtype: np.DType = np.float16,
): np.Array {
  // Accept F16 (native) and F32 (from BF16 conversion by parseSafetensors).
  if (tensor.dtype !== "F16" && tensor.dtype !== "F32") {
    throw new Error(
      `Expected F16 or F32 (BF16-converted) weights, but tensor has dtype ${tensor.dtype}.`,
    );
  }
  switch (dtype) {
    case np.float16:
      if (tensor.dtype === "F16") {
        return np.array(tensor.data as Float16Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float16,
        });
      }
      // F32 (from BF16) → F16: truncate mantissa precision.
      return np.array(tensor.data as Float32Array<ArrayBuffer>, {
        shape: tensor.shape,
        dtype: np.float16,
      });
    case np.float32:
      if (tensor.dtype === "F32") {
        return np.array(tensor.data as Float32Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float32,
        });
      }
      // F16 → F32: upcast.
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        { shape: tensor.shape, dtype: np.float32 },
      );
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

export function lfmFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<LfmModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as LfmModel;
  if (model.layers.length !== LFM_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${LFM_CONFIG.numHiddenLayers} LFM2.5 layers, ` +
        `found ${model.layers.length}`,
    );
  }

  return blockUntilReady(model);
}
