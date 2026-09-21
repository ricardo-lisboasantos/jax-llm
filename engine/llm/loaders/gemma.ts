import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { GEMMA_CONFIG } from "../configs/gemma_config.ts";
import type { GemmaModel } from "../gemma.ts";

const mapper = new WeightMapper({
  prefix: {
    "model.": "",
  },
  substring: {
    embed_tokens: "embedTokens",
    input_layernorm: "inputLayernorm",
    post_attention_layernorm: "postAttentionLayernorm",
    pre_feedforward_layernorm: "preFeedforwardLayernorm",
    post_feedforward_layernorm: "postFeedforwardLayernorm",
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
      // F32 (from BF16) → F16.
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
      // F16 → F32.
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        {
          shape: tensor.shape,
          dtype: np.float32,
        },
      );
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

export function fromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<GemmaModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as GemmaModel;
  if (model.layers.length !== GEMMA_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${GEMMA_CONFIG.numHiddenLayers} Gemma layers, ` +
        `found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
