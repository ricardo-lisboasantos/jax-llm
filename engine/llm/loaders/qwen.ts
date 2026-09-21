import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { QWEN_CONFIG } from "../configs/qwen_config.ts";
import type { QwenModel } from "../qwen.ts";

const mapper = new WeightMapper({
  prefix: { "model.": "" },
  substring: {
    embed_tokens: "embedTokens",
    input_layernorm: "inputLayernorm",
    post_attention_layernorm: "postAttentionLayernorm",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    o_proj: "oProj",
    gate_proj: "gateProj",
    up_proj: "upProj",
    down_proj: "downProj",
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

export function qwenFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<QwenModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }
  const model = safetensors.toNested(hydrated) as QwenModel;
  if (model.layers.length !== QWEN_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${QWEN_CONFIG.numHiddenLayers} Qwen2 layers, found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
