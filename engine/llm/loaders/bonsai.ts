import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { BONSAI_CONFIG } from "../configs/bonsai_config.ts";
import type { BonsaiModel } from "../bonsai.ts";

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

export function bonsaiFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<BonsaiModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (key.endsWith(".scales")) continue;
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }
  const model = safetensors.toNested(hydrated) as BonsaiModel;
  if (model.layers.length !== BONSAI_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${BONSAI_CONFIG.numHiddenLayers} Bonsai layers, found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
