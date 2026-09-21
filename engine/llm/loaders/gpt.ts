import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";
import { blockUntilReady, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { GPT_CONFIG } from "../configs/gpt_config.ts";
import type { GptModel } from "../gpt.ts";

const mapper = new WeightMapper({
  prefix: { "h.": "layers." },
  substring: {
    ln_1: "ln1",
    ln_2: "ln2",
    ln_f: "lnF",
    c_attn: "cAttn",
    c_proj: "cProj",
    c_fc: "cFc",
  },
});

function tensorToArray(
  tensor: safetensors.Tensor,
  dtype: np.DType = np.float16,
): np.Array {
  if (tensor.dtype !== "F16" && tensor.dtype !== "F32") {
    throw new Error(
      `Expected F16 or F32 weights, but tensor has dtype ${tensor.dtype}.`,
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

export function gptFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<GptModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    if (key.endsWith(".attn.bias") || key.endsWith(".attn.masked_bias")) {
      continue;
    }
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }
  const model = safetensors.toNested(hydrated) as GptModel;
  if (model.layers.length !== GPT_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${GPT_CONFIG.numHiddenLayers} GPT-2 layers, found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
