/**
 * @module llm/gpt
 *
 * GPT-2 architecture — the classic decoder-only transformer with
 * LayerNorm (pre-LN with bias), GeLU activation, learned absolute
 * positional embeddings, and Conv1D linear layers (weight stored as
 * [in, out], so no transpose is needed).  Embeddings and LM head are
 * tied.  There is no RoPE or GQA — all attention heads are full MHA.
 *
 * Reference: https://huggingface.co/gpt2
 */

import { blockUntilReady, jit, nn, numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { safetensors, WeightMapper } from "npm:@jax-js/loaders@^0.1.3";

export const GPT_CONFIG = {
  bosTokenId: 50_256,
  eosTokenId: 50_256,
  vocabSize: 50_257,
  hiddenSize: 768, // n_embd
  intermediateSize: 3_072, // 4 * n_embd
  numHiddenLayers: 12, // n_layer
  numAttentionHeads: 12, // n_head
  headDim: 64, // hiddenSize / numAttentionHeads
  maxPositions: 1_024, // n_positions
  layerNormEps: 1e-5,
} as const;

/**
 * GPT-2 "Conv1D" — equivalent to a linear layer but the weight is stored
 * as [in, out] (not [out, in]), so the forward pass is `x @ weight + bias`
 * with no transpose.
 */
export type Conv1D = {
  weight: np.Array; // [in, out]
  bias: np.Array;
};

export type LayerNorm = {
  weight: np.Array;
  bias: np.Array;
};

export type GptMLP = {
  cFc: Conv1D;
  cProj: Conv1D;
};

export type GptAttention = {
  cAttn: Conv1D; // combined QKV projection: [hidden, 3 * hidden]
  cProj: Conv1D; // output projection: [hidden, hidden]
};

export type GptBlock = {
  ln1: LayerNorm;
  attn: GptAttention;
  ln2: LayerNorm;
  mlp: GptMLP;
};

export type GptModel = {
  wte: { weight: np.Array }; // token embeddings [vocab, hidden]
  wpe: { weight: np.Array }; // position embeddings [max_positions, hidden]
  layers: GptBlock[];
  lnF: LayerNorm;
};

export type GptKVCache = {
  key: np.Array; // [capacity, num_heads, head_dim]
  value: np.Array; // [capacity, num_heads, head_dim]
};

export type GptState = {
  caches: GptKVCache[];
  position: number;
  capacity: number;
};

const ATTENTION_SCALE = 1 / Math.sqrt(GPT_CONFIG.headDim);
const KV_CACHE_BLOCK_SIZE = 512;

const runConv1D = jit(function runConv1D(
  { weight, bias }: Conv1D,
  x: np.Array,
): np.Array {
  // GPT-2 Conv1D: weight is [in, out], so x @ weight (no transpose).
  x = np.dot(x, weight);
  return x.add(bias);
});

const runLayerNorm = jit(
  function runLayerNorm(
    { weight, bias }: LayerNorm,
    x: np.Array,
    eps: number = GPT_CONFIG.layerNormEps,
  ): np.Array {
    const dtype = x.dtype;
    x = x.astype(np.float32);
    const mean = x.ref.mean(-1, { keepdims: true });
    const diff = x.ref.sub(mean);
    const variance = diff.mul(diff).mean(-1, { keepdims: true });
    x = x.sub(mean).div(np.sqrt(variance.add(eps)));
    return x.mul(weight.astype(np.float32)).add(bias.astype(np.float32)).astype(
      dtype,
    );
  },
  { staticArgnums: [2] },
);

const runEmbedding = jit(function runEmbedding(
  { weight }: { weight: np.Array },
  tokenIds: np.Array,
): np.Array {
  return weight.slice(tokenIds).astype(np.float32);
});

function runMLP({ cFc, cProj }: GptMLP, x: np.Array): np.Array {
  const h = nn.gelu(runConv1D(cFc, x.ref), { approximate: true });
  return runConv1D(cProj, h);
}

function runAttentionPrefill(
  { cAttn, cProj }: GptAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  const H = GPT_CONFIG.hiddenSize;
  const numHeads = GPT_CONFIG.numAttentionHeads;
  const headDim = GPT_CONFIG.headDim;

  // Combined QKV projection: [T, 3 * hidden]
  const qkv = runConv1D(cAttn, x.ref);
  const [q, k, v] = np.split(qkv, 3, -1);

  const qHeads = q.reshape([T, numHeads, headDim]);
  const kHeads = k.reshape([T, numHeads, headDim]);
  const vHeads = v.reshape([T, numHeads, headDim]);

  const attn = nn.dotProductAttention(qHeads, kHeads.ref, vHeads.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runConv1D(cProj, attn.reshape([T, H]));
  return { output, key: kHeads, value: vHeads };
}

function runAttentionStep(
  { cAttn, cProj }: GptAttention,
  cache: GptKVCache,
  x: np.Array,
  slot: number,
  validLength: number,
): { output: np.Array; cache: GptKVCache } {
  const T = 1;
  const H = GPT_CONFIG.hiddenSize;
  const numHeads = GPT_CONFIG.numAttentionHeads;
  const headDim = GPT_CONFIG.headDim;

  const qkv = runConv1D(cAttn, x.ref);
  const [q, k, v] = np.split(qkv, 3, -1);

  const qHeads = q.reshape([T, numHeads, headDim]);
  const kHeads = k.reshape([T, numHeads, headDim]);
  const vHeads = v.reshape([T, numHeads, headDim]);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(
    slotMask.ref,
    np.tile(kHeads, [capacity, 1, 1]),
    cache.key,
  );
  const value = np.where(
    slotMask,
    np.tile(vHeads, [capacity, 1, 1]),
    cache.value,
  );

  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(qHeads, key.ref, value.ref, {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runConv1D(cProj, attn.reshape([T, H]));
  return { output, cache: { key, value } };
}

function padCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): GptKVCache {
  const T = key.shape[0];
  if (T > capacity) {
    throw new Error(`Prompt length ${T} exceeds cache capacity ${capacity}`);
  }
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

const runGptBlockPrefill = jit(
  function runGptBlockPrefill(
    { ln1, attn, ln2, mlp }: GptBlock,
    x: np.Array,
    capacity: number,
  ): [np.Array, GptKVCache] {
    // Pre-LN attention
    const residual = x.ref;
    const h = runLayerNorm(ln1, x);
    const { output: attnOut, key, value } = runAttentionPrefill(attn, h);
    x = residual.add(attnOut);

    // Pre-LN MLP
    const residual2 = x.ref;
    const h2 = runLayerNorm(ln2, x);
    const mlpOut = runMLP(mlp, h2);
    return [residual2.add(mlpOut), padCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

const runGptBlockStep = jit(
  function runGptBlockStep(
    { ln1, attn, ln2, mlp }: GptBlock,
    cache: GptKVCache,
    x: np.Array,
    slot: number,
    validLength: number,
  ): [np.Array, GptKVCache] {
    const residual = x.ref;
    const h = runLayerNorm(ln1, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      attn,
      cache,
      h,
      slot,
      validLength,
    );
    x = residual.add(attnOut);

    const residual2 = x.ref;
    const h2 = runLayerNorm(ln2, x);
    const mlpOut = runMLP(mlp, h2);
    return [residual2.add(mlpOut), updatedCache];
  },
  { staticArgnums: [4] },
);

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createGptState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): GptState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: GPT_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros(
        [capacity, GPT_CONFIG.numAttentionHeads, GPT_CONFIG.headDim],
        { dtype },
      ),
      value: np.zeros(
        [capacity, GPT_CONFIG.numAttentionHeads, GPT_CONFIG.headDim],
        { dtype },
      ),
    })),
  };
}

function ensureGptStateCapacity(state: GptState, requiredCapacity: number) {
  if (state.capacity >= requiredCapacity) return;
  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    cache.key = np.pad(cache.key, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(cache.value, { 0: [0, newCapacity - oldCapacity] });
  }
  state.capacity = newCapacity;
}

export function runGptPrefill(
  model: GptModel,
  tokenIds: np.Array,
  state: GptState,
): np.Array {
  const T = tokenIds.shape[0];
  if (T > GPT_CONFIG.maxPositions) {
    throw new Error(
      `Prompt length ${T} exceeds GPT-2 max positions ${GPT_CONFIG.maxPositions}`,
    );
  }
  ensureGptStateCapacity(state, T);

  // Token embeddings + learned positional embeddings.
  const tokenEmb = runEmbedding(model.wte, tokenIds);
  const positions = np.arange(T, undefined, undefined, { dtype: np.uint32 });
  const posEmb = runEmbedding(model.wpe, positions);
  let x = tokenEmb.add(posEmb);

  for (let i = 0; i < GPT_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runGptBlockPrefill(
      model.layers[i],
      x,
      state.capacity,
    );
  }

  x = runLayerNorm(model.lnF, x);
  x = x.slice([-1]);
  // GPT-2 ties embeddings and LM head.
  const logits = np.dot(x, model.wte.weight.transpose()).reshape([
    GPT_CONFIG.vocabSize,
  ]);
  state.position = T;
  return logits;
}

export function runGptStep(
  model: GptModel,
  tokenId: number,
  state: GptState,
): np.Array {
  const position = state.position;
  if (position >= GPT_CONFIG.maxPositions) {
    throw new Error(
      `Position ${position} exceeds GPT-2 max positions ${GPT_CONFIG.maxPositions}`,
    );
  }
  ensureGptStateCapacity(state, position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  const posIds = np.array([position], { dtype: np.uint32 });
  const tokenEmb = runEmbedding(model.wte, tokenIds);
  const posEmb = runEmbedding(model.wpe, posIds);
  let x = tokenEmb.add(posEmb);

  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < GPT_CONFIG.numHiddenLayers; i++) {
    const oldCache = state.caches[i];
    [x, state.caches[i]] = runGptBlockStep(
      model.layers[i],
      oldCache,
      x,
      slot,
      validLength,
    );
  }

  x = runLayerNorm(model.lnF, x);
  // GPT-2 ties embeddings and LM head.
  const logits = np.dot(x, model.wte.weight.transpose()).reshape([
    GPT_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}

const mapper = new WeightMapper({
  prefix: {
    "h.": "layers.",
  },
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
      if (tensor.dtype === "F16") {
        return np.array(tensor.data as Float16Array<ArrayBuffer>, {
          shape: tensor.shape,
          dtype: np.float16,
        });
      }
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
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        { shape: tensor.shape, dtype: np.float32 },
      );
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
    // Skip the causal mask buffer — it's a constant, not a learned weight.
    if (key.endsWith(".attn.bias") || key.endsWith(".attn.masked_bias")) {
      continue;
    }
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as GptModel;
  if (model.layers.length !== GPT_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${GPT_CONFIG.numHiddenLayers} GPT-2 layers, ` +
        `found ${model.layers.length}`,
    );
  }
  return blockUntilReady(model);
}
