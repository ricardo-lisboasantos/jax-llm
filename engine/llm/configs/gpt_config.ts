export const GPT_CONFIG = {
  bosTokenId: 50_256,
  eosTokenId: 50_256,
  vocabSize: 50_257,
  hiddenSize: 768,
  intermediateSize: 3_072,
  numHiddenLayers: 12,
  numAttentionHeads: 12,
  headDim: 64,
  maxPositions: 1_024,
  layerNormEps: 1e-5,
} as const;
