export const QWEN_CONFIG = {
  bosTokenId: 151_643,
  eosTokenId: 151_643,
  padTokenId: 151_643,
  vocabSize: 151_936,
  hiddenSize: 896,
  intermediateSize: 4_864,
  numHiddenLayers: 24,
  numAttentionHeads: 14,
  numKeyValueHeads: 2,
  headDim: 64, // hiddenSize / numAttentionHeads
  rmsNormEps: 1e-6,
  ropeTheta: 1_000_000,
} as const;
