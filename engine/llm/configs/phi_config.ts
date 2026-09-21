export const PHI_CONFIG = {
  bosTokenId: 50_256,
  eosTokenId: 50_256,
  vocabSize: 51_200,
  hiddenSize: 2_560,
  intermediateSize: 10_240,
  numHiddenLayers: 32,
  numAttentionHeads: 32,
  numKeyValueHeads: 32,
  headDim: 80, // hiddenSize / numAttentionHeads
  layerNormEps: 1e-5,
  ropeTheta: 10_000,
  partialRotaryFactor: 0.4,
} as const;
