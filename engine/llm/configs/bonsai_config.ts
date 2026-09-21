export const BONSAI_CONFIG = {
  bosTokenId: 1,
  eosTokenId: 2,
  padTokenId: 0,
  vocabSize: 32_000,
  hiddenSize: 1_536,
  intermediateSize: 4_096,
  numHiddenLayers: 16,
  numAttentionHeads: 16,
  numKeyValueHeads: 8,
  headDim: 96,
  rmsNormEps: 1e-5,
  ropeTheta: 100_000,
} as const;
