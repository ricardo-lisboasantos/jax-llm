import type { LfmMLP } from "./lfm_mlp.ts";
import type { RMSNorm } from "./lfm_rms_norm.ts";

export type LfmLayerBase = {
  operatorNorm: RMSNorm;
  ffnNorm: RMSNorm;
  feedForward: LfmMLP;
};
