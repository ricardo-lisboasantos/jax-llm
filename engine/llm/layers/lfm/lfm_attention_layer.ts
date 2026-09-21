import type { LfmAttention } from "./lfm_attention.ts";
import type { LfmLayerBase } from "./lfm_layer_base.ts";

export type LfmAttentionLayer = LfmLayerBase & {
  selfAttn: LfmAttention;
};
