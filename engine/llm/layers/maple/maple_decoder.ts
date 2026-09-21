import type { RMSNorm } from "./maple_rms_norm.ts";
import type { MapleAttention } from "./maple_attention.ts";
import type { MapleMoE } from "./maple_moe.ts";

export type MapleDecoderLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  selfAttn: MapleAttention;
  mlp: MapleMoE;
};
