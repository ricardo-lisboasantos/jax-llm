import type { LfmLinear } from "./lfm_linear.ts";

export type LfmShortConv = {
  conv: LfmLinear;
  inProj: LfmLinear;
  outProj: LfmLinear;
};
