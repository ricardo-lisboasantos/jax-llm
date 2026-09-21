import { GEMMA_CONFIG } from "../../configs/gemma_config.ts";

export function layerRopeTheta(layerIndex: number): number {
  return GEMMA_CONFIG.layerTypes[layerIndex] === "full_attention"
    ? GEMMA_CONFIG.ropeTheta
    : GEMMA_CONFIG.ropeLocalBaseFreq;
}
