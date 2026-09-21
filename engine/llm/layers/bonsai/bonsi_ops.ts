import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { BONSAI_CONFIG } from "../../configs/bonsai_config.ts";
import { rotateHalf } from "../common/ops.ts";

export function applyRoPE(
  q: np.Array, // [T, num_heads, head_dim]
  k: np.Array, // [T, num_key_value_heads, head_dim]
  offset: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(BONSAI_CONFIG.ropeTheta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq);

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}
