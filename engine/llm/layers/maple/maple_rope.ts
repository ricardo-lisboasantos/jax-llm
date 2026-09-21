import { numpy as np } from "npm:@jax-js/jax@^0.1.25";
import { MAPLE_CONFIG } from "../../configs/maple_config.ts";
import { rotateHalf } from "../common/ops.ts";

const ROTARY_DIM = Math.floor(
  MAPLE_CONFIG.headDim * MAPLE_CONFIG.partialRotaryFactor,
); // 64

/**
 * Apply partial rotary position embeddings to the first `ROTARY_DIM`
 * dimensions of each head.  The remaining dimensions pass through.
 */
export function applyPartialRoPE(
  q: np.Array, // [T, num_heads, head_dim]
  k: np.Array, // [T, num_kv_heads, head_dim]
  offset: number,
): [np.Array, np.Array] {
  const [T] = q.shape;
  const halfRotary = ROTARY_DIM / 2;

  const dim = np.arange(halfRotary, undefined, undefined, {
    dtype: np.float32,
  });
  const invFreq = np.exp(
    dim.mul((-Math.log(MAPLE_CONFIG.ropeTheta) * 2) / ROTARY_DIM),
  );
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq);

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([
    T,
    1,
    ROTARY_DIM,
  ]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([
    T,
    1,
    ROTARY_DIM,
  ]);

  const [qRot, qPass] = np.split(q, [ROTARY_DIM], -1);
  const [kRot, kPass] = np.split(k, [ROTARY_DIM], -1);

  const qRotOut = qRot.ref.mul(cos.ref).add(rotateHalf(qRot).mul(sin.ref));
  const kRotOut = kRot.ref.mul(cos).add(rotateHalf(kRot).mul(sin));

  const qOut = np.concatenate([qRotOut, qPass], -1);
  const kOut = np.concatenate([kRotOut, kPass], -1);
  return [qOut, kOut];
}
