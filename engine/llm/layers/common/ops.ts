import { numpy as np } from "npm:@jax-js/jax@^0.1.25";

export function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.mul(-1), x1], -1);
}
