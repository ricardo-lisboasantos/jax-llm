/**
 * @module runtime/offload/nvme
 *
 * NVMe (disk) weight offloading: spills inactive tensors to local disk
 * under `.opfs/` (Deno) so large models can exceed available RAM.
 *
 * The implementation is intentionally minimal and async to avoid
 * blocking the inference loop.
 */

/** Configuration for {@linkcode NvmeOffload}. */
export type NvmeOffloadConfig = {
  /** Directory used for spilled tensors (created on demand). */
  dir?: string;
};

function joinPath(dir: string, key: string): string {
  const safe = key.replace(/[^a-zA-Z0-9._-]/g, "_");
  return `${dir}/${safe}.bin`;
}

/** Disk-backed offload store. */
export class NvmeOffload {
  /** Spill directory (created on demand). */
  readonly dir: string;

  /** Create a store rooted at `dir` (default `./.opfs/offload`). */
  constructor(config: NvmeOffloadConfig = {}) {
    this.dir = config.dir ?? "./.opfs/offload";
  }

  /** Spill raw bytes to disk (overwrites). */
  async put(key: string, data: Uint8Array): Promise<void> {
    await Deno.mkdir(this.dir, { recursive: true });
    await Deno.writeFile(joinPath(this.dir, key), data);
  }

  /** Read spilled bytes, or `undefined` when absent. */
  async get(key: string): Promise<Uint8Array | undefined> {
    try {
      return await Deno.readFile(joinPath(this.dir, key));
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return undefined;
      throw e;
    }
  }

  /** Remove one spill file; returns whether it existed. */
  async evict(key: string): Promise<boolean> {
    try {
      await Deno.remove(joinPath(this.dir, key));
      return true;
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) return false;
      throw e;
    }
  }
}
