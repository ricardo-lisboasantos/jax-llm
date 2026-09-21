// Deno shim for the File System Access API (OPFS)
// Maps browser-based storage requests to the local Deno filesystem.

const OPFS_ROOT = "./.opfs";

// Ensure the root directory exists (only when write access is available,
// so importing this module under restricted permissions doesn't throw).
try {
  const { state } = await Deno.permissions.query({
    name: "write",
    path: OPFS_ROOT,
  });
  if (state === "granted") {
    await Deno.mkdir(OPFS_ROOT, { recursive: true });
  }
} catch (e) {
  if (!(e instanceof Deno.errors.AlreadyExists)) throw e;
}

// Interfaces to match Browser API
interface FileSystemHandle {
  readonly kind: "file" | "directory";
  readonly name: string;
}

class FileSystemFileHandle implements FileSystemHandle {
  readonly kind = "file";
  constructor(readonly name: string, private path: string) {}

  async getFile(): Promise<File> {
    try {
      const data = await Deno.readFile(this.path);
      return new File([data], this.name);
    } catch (e) {
      if (e instanceof Deno.errors.NotFound) {
        throw new DOMException(`File not found: ${this.name}`, "NotFoundError");
      }
      throw e;
    }
  }

  async createWritable(): Promise<FileSystemWritableFileStream> {
    const file = await Deno.open(this.path, {
      write: true,
      create: true,
      truncate: true,
    });
    return {
      write: async (data: string | Uint8Array) => {
        const buffer = typeof data === "string"
          ? new TextEncoder().encode(data)
          : data;
        await file.write(buffer);
      },
      close: () => file.close(),
    } as FileSystemWritableFileStream;
  }
}

// Minimal interface for Writable Stream
interface FileSystemWritableFileStream {
  write(data: string | Uint8Array): Promise<void>;
  close(): Promise<void>;
}

class FileSystemDirectoryHandle implements FileSystemHandle {
  readonly kind = "directory";
  constructor(readonly name: string, private path: string) {}

  async getFileHandle(
    name: string,
    { create = false } = {},
  ): Promise<FileSystemFileHandle> {
    const filePath = `${this.path}/${name}`;
    if (create) {
      await Deno.writeFile(filePath, new Uint8Array(), { create: true });
    }
    return new FileSystemFileHandle(name, filePath);
  }

  async getDirectoryHandle(
    name: string,
    { create = false } = {},
  ): Promise<FileSystemDirectoryHandle> {
    const dirPath = `${this.path}/${name}`;
    if (create) {
      await Deno.mkdir(dirPath, { recursive: true });
    }
    return new FileSystemDirectoryHandle(name, dirPath);
  }

  async *entries() {
    for await (const entry of Deno.readDir(this.path)) {
      if (entry.isFile) {
        yield [
          entry.name,
          new FileSystemFileHandle(entry.name, `${this.path}/${entry.name}`),
        ];
      } else if (entry.isDirectory) {
        yield [
          entry.name,
          new FileSystemDirectoryHandle(
            entry.name,
            `${this.path}/${entry.name}`,
          ),
        ];
      }
    }
  }
}

// Patch Global Scope
if (!globalThis.navigator) {
  // @ts-ignore: Deno doesn't have navigator
  globalThis.navigator = {};
}

// @ts-ignore: Deno type definition for Navigator doesn't have storage
if (!(globalThis.navigator as Navigator & { storage?: unknown }).storage) {
  // @ts-ignore: Patching
  globalThis.navigator.storage = {
    getDirectory: () => new FileSystemDirectoryHandle("root", OPFS_ROOT),
  };
}

// Ensure File is available (in browser this is standard)
if (!globalThis.File) {
  // @ts-ignore: Deno has File
  globalThis.File = File;
}
