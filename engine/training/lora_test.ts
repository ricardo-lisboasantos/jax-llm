/**
 * unit: LoRA / QLoRA configs + offload stores.
 */
import { assert, assertEquals, assertThrows } from "@std/assert";
import { DEFAULT_LORA_CONFIG, loraParamCount, loraScale } from "./lora.ts";
import { DEFAULT_QLORA_CONFIG, validateQloraConfig } from "./qlora.ts";
import { RamOffload } from "../runtime/offload/ram.ts";
import { NvmeOffload } from "../runtime/offload/nvme.ts";

Deno.test("unit: lora scale + param count", () => {
  assertEquals(loraScale(DEFAULT_LORA_CONFIG), 2); // 16 / 8
  assertEquals(loraParamCount(768, 768, DEFAULT_LORA_CONFIG), 8 * 1536);
  assertThrows(() => loraScale({ ...DEFAULT_LORA_CONFIG, rank: 0 }), Error);
});

Deno.test("unit: qlora defaults + validation", () => {
  assertEquals(DEFAULT_QLORA_CONFIG.bits, 4);
  validateQloraConfig(DEFAULT_QLORA_CONFIG);
  assertThrows(
    () =>
      validateQloraConfig({
        ...DEFAULT_QLORA_CONFIG,
        bits: 2 as never,
      }),
    Error,
  );
});

Deno.test("unit: RamOffload put/get/evict", () => {
  const ram = new RamOffload();
  ram.put("w", new Uint8Array([1, 2, 3]));
  assertEquals(ram.get("w"), new Uint8Array([1, 2, 3]));
  assertEquals(ram.size, 1);
  assert(ram.evict("w"));
  assertEquals(ram.get("w"), undefined);
});

Deno.test("unit: NvmeOffload put/get/evict (tmp dir)", async () => {
  const dir = await Deno.makeTempDir();
  const nvme = new NvmeOffload({ dir });
  await nvme.put("t1", new Uint8Array([9, 9]));
  assertEquals(await nvme.get("t1"), new Uint8Array([9, 9]));
  assert(await nvme.evict("t1"));
  assertEquals(await nvme.get("t1"), undefined);
  await Deno.remove(dir, { recursive: true });
});
