/**
 * unit: maple checkpoint key classification (B2).
 *
 * Real key samples from `deepgrove/maple-preview`
 * (`model.safetensors.index.json`, 18,651 tensors, 9 shards). Guards the
 * `.weight`-suffix form that the first version of the classifier missed
 * (would have hydrated all 18,432 expert tensors as dense → OOM).
 */
import { assertEquals } from "@std/assert";
import { classifyMapleKey } from "./maple.ts";

Deno.test("unit: classifyMapleKey matches real expert keys", () => {
  const samples = [
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.23.mlp.experts.255.up_proj.weight",
  ];
  const want = [
    { layer: 0, expert: 0, proj: "downProj" },
    { layer: 0, expert: 0, proj: "gateProj" },
    { layer: 0, expert: 0, proj: "upProj" },
    { layer: 23, expert: 255, proj: "upProj" },
  ] as const;
  for (let i = 0; i < samples.length; i++) {
    assertEquals(classifyMapleKey(samples[i]), want[i]);
  }
});

Deno.test("unit: classifyMapleKey rejects dense keys", () => {
  const dense = [
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.gate.weight",
    "model.word_embeddings.weight",
    "model.norm.weight",
    "lm_head.weight",
  ];
  for (const key of dense) {
    assertEquals(classifyMapleKey(key), null, key);
  }
});
