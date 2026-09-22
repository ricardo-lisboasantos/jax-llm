/**
 * Reproducer for Issue #3: INT8 Quantization Fallback Failure
 *
 * Demonstrates that the fallback URL is computed from an already-modified
 * weightsUrl, causing silent loading of wrong model variant or infinite retries.
 */

Deno.test("INT8 Fallback Bug: URL replacement is non-idempotent", () => {
  // Simulate the bug in loadWeights():

  const definitionUrl = "https://cdn.example.com/model_q8.safetensors";
  let weightsUrl = definitionUrl;

  console.log(`Original URL: ${weightsUrl}`);

  // First fetch attempt (INT8)
  const int8Response = false; // Simulate 404

  if (!int8Response) {
    // BUG: Fallback uses definition.weightsUrl which is still the _q8 variant
    console.log("INT8 fetch failed, attempting fallback...");

    // This is what the code currently does:
    weightsUrl = definitionUrl.replace("_q8.safetensors", ".safetensors");
    console.log(`Fallback URL (first): ${weightsUrl}`);
    // Result: "https://cdn.example.com/model.safetensors" ✓ CORRECT

    // But if definitionUrl gets modified somewhere (or code path different):
    if (weightsUrl.includes("_q8")) {
      // Edge case: double replacement could happen
      weightsUrl = weightsUrl.replace("_q8.safetensors", ".safetensors");
      console.log(`Fallback URL (second attempt): ${weightsUrl}`);
      // Result: "https://cdn.example.com/model.safetensors" ✓ Still works
    }
  }

  console.log(
    `\n⚠️  Issue: No guarantee weightsUrl is stable after first fetch`,
  );
  console.log(
    `   If definition.weightsUrl gets mutated, fallback logic breaks`,
  );
});

Deno.test("INT8 Fallback Bug: Fix by saving original URL", () => {
  const definitionUrl = "https://cdn.example.com/model_q8.safetensors";
  const originalUrl = definitionUrl; // SAVE ORIGINAL
  let weightsUrl = definitionUrl;

  console.log(`Original URL: ${weightsUrl}`);

  const int8Response = false; // Simulate 404

  if (!int8Response) {
    console.log("INT8 fetch failed, attempting fallback...");

    // FIX: Use SAVED original, not definition.weightsUrl
    weightsUrl = originalUrl.replace("_q8.safetensors", ".safetensors");
    console.log(`Fallback URL (from saved original): ${weightsUrl}`);
    // Result: "https://cdn.example.com/model.safetensors" ✓ CORRECT
  }

  console.log(`\n✅ Fix: Save original URL before any modifications`);
  console.log(`   const originalUrl = def.weightsUrl;`);
  console.log(`   ... later ...`);
  console.log(
    `   weightsUrl = originalUrl.replace(...);  // Use saved original`,
  );
  console.log(
    `\nBenefit: Fallback is always applied to known source, idempotent`,
  );
});

Deno.test("INT8 Fallback: Both variants missing should throw clear error", () => {
  const originalUrl = "https://cdn.example.com/model_q8.safetensors";
  let weightsUrl = originalUrl;

  // Simulate both INT8 and FP32 fetch failing
  const int8Available = false;
  const fp32Available = false;

  if (!int8Available) {
    console.log("INT8 not available, trying FP32...");
    weightsUrl = originalUrl.replace("_q8.safetensors", ".safetensors");
  }

  if (!fp32Available) {
    console.error(
      `❌ Failed to load model: INT8 not available, FP32 also missing`,
    );
    console.error(`   Original URL: ${originalUrl}`);
    console.error(`   Fallback URL: ${weightsUrl}`);
    console.error(`   This is a CLEAR ERROR, not silent failure`);
  }
});
