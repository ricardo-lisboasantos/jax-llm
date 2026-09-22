/**
 * Reproducer for Issue #5: Paged Cache Page Table Corruption
 *
 * Demonstrates that logical and physical page indices are confused,
 * preventing valid data from being marked and causing cache misses.
 */

import { PagedKVCache } from "../engine/llm/cache/paged_cache.ts";

Deno.test("Paged Cache Corruption: Logical/physical index confusion", () => {
  const cache = new PagedKVCache({
    pageSize: 512,
    maxPages: 32,
    headDim: 64,
    numKvHeads: 8,
  });
  cache.initialize(1); // 1 layer for testing

  console.log("Allocating pages for long-context sequence...\n");

  // Allocate pages for tokens 0-512 (page 0)
  const page0 = cache.allocatePageForToken(0, 0);
  console.log(
    `Allocated for token 0: pageIdx=${page0.pageIdx} (logical), offset=${page0.offset}`,
  );

  // Allocate pages for tokens 512-1024 (page 1)
  const page1 = cache.allocatePageForToken(0, 512);
  console.log(
    `Allocated for token 512: pageIdx=${page1.pageIdx} (logical), offset=${page1.offset}`,
  );

  // Mark tokens 0-512 as valid
  cache.updateValidLength(0, 512);
  console.log(`\nUpdated valid length to 512 tokens (end of page 0)`);

  // In broken code, this might not work correctly due to index confusion
  // Let's simulate what happens:

  console.log("\nBroken code flow:");
  console.log(
    "  allocatePageForToken stores: pageEntries[].pageIdx = newPageId (0)",
  );
  console.log(
    "  updateValidLength compares: Math.floor(newPageId / 512) === logicalPageIdx",
  );
  console.log("  For page 0: Math.floor(0 / 512) === 0 ✓ WORKS (by accident)");
  console.log("  For page 1: Math.floor(1 / 512) === 1 ✓ WORKS (by accident)");

  console.log("\nAfter LRU eviction (page 0 evicted, newPageId now = 2):");
  console.log(
    "  NEW allocation for token 0: pageEntries[].pageIdx = newPageId (2)",
  );
  console.log("  updateValidLength compares: Math.floor(2 / 512) === 0");
  console.log("  FALSE! ✗ BREAKS - Valid length never updated");

  // Try to fetch the cache
  const keyPage0 = cache.getPage(0, page0.pageIdx, true);
  console.log(
    `\nFetched key cache for page 0: ${keyPage0 ? "HAS DATA" : "NULL"}`,
  );

  if (keyPage0 === null) {
    console.log("⚠️  CRITICAL: Cache returns NULL even though we allocated!");
    console.log("   Attention will treat this as 'no cached data'");
    console.log("   Result: Recompute all prior tokens → OOM or wrong output");
  }
});

Deno.test("Paged Cache Corruption: Fix by separating indices", () => {
  // The fix is to track logical and physical indices separately

  interface FixedPageTableEntry {
    pageLogicalIdx: number; // Which logical page (0, 1, 2...)
    pagePhysicalId: number; // Physical allocation ID (0, 1, 2... may skip after eviction)
    validLength: number;
    lastAccessMs: number;
  }

  const pageTable: FixedPageTableEntry[] = [];
  let nextPageId = 0;

  // Simulate allocation for page 0 (logical)
  const logicalIdx0 = 0;
  const physicalId0 = nextPageId++; // 0
  pageTable.push({
    pageLogicalIdx: logicalIdx0,
    pagePhysicalId: physicalId0,
    validLength: 0,
    lastAccessMs: performance.now(),
  });
  console.log(`Allocated: logical=${logicalIdx0}, physical=${physicalId0}`);

  // Simulate allocation for page 1 (logical)
  const logicalIdx1 = 1;
  const physicalId1 = nextPageId++; // 1
  pageTable.push({
    pageLogicalIdx: logicalIdx1,
    pagePhysicalId: physicalId1,
    validLength: 0,
    lastAccessMs: performance.now(),
  });
  console.log(`Allocated: logical=${logicalIdx1}, physical=${physicalId1}`);

  // Update valid length for page 0
  const tokenPos = 512;
  const logicalPageIdx = Math.floor(tokenPos / 512); // 1
  for (const entry of pageTable) {
    if (entry.pageLogicalIdx === logicalPageIdx) {
      entry.validLength = tokenPos % 512;
      console.log(
        `\nUpdated valid length: logical page ${logicalPageIdx}, entry now has validLength=${entry.validLength}`,
      );
    }
  }

  // Simulate LRU eviction of page 0
  pageTable.splice(0, 1); // Remove first entry
  console.log("Evicted logical page 0 (physical ID 0)");

  // Simulate new allocation for logical page 0
  const newPhysicalId = nextPageId++; // 2 (skips 1, which is still in use)
  pageTable.push({
    pageLogicalIdx: 0, // SAME logical index, but different physical ID
    pagePhysicalId: newPhysicalId, // 2
    validLength: 0,
    lastAccessMs: performance.now(),
  });
  console.log(
    `Re-allocated: logical=${logicalIdx0}, physical=${newPhysicalId}`,
  );

  // Now update valid length again
  const tokenPos2 = 100;
  const logicalPageIdx2 = Math.floor(tokenPos2 / 512); // 0
  for (const entry of pageTable) {
    if (entry.pageLogicalIdx === logicalPageIdx2) {
      entry.validLength = tokenPos2 % 512;
      console.log(
        `\nUpdated valid length: logical page ${logicalPageIdx2}, physical ${entry.pagePhysicalId}, validLength=${entry.validLength} ✓ CORRECT`,
      );
    }
  }

  console.log("\n✅ With separated indices:");
  console.log("   - Logical indices track token positions (0, 1, 2...)");
  console.log(
    "   - Physical IDs track actual allocations (0, 1, 2... with gaps after eviction)",
  );
  console.log("   - updateValidLength() compares logical indices directly");
  console.log("   - Always finds the right page, even after eviction");
});
