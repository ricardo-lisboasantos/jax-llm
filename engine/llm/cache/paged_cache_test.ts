/**
 * unit: Paged KV cache tests
 */
import { assertEquals, assert } from "@std/assert";
import { PagedKVCache, PAGE_SIZE } from "./paged_cache.ts";

Deno.test("unit: PagedKVCache initializes with correct structure", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 10 });
  cache.initialize(4); // 4 layers

  const stats = cache.getStats();
  assertEquals(stats.numLayers, 4);
  assertEquals(stats.usedPages, 0);
  assertEquals(stats.freePages, 10);
});

Deno.test("unit: PagedKVCache allocates pages on demand", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 16 });
  cache.initialize(2);

  // Allocate page for first token
  const { pageIdx: page0, offset: offset0 } = cache.allocatePageForToken(0, 0);
  assertEquals(page0, 0);
  assertEquals(offset0, 0);

  const stats = cache.getStats();
  assertEquals(stats.usedPages, 1);

  // Allocate page for token within same page
  const { pageIdx: page1, offset: offset1 } = cache.allocatePageForToken(0, 256);
  assertEquals(page1, 0);
  assertEquals(offset1, 256);

  // Still only 1 page
  const stats2 = cache.getStats();
  assertEquals(stats2.usedPages, 1);

  // Allocate page for token beyond first page
  const { pageIdx: page2, offset: offset2 } = cache.allocatePageForToken(0, 512);
  assertEquals(page2, 1);
  assertEquals(offset2, 0);

  const stats3 = cache.getStats();
  assertEquals(stats3.usedPages, 2);
});

Deno.test("unit: PagedKVCache tracks page allocation per layer", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 20 });
  cache.initialize(3);

  // Allocate pages for different layers
  cache.allocatePageForToken(0, 0);
  cache.allocatePageForToken(0, 100);
  cache.allocatePageForToken(1, 0);
  cache.allocatePageForToken(2, 50);

  const stats = cache.getStats();
  assertEquals(stats.usedPages, 3); // 3 distinct pages
});

Deno.test("unit: PagedKVCache estimates sequence memory", () => {
  const cache = new PagedKVCache({
    pageSize: 512,
    maxPages: 20,
    headDim: 64,
    numKvHeads: 8,
  });
  cache.initialize(12); // LLaMA-like

  // For 1024 tokens, should need at least 2 pages
  const mem1k = cache.estimateMemoryForSequence(1024);
  assert(mem1k > 0);

  // 8K tokens should need ~16 pages
  const mem8k = cache.estimateMemoryForSequence(8192);
  assert(mem8k > mem1k);

  // Memory should scale roughly linearly
  assertEquals(mem8k > mem1k * 7, true, "8K should use more memory than 1K");
});

Deno.test("unit: PagedKVCache clears all data", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 10 });
  cache.initialize(2);

  // Allocate some pages
  cache.allocatePageForToken(0, 0);
  cache.allocatePageForToken(0, 600);
  cache.allocatePageForToken(1, 100);

  let stats = cache.getStats();
  assertEquals(stats.usedPages > 0, true);

  // Clear cache
  cache.clear();

  stats = cache.getStats();
  assertEquals(stats.usedPages, 0);
  assertEquals(stats.freePages, 10);
});

Deno.test("unit: PagedKVCache respects maxPages limit", () => {
  const cache = new PagedKVCache({ pageSize: 100, maxPages: 3 });
  cache.initialize(1);

  // Allocate 3 pages
  cache.allocatePageForToken(0, 0);
  cache.allocatePageForToken(0, 100);
  cache.allocatePageForToken(0, 200);

  let stats = cache.getStats();
  assertEquals(stats.usedPages, 3);
  assertEquals(stats.freePages, 0);

  // Allocate 4th page should trigger LRU eviction
  cache.allocatePageForToken(0, 300);

  stats = cache.getStats();
  assertEquals(stats.usedPages, 3); // Still at max
  assertEquals(stats.freePages, 0);
});

Deno.test("unit: PagedKVCache updates valid length", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 10 });
  cache.initialize(1);

  cache.allocatePageForToken(0, 0);
  cache.updateValidLength(0, 256);

  const stats = cache.getStats();
  assertEquals(stats.usedPages >= 1, true);
});

Deno.test("unit: PagedKVCache getPage returns allocated pages", () => {
  const cache = new PagedKVCache({ pageSize: 512, maxPages: 10, headDim: 64 });
  cache.initialize(1);

  // Allocate a page
  const { pageIdx } = cache.allocatePageForToken(0, 0);

  // Get the page (note: pageIdx in this simplified implementation)
  const page = cache.getPage(0, pageIdx, true);
  assert(page !== null, "Allocated page should be retrievable");
});

Deno.test("unit: PagedKVCache supports multi-layer long prompts", () => {
  const cache = new PagedKVCache({
    pageSize: 512,
    maxPages: 64,
    headDim: 64,
    numKvHeads: 8,
  });
  cache.initialize(12); // 12 layers

  // Simulate 8K-token prompt
  const promptLength = 8192;
  for (let pos = 0; pos < promptLength; pos += 256) {
    for (let layer = 0; layer < 12; layer++) {
      cache.allocatePageForToken(layer, pos);
    }
  }

  const stats = cache.getStats();
  assert(stats.usedPages > 0);
  assert(stats.usedPages <= stats.totalPages);
});
