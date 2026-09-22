/**
 * @module llm/profiling/webgpu_profiler
 *
 * WebGPU kernel profiling for identifying per-kernel latency bottlenecks.
 * Measures execution time distribution (p50/p90/p99) to detect variance sources.
 *
 * Usage:
 * ```typescript
 * const profiler = new WebGPUProfiler();
 * profiler.start("attention_forward");
 * // ... kernel execution ...
 * profiler.end("attention_forward");
 *
 * const stats = profiler.getStats("attention_forward");
 * console.log(`Attention p50=${stats.p50}ms, p99=${stats.p99}ms`);
 * ```
 */

export interface ProfilerStats {
  /** Number of samples collected. */
  count: number;
  /** Total time in milliseconds. */
  totalMs: number;
  /** Average latency in milliseconds. */
  meanMs: number;
  /** Minimum latency observed. */
  minMs: number;
  /** Maximum latency observed. */
  maxMs: number;
  /** 50th percentile (median) in milliseconds. */
  p50Ms: number;
  /** 90th percentile in milliseconds. */
  p90Ms: number;
  /** 99th percentile in milliseconds. */
  p99Ms: number;
  /** Standard deviation in milliseconds. */
  stddevMs: number;
}

/**
 * WebGPU kernel profiler for measuring per-operation latency.
 * Collects timing samples and computes distribution statistics.
 */
export class WebGPUProfiler {
  private samples = new Map<string, number[]>();
  private timestamps = new Map<string, number[]>();

  /**
   * Mark the start of a timed operation.
   * @param label Operation label (e.g., "attention_forward", "embedding")
   */
  start(label: string): void {
    if (!this.timestamps.has(label)) {
      this.timestamps.set(label, []);
    }
    this.timestamps.get(label)!.push(performance.now());
  }

  /**
   * Mark the end of a timed operation and record latency.
   * @param label Operation label (must match a prior `start()` call)
   */
  end(label: string): void {
    const stack = this.timestamps.get(label);
    if (!stack || stack.length === 0) {
      console.warn(`No start timestamp for label: ${label}`);
      return;
    }
    const startTime = stack.pop()!;
    const elapsedMs = performance.now() - startTime;
    if (!this.samples.has(label)) {
      this.samples.set(label, []);
    }
    this.samples.get(label)!.push(elapsedMs);
  }

  /**
   * Get distribution statistics for a labeled operation.
   * @param label Operation label
   * @returns Percentile and statistical summary, or undefined if no samples
   */
  getStats(label: string): ProfilerStats | undefined {
    const samples = this.samples.get(label);
    if (!samples || samples.length === 0) return undefined;

    const sorted = [...samples].sort((a, b) => a - b);
    const count = sorted.length;
    const totalMs = sorted.reduce((a, b) => a + b, 0);
    const meanMs = totalMs / count;
    const minMs = sorted[0];
    const maxMs = sorted[count - 1];

    // Compute percentiles
    const getPercentile = (p: number) => {
      const idx = Math.ceil((p / 100) * count) - 1;
      return sorted[Math.max(0, Math.min(idx, count - 1))];
    };

    // Compute standard deviation
    const variance =
      sorted.reduce((sum, x) => sum + Math.pow(x - meanMs, 2), 0) / count;
    const stddevMs = Math.sqrt(variance);

    return {
      count,
      totalMs,
      meanMs,
      minMs,
      maxMs,
      p50Ms: getPercentile(50),
      p90Ms: getPercentile(90),
      p99Ms: getPercentile(99),
      stddevMs,
    };
  }

  /**
   * Get statistics for all recorded labels.
   * @returns Map of label → stats for all operations
   */
  getAllStats(): Map<string, ProfilerStats> {
    const result = new Map<string, ProfilerStats>();
    for (const label of this.samples.keys()) {
      const stats = this.getStats(label);
      if (stats) result.set(label, stats);
    }
    return result;
  }

  /**
   * Reset all collected samples.
   */
  reset(): void {
    this.samples.clear();
    this.timestamps.clear();
  }

  /**
   * Format statistics as a human-readable table.
   * @param stats Map of operation → statistics
   * @returns Formatted string
   */
  static formatStats(stats: Map<string, ProfilerStats>): string {
    const rows: string[] = [];
    rows.push(
      "Operation\t\tCount\tMean(ms)\tP50(ms)\tP90(ms)\tP99(ms)\tStdDev(ms)",
    );
    rows.push("─".repeat(100));
    for (const [label, stat] of stats) {
      rows.push(
        `${label.padEnd(20)}\t${stat.count}\t${stat.meanMs.toFixed(2)}\t${
          stat.p50Ms.toFixed(2)
        }\t${stat.p90Ms.toFixed(2)}\t${stat.p99Ms.toFixed(2)}\t${
          stat.stddevMs.toFixed(2)
        }`,
      );
    }
    return rows.join("\n");
  }

  /**
   * Export metrics as JSON for external monitoring systems (Phase 3.0).
   * Suitable for APM dashboards and regression tracking.
   *
   * @param stats Map of operation → statistics
   * @returns JSON object with latency distribution and timestamp
   *
   * @example
   * ```typescript
   * const metrics = WebGPUProfiler.metricsJSON(globalProfiler.getAllStats());
   * console.log(JSON.stringify(metrics)); // Send to DataDog, Prometheus, etc.
   * ```
   */
  static metricsJSON(
    stats: Map<string, ProfilerStats>,
  ): Record<string, unknown> {
    const operations: Record<string, unknown> = {};

    for (const [label, stat] of stats) {
      operations[label] = {
        count: stat.count,
        meanMs: Math.round(stat.meanMs * 100) / 100,
        p50Ms: Math.round(stat.p50Ms * 100) / 100,
        p90Ms: Math.round(stat.p90Ms * 100) / 100,
        p99Ms: Math.round(stat.p99Ms * 100) / 100,
        stddevMs: Math.round(stat.stddevMs * 100) / 100,
      };
    }

    return {
      timestamp: new Date().toISOString(),
      operations,
    };
  }

  /**
   * Detect performance regression by comparing p90 latency against baseline (Phase 3.0).
   * Used in CI/CD to catch performance regressions automatically.
   *
   * @param stats Current metrics
   * @param baseline Expected p90 latency (ms)
   * @param threshold Regression threshold in ms (default: 5ms)
   * @returns True if regression detected (p90 > baseline + threshold)
   *
   * @example
   * ```typescript
   * const regression = WebGPUProfiler.detectRegression(
   *   globalProfiler.getAllStats(),
   *   50,  // baseline: 50ms
   *   5,   // threshold: 5ms
   * );
   * if (regression) {
   *   process.exit(1); // Fail CI on regression
   * }
   * ```
   */
  static detectRegression(
    stats: Map<string, ProfilerStats>,
    baseline: number,
    threshold: number = 5,
  ): boolean {
    const decodeStat = stats.get("decode_step");
    if (!decodeStat) return false;
    const regression = decodeStat.p90Ms - baseline;
    if (regression > threshold) {
      console.warn(
        `⚠️ Performance regression detected: p90 ${decodeStat.p90Ms}ms (baseline ${baseline}ms, delta +${regression}ms)`,
      );
      return true;
    }
    return false;
  }
}

/**
 * Global singleton profiler instance (shared across all operations).
 */
export const globalProfiler = new WebGPUProfiler();

/**
 * Decorator function to auto-profile an async function.
 * @param label Operation label for profiling
 * @param fn Async function to profile
 * @returns Wrapped function that profiles execution
 *
 * @example
 * ```typescript
 * const profiledAttention = profileAsync("attention", originalAttentionFn);
 * await profiledAttention(...args);
 * console.log(globalProfiler.getStats("attention"));
 * ```
 */
export function profileAsync<T extends unknown[], R>(
  label: string,
  fn: (...args: T) => Promise<R>,
): (...args: T) => Promise<R> {
  return async (...args: T): Promise<R> => {
    globalProfiler.start(label);
    const result = await fn(...args);
    globalProfiler.end(label);
    return result;
  };
}

/**
 * Decorator for sync functions.
 */
export function profileSync<T extends unknown[], R>(
  label: string,
  fn: (...args: T) => R,
): (...args: T) => R {
  return (...args: T): R => {
    globalProfiler.start(label);
    const result = fn(...args);
    globalProfiler.end(label);
    return result;
  };
}
