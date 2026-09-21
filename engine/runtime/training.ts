/**
 * @module runtime/training
 *
 * Training utilities built on optax for fine-tuning loaded models.
 *
 * Provides:
 *  - `createOptimizer()` — builds optax gradient transforms (sgd, adamw, ...)
 *  - `crossEntropyLoss()` — standard next-token prediction loss
 *  - `TrainingRunner` — orchestrates the train loop over a model + optimizer
 *
 * This module operates at the runtime level: it works with raw JAX arrays
 * and model weight trees, not chat abstractions.
 */

import { grad, numpy as np, tree } from "npm:@jax-js/jax@^0.1.25";
import * as optax from "npm:@jax-js/optax@^0.1.2";

import type { ModelDefinition } from "./types.ts";

// ── Optimizer factory ─────────────────────────────────────────────────────

export type OptimizerType = "sgd" | "adam" | "adamw";

export type OptimizerOptions = {
  learningRate: number;
  weightDecay?: number;
  /** Max global norm for gradient clipping (0 = no clipping). */
  maxGradNorm?: number;
  /** Adam beta1. */
  beta1?: number;
  /** Adam beta2. */
  beta2?: number;
  /** Adam epsilon. */
  eps?: number;
};

/**
 * Create an optax optimizer (gradient transform + init).
 * Returns the transform and a function to initialise optimizer state.
 */
export function createOptimizer(
  type: OptimizerType,
  opts: OptimizerOptions,
): optax.GradientTransformation {
  const lr = optax.scaleByLearningRate(opts.learningRate);

  let core: optax.GradientTransformation;
  switch (type) {
    case "sgd":
      core = optax.chain(
        opts.maxGradNorm
          ? optax.clipByGlobalNorm(opts.maxGradNorm)
          : optax.identity(),
        lr,
      );
      break;
    case "adam":
      core = optax.chain(
        opts.maxGradNorm
          ? optax.clipByGlobalNorm(opts.maxGradNorm)
          : optax.identity(),
        optax.scaleByAdam({
          b1: opts.beta1 ?? 0.9,
          b2: opts.beta2 ?? 0.999,
          eps: opts.eps ?? 1e-8,
        }),
        lr,
      );
      break;
    case "adamw":
      core = optax.chain(
        opts.maxGradNorm
          ? optax.clipByGlobalNorm(opts.maxGradNorm)
          : optax.identity(),
        optax.scaleByAdam({
          b1: opts.beta1 ?? 0.9,
          b2: opts.beta2 ?? 0.999,
          eps: opts.eps ?? 1e-8,
        }),
        optax.addDecayedWeights({ weightDecay: opts.weightDecay ?? 0.01 }),
        lr,
      );
      break;
  }
  return core;
}

// ── Loss ───────────────────────────────────────────────────────────────────

/**
 * Standard causal language-modeling cross-entropy loss.
 *
 * @param logits  [T, vocabSize] — model output logits.
 * @param targets [T]            — ground-truth token IDs.
 * @returns Scalar loss value.
 */
export function crossEntropyLoss(
  logits: np.Array,
  targets: np.Array,
): np.Array {
  const [T, V] = logits.shape;
  // Shift: predict token t+1 from logits at position t.
  const shiftedLogits = logits.slice([0, 0], [T - 1, V]);
  const shiftedTargets = targets.slice([1], [T]);

  // Gather the logit at each target position, then compute softmax CE.
  const logProbs = nnLogSoftmax(shiftedLogits);
  const gathered = gatherRows(logProbs, shiftedTargets);
  const nll = gathered.mul(-1);
  return nll.mean();
}

function nnLogSoftmax(x: np.Array): np.Array {
  const maxVal = x.max(-1, { keepdims: true });
  const shifted = x.sub(maxVal);
  const logSumExp = np.log(np.exp(shifted).sum(-1, { keepdims: true }));
  return shifted.sub(logSumExp);
}

function gatherRows(matrix: np.Array, indices: np.Array): np.Array {
  // Gather row i from matrix for each indices[i].
  // jax-js doesn't have a direct gather, so we use one-hot multiplication.
  const [_T, V] = matrix.shape;
  const oneHot = oneHotRows(indices, V);
  return matrix.mul(oneHot).sum(-1);
}

function oneHotRows(indices: np.Array, numClasses: number): np.Array {
  const eye = np.eye(numClasses);
  return eye.slice(indices); // [T, V]
}

// ── Training Runner ────────────────────────────────────────────────────────

export type TrainingBatch = {
  inputIds: np.Array; // [batch, T]
  targets: np.Array; // [batch, T]
};

export type TrainingConfig = {
  optimizer: OptimizerType;
  optimizerOptions: OptimizerOptions;
  epochs: number;
  batchSize: number;
  /** Optional callback invoked after each epoch. */
  onEpochEnd?: (epoch: number, loss: number) => void;
  /** Optional callback invoked after each batch. */
  onBatchEnd?: (batch: number, loss: number) => void;
};

/**
 * Orchestrates a fine-tuning loop over a loaded model.
 *
 * The `ModelRuntime` provides the model weights; this class handles the
 * gradient computation, optimizer updates, and loss tracking.
 *
 * @example
 * ```ts
 * const runner = new TrainingRunner(runtime.definition, config);
 * runner.init(modelWeights);
 * for (const epoch of runner.train(dataset)) {
 *   console.log(`Epoch ${epoch.index} loss: ${epoch.loss}`);
 * }
 * ```
 */
export class TrainingRunner {
  private optimizer: optax.GradientTransformation;
  // deno-lint-ignore no-explicit-any
  private optState?: any;
  // deno-lint-ignore no-explicit-any
  private params?: any;
  private config: TrainingConfig;

  constructor(
    private definition: ModelDefinition,
    config: TrainingConfig,
  ) {
    this.config = config;
    this.optimizer = createOptimizer(config.optimizer, config.optimizerOptions);
  }

  /** Initialize optimizer state from the current model parameters. */
  // deno-lint-ignore no-explicit-any
  init(params: any): void {
    this.params = params;
    this.optState = this.optimizer.init(params);
  }

  /**
   * Run a single training step on one batch.
   * Returns the loss value for that batch.
   */
  async step(batch: TrainingBatch): Promise<number> {
    if (!this.params || !this.optState) {
      throw new Error("TrainingRunner not initialized. Call `init()` first.");
    }

    // deno-lint-ignore no-explicit-any
    const lossFn = (params: any, inputIds: np.Array, targets: np.Array) => {
      // This is a placeholder: actual forward pass depends on the model
      // architecture. The model definition would need to expose a training
      // forward pass. For now, we define the interface so users can plug in
      // model-specific forward functions.
      const logits = this.forwardPass(params, inputIds);
      return crossEntropyLoss(logits, targets);
    };

    const [loss, grads] = computeLossAndGrad(
      lossFn,
      this.params,
      batch.inputIds,
      batch.targets,
    );

    const [updates, newOptState] = this.optimizer.update(
      grads,
      this.optState,
      this.params,
    );
    this.params = optax.applyUpdates(this.params, updates);
    this.optState = newOptState;

    const lossData = await loss.data();
    return lossData[0] as number;
  }

  /**
   * Train over a dataset for the configured number of epochs.
   * Yields per-epoch summary objects.
   */
  async *train(
    dataset: TrainingBatch[],
  ): AsyncGenerator<{ index: number; loss: number }> {
    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      let totalLoss = 0;
      let batchCount = 0;

      for (let i = 0; i < dataset.length; i++) {
        const loss = await this.step(dataset[i]);
        totalLoss += loss;
        batchCount++;
        this.config.onBatchEnd?.(i, loss);
      }

      const avgLoss = batchCount > 0 ? totalLoss / batchCount : 0;
      this.config.onEpochEnd?.(epoch, avgLoss);
      yield { index: epoch, loss: avgLoss };
    }
  }

  /** Get the current (potentially updated) parameters. */
  // deno-lint-ignore no-explicit-any
  getParams(): any {
    if (!this.params) {
      throw new Error("TrainingRunner not initialized.");
    }
    return this.params;
  }

  /**
   * Model-specific forward pass.
   *
   * This is a hook that model implementations can override. By default it
   * throws — each model family (Gemma, LFM, etc.) needs to provide a
   * training-mode forward pass that accepts raw parameters instead of the
   * session-based prefill/step interface.
   */
  // deno-lint-ignore no-explicit-any
  private forwardPass(_params: any, _inputIds: np.Array): np.Array {
    throw new Error(
      `Training forward pass not implemented for model "${this.definition.id}". ` +
        `Override forwardPass() in a subclass or provide a training adapter.`,
    );
  }
}

// ── Gradient computation ──────────────────────────────────────────────────

/**
 * Compute loss and gradients w.r.t. parameters using jax-js autodiff.
 * Returns [lossArray, gradientTree].
 */
function computeLossAndGrad(
  // deno-lint-ignore no-explicit-any
  lossFn: (params: any, inputIds: np.Array, targets: np.Array) => np.Array,
  // deno-lint-ignore no-explicit-any
  params: any,
  inputIds: np.Array,
  targets: np.Array,
  // deno-lint-ignore no-explicit-any
): [np.Array, any] {
  const lossGradFn = grad(lossFn);
  const grads = lossGradFn(params, inputIds, targets);
  const loss = lossFn(tree.ref(params), inputIds, targets);
  return [loss, grads];
}
