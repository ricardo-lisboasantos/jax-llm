# Training

Fine-tuning utilities in `engine/runtime/training.ts`, built on optax. They
operate on raw JAX arrays and weight trees, not chat abstractions.

## Optimizer

```typescript
import { createOptimizer } from "@ricardo/jax-llm";

const optimizer = createOptimizer("adamw", {
  learningRate: 1e-4,
  weightDecay: 0.01,
  maxGradNorm: 1.0, // 0 or omitted = no clipping
});
```

`OptimizerType` is `"sgd" | "adam" | "adamw"`. `OptimizerOptions`:
`learningRate` (required), `weightDecay?`, `maxGradNorm?`, `beta1?`, `beta2?`,
`eps?`.

## Loss

`crossEntropyLoss(logits, targets)` — standard causal language-modeling loss.
`logits` is `[T, vocabSize]`, `targets` is `[T]`; both are shifted internally so
position `t` predicts token `t+1`.

## `TrainingRunner`

```typescript
import { TrainingRunner } from "@ricardo/jax-llm";

const runner = new TrainingRunner(runtime.definition, {
  optimizer: "adamw",
  optimizerOptions: { learningRate: 1e-4 },
  epochs: 3,
  batchSize: 4,
  onBatchEnd: (batch, loss) => console.log(batch, loss),
  onEpochEnd: (epoch, loss) => console.log(epoch, loss),
});
runner.init(modelWeights);
for await (const epoch of runner.train(dataset)) {
  console.log(`Epoch ${epoch.index} loss: ${epoch.loss}`);
}
const updated = runner.getParams();
```

- `TrainingBatch` is `{ inputIds, targets }`; `TrainingConfig` is
  `{ optimizer, optimizerOptions, epochs, batchSize, onEpochEnd?,
  onBatchEnd? }`.
- `step(batch)` runs one batch and returns its loss; `train(dataset)` yields
  per-epoch `{ index, loss }` summaries.
- The default `forwardPass` hook throws per model — each family needs a
  training-mode forward pass; override it in a subclass or training adapter.
