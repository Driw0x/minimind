# MiniMind --- DirectML Technical Issues

> **Project direction update — 2026-09-21**
>
> The DirectML work is now retained as a compatibility and feasibility study,
> not as the active MiniMind training backend. The upstream official
> `pretrain_768.pth` checkpoint generates coherent text on both CPU and
> DirectML, while the locally trained DirectML checkpoints remained incoherent
> after epoch 1 and epoch 2 despite apparently healthy loss and checkpoint
> diagnostics. This isolates the unresolved problem to the custom DirectML
> training path rather than the tokenizer, dataset, checkpoint loader, or
> DirectML inference path.
>
> Because acceptable pretraining quality could not be obtained reliably with
> DirectML, the DirectML training track is **abandoned for this project**.
> Development is moving to **ROCm**. DirectML benchmarks, issues, workarounds,
> commands, and validation results below are preserved as historical technical
> evidence unless explicitly stated otherwise.

This document records technical compatibility issues encountered while
adapting MiniMind to DirectML.

Benchmark and performance-related observations are documented separately
in [`directml_benchmarks.md`](directml_benchmarks.md).

------------------------------------------------------------------------

# DirectML Device Displayed as `privateuseone:0`

## Problem

After initializing DirectML, PyTorch reports:

``` text
privateuseone:0
```

instead of a device name such as `directml:0`.

## Cause

`torch-directml` integrates DirectML into PyTorch through the
`PrivateUse1` backend.

Therefore, `privateuseone:0` is the expected internal PyTorch
representation of the DirectML device.

## Solution

No code change was required.

Device placement was verified by checking model parameters, inputs, and
outputs.

## Decision

`privateuseone:0` is treated as expected DirectML behavior.

The user-facing device option remains:

``` text
directml
```

------------------------------------------------------------------------

# DirectML Not Available as an Original Device Option

## Problem

Upstream MiniMind did not provide DirectML as an execution device.

## Cause

MiniMind was not originally designed around `torch-directml`, which
requires explicit initialization through:

``` python
import torch_directml

device = torch_directml.device()
```

## Solution

`directml` was added as an explicit device option.

The resulting device continues to use standard PyTorch operations:

``` python
model.to(device)
tensor.to(device)
```

## Decision

DirectML should reuse the existing MiniMind execution pipeline instead
of introducing a separate training implementation.

------------------------------------------------------------------------

# AdamW CPU Fallback

## Problem

During an AdamW optimizer step, DirectML reports:

``` text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

## Cause

DirectML does not implement every PyTorch operation used internally by
`torch.optim.AdamW`.

`torch-directml` automatically executes the unsupported operation on
CPU.

## Solution

The complete training step was validated despite the warning:

``` text
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

## Decision

The fallback is accepted because it does not prevent correct training.

It remains documented because it may affect performance.

------------------------------------------------------------------------

# Checkpoint and Model Architecture Mismatch

## Problem

`eval_llm.py` failed when loading a validation checkpoint with strict
state-dictionary matching.

## Cause

The evaluation model architecture did not match the architecture used
during training.

Checkpoint compatibility depends on parameters such as:

``` text
hidden_size
num_hidden_layers
vocab_size
```

The validation checkpoint used:

``` text
hidden_size = 128
num_hidden_layers = 2
```

## Solution

Evaluation must instantiate a compatible model.

For the validation configuration:

``` powershell
--hidden_size 128 `
--num_hidden_layers 2
```

## Decision

Checkpoint-loading failures should not automatically be attributed to
DirectML.

Architecture compatibility must be verified first.

------------------------------------------------------------------------

# Reward Model Fails on DirectML

## Problem

During GRPO training, the main MiniMind model works on DirectML while
the reward model fails when forced onto the same device.

## Cause

The reward-model inference path uses operations that are not reliably
supported by DirectML.

The original workflow also assumed that participating models could share
the same execution device.

## Solution

The trainable model remains on DirectML while the reward model executes
on CPU:

``` text
Trainable model → DirectML
Reward model    → CPU
```

Reward-model inputs are moved to CPU before inference.

Reward values are transferred back to the training device when required.

## Decision

The reward model intentionally remains on CPU under DirectML.

Correctness and stability take priority over forcing every component
onto the GPU.

------------------------------------------------------------------------

# Duplicated Device Handling Across Trainers

## Problem

As DirectML support expanded, device-specific handling began to appear
across individual trainers.

This risked duplicated and inconsistent backend logic.

## Cause

DirectML introduces component-specific device requirements that do not
fit the original assumption of one shared execution device.

## Solution

Device and backend compatibility handling was centralized in shared
trainer utilities.

These utilities provide a common location for:

-   device resolution;
-   DirectML initialization;
-   model placement;
-   component-specific device requirements;
-   future backend compatibility rules.

## Decision

Individual trainers should remain focused on their training algorithm.

Backend compatibility belongs in shared trainer infrastructure.

------------------------------------------------------------------------

# Empty Token Handling During Generation

## Problem

Generation-based training could produce an empty token sequence.

Passing the empty result downstream could create an invalid training
sample.

## Cause

The generation path assumed that generated output would always contain
usable token content.

There was no shared protection against an empty sequence.

## Solution

Empty-token handling was added to shared trainer utilities.

Generated sequences are validated before downstream processing, and a
safe fallback is used when necessary.

## Decision

Shared generation utilities must not return unusable empty token
sequences.

The protection belongs in common utilities rather than individual GRPO
or PPO implementations.

------------------------------------------------------------------------

# Empty Distillation Masks

## Problem

Some truncated SFT samples may contain no supervised tokens within the
configured sequence length.

This caused the distillation KL-divergence computation to receive empty
logits and fail.

## Cause

The distillation loss assumed that every training sample contained at
least one supervised token after truncation and masking.

This assumption does not always hold for short or truncated validation
samples.

## Solution

The distillation loss now detects empty token selections and returns a
zero loss connected to the computation graph.

## Decision

Training losses that operate on masked token selections must safely
handle valid empty selections.

This is treated as a training robustness issue rather than a DirectML
limitation.

------------------------------------------------------------------------

# Checkpoint Compatibility Across Training Stages

## Problem

DirectML modifications could potentially affect checkpoint
interoperability between MiniMind training stages.

Relevant stages include:

``` text
Pretraining
SFT
DPO
GRPO
PPO
```

## Cause

The training stages depend on compatible architectures and state
dictionaries.

Device handling must remain independent from checkpoint representation.

## Solution

Checkpoint compatibility is tested independently from DirectML
execution.

## Decision

DirectML must not introduce a backend-specific checkpoint format.

Existing MiniMind checkpoint semantics are preserved.

------------------------------------------------------------------------

# Training Measurement Introduced DirectML Synchronization Overhead

## Problem

The initial real-data benchmark reported approximately:

``` text
5.280 s / iteration
```

which implied an estimated pretraining duration of approximately
`9.70 days / epoch`.

This made DirectML training appear substantially slower than expected.

## Cause

The training measurement path introduced unnecessary synchronization
between asynchronous DirectML execution and the CPU.

Frequent materialization of device-produced values on the CPU caused the
benchmark to measure synchronization and instrumentation overhead in
addition to the actual training workload.

## Solution

The critical training and measurement path was adjusted to avoid
unnecessary per-step DirectML-to-CPU synchronization.

The same reference workload was then rerun for 100 steps:

``` text
batch_size = 8
max_seq_len = 340
gradient_accumulation_steps = 8
```

The corrected benchmark measured:

``` text
Average iteration time:    0.482 s
Samples / second:          16.61
Effective tokens / second: 3341.37
Estimated epoch duration:  21.24 h
```

## Decision

Performance measurements must not introduce unnecessary synchronization
into the critical training path.

Unexpectedly slow DirectML performance should be investigated for
measurement, logging, and synchronization overhead before being
attributed to the backend itself.

------------------------------------------------------------------------

# Multiple GPUs and DirectML Device Selection

## Problem

The development machine exposes two graphics adapters:

``` text
Integrated GPU (iGPU)
Dedicated GPU (dGPU)
```

This made it unclear which physical GPU DirectML was using during
training and benchmarking.

## Cause

The integrated GPU and dedicated GPU are separate DirectML-compatible
graphics adapters.

Therefore:

``` text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

Implicit device selection does not provide sufficient certainty about
the physical GPU executing the workload.

## Solution

DirectML adapter selection must explicitly target the intended GPU.

For MiniMind training, the dedicated graphics card should be selected.

Conceptually:

``` text
Available DirectML adapters
        ↓
Identify dedicated GPU
        ↓
Explicitly select adapter
        ↓
Create DirectML device
        ↓
MiniMind training
```

GPU utilization can then be verified against the corresponding adapter
in Windows Task Manager.

## Decision

The dedicated GPU is the intended MiniMind training device.

Performance benchmarks should only be compared when they were obtained
using the same explicitly selected physical GPU.

# DirectML FP16 AdamW Instability

## Problem

DirectML FP16 pretraining became non-finite immediately after the first
optimizer update when AdamW used its default epsilon:

``` text
eps = 1e-8
```

Static loss scaling with a scale of `1024` did not solve the failure by
itself.

## Cause

The gradients were finite before the optimizer step, which isolated the
failure to the optimizer update path rather than the forward or backward
pass.

The default AdamW epsilon is too small for the validated pure-FP16
DirectML optimizer path.

## Solution

The DirectML FP16 path uses:

``` text
Static loss scale = 1024
AdamW epsilon     = 1e-4
```

The behavior is centralized in shared trainer utilities and reused by
the DirectML FP16 trainers.

## Decision

DirectML FP16 training must not use the default AdamW epsilon without
validation.

The validated epsilon for the historical direct-FP16 optimizer path is
`1e-4`. The retained Dense FP32-master optimizer path uses AdamW
`eps = 1e-8`.

------------------------------------------------------------------------

# Inconsistent Bounded-Run Support Across Trainers

## Problem

The initial M4 bounded-run implementation added `--max_steps` to
pretraining first, while other trainers still expected complete
epoch-based execution.

This prevented one consolidated smoke runner from controlling every
training workload consistently.

## Solution

Bounded global `--max_steps` handling was propagated across the trainers
used by the DirectML FP16 smoke suite.

A Python runner was added at:

``` text
tests/test_all_trainers.py
```

It launches the trainers sequentially and stops on the first failure.

## Decision

Bounded execution is part of the shared validation strategy for
trainable DirectML workflows.

Heavy trainer smoke tests are executed explicitly rather than being
treated as ordinary lightweight Pytest tests.


------------------------------------------------------------------------

# MoE Routing Uses Unsupported Scatter Behavior on DirectML

## Problem

The upstream MoE routing path failed during DirectML validation.

The first observed failure occurred around the auxiliary expert-routing
logic using `F.one_hot`. Replacing that operation alone allowed the
forward pass to progress, but backward execution still encountered
unsupported scatter behavior in the sparse routing path.

## Cause

The upstream sparse MoE implementation relies on indexed routing and
scatter-like operations that are not fully supported by the tested
DirectML forward/backward path.

## Solution

A DirectML-specific scatter-free routing path was added.

Conceptually:

``` text
Gate scores
    ↓
Top-k expert selection
    ↓
Broadcast routing mask
    ↓
Differentiable routing weights
    ↓
Evaluate experts
    ↓
Weighted expert combination
```

CPU and CUDA retain the original sparse routing implementation.

## Decision

DirectML uses the scatter-free MoE compatibility path.

The fallback prioritizes correct execution over sparse-MoE efficiency
and therefore should not be used to infer native sparse-MoE performance.

------------------------------------------------------------------------

# Agent RL Compatibility on Windows and DirectML FP16

## Problem

Agent RL encountered two independent issues during the final M4 smoke
validation.

First, Windows DataLoader workers failed to import a locally defined
`collate_fn`.

Second, after fixing multiprocessing, the first Agent training step
produced a non-finite loss.

## Cause

Windows multiprocessing uses process spawning, so the DataLoader
`collate_fn` must be importable from module scope.

For the numerical failure, targeted finite-value checks showed:

``` text
rewards     = finite
advantages  = finite
behavior    = finite
policy      = non-finite
reference   = non-finite
```

Additional checks confirmed that policy logits contained NaNs during the
full-sequence policy/reference recomputation path using
`attention_mask=full_mask`.

Rollout generation itself remained finite.

## Solution

The Agent `collate_fn` was moved to module scope for Windows
multiprocessing compatibility.

For the right-padded Agent batches, policy and reference full-sequence
recomputation avoid the problematic DirectML FP16 attention-mask path.

Numerically sensitive log-probability, KL, and ratio calculations are
performed in FP32.

## Decision

The Agent-specific compatibility handling is retained because it fixes
the DirectML FP16 path without requiring a broader change to MiniMind
attention semantics.

After these changes, Agent RL passed and the consolidated smoke suite
completed successfully with:

``` text
Passed: 9/9
```

------------------------------------------------------------------------

# Long-Run DirectML FP16 Pretraining Collapse

## Problem

A full Dense pretraining run remained finite but later produced a
degenerate self-copying model.

Teacher-forced diagnostics showed approximately:

``` text
Mean loss:           13.57
Top-1 accuracy:      0.18%
Top-1 repeat rate:  93.48%
```

An untrained model did not show this behavior.

## Cause

The DirectML FP16 path updated FP16 model parameters directly with
AdamW. Static loss scaling protected gradients but did not provide the
FP32 master weights used by conventional mixed-precision optimization.

## Solution

Dense DirectML pretraining now uses:

``` text
FP16 model compute
        ↓
FP32 gradient unscale
        ↓
AdamW on FP32 master weights
        ↓
FP32 master weights copied back to FP16 model
```

The FP32 optimizer path uses AdamW epsilon `1e-8`.

A 100-step validation improved mean diagnostic loss from `8.89` for an
untrained model to `7.48`. After resuming to global step `1100`, mean
loss reached `6.75`, Top-1 accuracy reached `6.30%`, and Top-1 repeat
rate remained low at `0.89%`.

## Decision

Finite loss alone is not sufficient to validate DirectML FP16
pretraining.

Dense pretraining uses FP32 master weights for optimizer updates, and
teacher-forced checkpoint diagnostics are retained to detect silent
training collapse.

------------------------------------------------------------------------

# DirectML Cross-Entropy Reduction and Valid-Token Normalization

## Problem

On the tested DirectML FP16 path, causal-LM cross-entropy with
`ignore_index=-100` did not reproduce the expected valid-token
normalization.

The default `reduction="mean"` divided the effective loss by all shifted
positions instead of only valid non-ignored tokens.

An initial workaround using:

``` python
loss = F.cross_entropy(x, y, ignore_index=-100, reduction="sum")
loss = loss / (y != -100).sum().clamp_min(1)
```

was later found to be incorrect as well on the tested DirectML FP16
path.

## Cause

Targeted CPU/DirectML tests first confirmed the incorrect denominator of
the DirectML mean reduction when ignored padding tokens were present.

A second reduction comparison then evaluated the same model logits with:

``` text
A = DirectML reduction="sum" / valid tokens
B = DirectML reduction="none" → valid tokens → FP32 mean
C = CPU FP32 reference
```

The results were:

``` text
Batch 8:
A = 14.1078
B =  6.6688
C =  6.6715

Batch 32:
A = 10.7772
B =  6.3027
C =  6.3053
```

`B` matched the CPU FP32 reference within approximately `0.003`, while
`A` did not.

## Solution

The final DirectML causal-LM loss uses per-token cross-entropy followed
by FP32 averaging over valid tokens:

``` python
token_loss = F.cross_entropy(
    x,
    y,
    ignore_index=-100,
    reduction="none",
)

valid = y != -100
loss = token_loss[valid].float().mean()
```

After applying the change, the model training loss on the batch-32
reference test became:

``` text
Model outputs.loss:             6.30274916
DirectML none → FP32 mean:      6.30274916
CPU FP32 reference:             6.30530691
|model - CPU|:                  0.0025577545
```

## Decision

DirectML causal-LM training must not rely on either the backend default
mean reduction or the tested FP16 `sum / valid_tokens` workaround.

The retained implementation computes per-token losses with
`reduction="none"` and performs the valid-token mean in FP32.

------------------------------------------------------------------------

# Pretraining Diagnostic Loss Aggregation Mismatch

## Problem

After the first complete corrected Dense pretraining epoch,
`diagnose_pretrain.py` reported aggregate train-path losses around
`12–13`, despite improving Top-1 accuracy and a low repeat rate.

## Cause

The investigation initially suggested a diagnostic aggregation problem,
but the dedicated reduction test later showed that the current
`reduction="sum" / valid_tokens` training path itself was incorrect on
the tested DirectML FP16 backend.

The diagnostic's manual token loss remained useful because it was close
to the independent CPU FP32 reference.

## Solution

The diagnostic was changed to use batched evaluation, and the model loss
implementation was replaced with per-token DirectML cross-entropy
followed by FP32 valid-token averaging.

The decisive batch-32 validation produced:

``` text
Model outputs.loss:        6.30274916
Manual DirectML loss:      6.30274916
CPU FP32 reference:        6.30530691
Difference vs CPU:         0.0025577545
```

## Decision

Checkpoint diagnostics must compare the model training path with an
independent reference on the same logits.

The `reduction="none"` → FP32 valid-token mean implementation is the
retained DirectML loss path.
------------------------------------------------------------------------

# DirectML BF16 Tensor Execution Is Unsupported in the Current Stack

## Problem

An upstream-like DirectML pretraining experiment attempted to use
`bfloat16` model tensors directly.

The run failed before the first training step with:

``` text
Invalid or unsupported data type BFloat16
```

from the DirectML backend path.

## Cause

The tested PyTorch / `torch-directml` / `PrivateUse1` execution stack
does not accept the requested BF16 tensor type for this workload.

This result establishes a limitation of the current software stack. It
does not by itself prove that the physical GPU hardware is incapable of
BF16 arithmetic.

The experiment also differs from the upstream CUDA mixed-precision path:
upstream keeps model parameters in FP32 and uses BF16 autocast for
compatible compute operations, whereas the DirectML experiment had to
request BF16 tensors directly because an equivalent validated DirectML
autocast path is not available.

## Solution

DirectML BF16 is not used for the current MiniMind training path.

The retained mixed-precision strategy remains:

``` text
FP16 model compute
        ↓
FP32 master weights
        ↓
FP32 AdamW update
        ↓
FP32 master → FP16 model synchronization
```

## Decision

Do not treat DirectML BF16 as a usable replacement for CUDA BF16
autocast in the current environment.

------------------------------------------------------------------------

# Fast Direct FP16 Optimizer Re-Test

## Problem

To reduce the overhead of FP32 master weights, Dense pretraining was
re-tested with an experimental fast path using AdamW directly on FP16
model parameters.

The live training loss remained finite and decreased, but checkpoint
diagnostics showed poor learning followed by rapidly increasing
self-repetition.

Results on samples `0–255` were:

``` text
Step 1000:
loss   = 9.2573
Top-1  = 0.31%
repeat = 0.29%

Step 2000:
loss   = 8.3102
Top-1  = 0.92%
repeat = 1.20%

Step 5000:
loss   = 7.3196
Top-1  = 1.80%
repeat = 39.24%
```

The true-data repeat rate remained approximately `0.29%`.

## Cause

The experimental path removed FP32 master parameters and applied AdamW
updates directly to FP16 parameters.

The results reinforce the earlier finding that static loss scaling can
protect the backward signal without providing sufficient precision for
reliable long-run FP16 parameter updates.

## Solution

The fast direct-FP16 optimizer path is not retained.

Dense DirectML pretraining continues to use FP16 model compute with FP32
master weights and FP32 AdamW updates.

## Decision

A decreasing training loss is not sufficient to validate direct FP16
optimization.

FP32 master weights remain required for the retained Dense DirectML FP16
pretraining architecture.

------------------------------------------------------------------------

# Diagnostic CPU Synchronization Limited to Logging Intervals

## Problem

The stable FP16 + FP32-master trainer still performed diagnostic
DirectML-to-CPU synchronization more often than required for the actual
optimizer update.

Examples included:

``` text
finite-loss read on every batch
gradient-norm scalar read on optimizer updates
```

These scalar reads force the CPU to wait for asynchronous DirectML work.

## Solution

An optimized trainer variant keeps the complete FP32-master optimization
path unchanged but disables per-step scalar materialization for
diagnostic checks.

Loss and the latest gradient norm are transferred to the CPU and checked
only when:

``` text
step % log_interval == 0
```

or at the final logging point.

Gradient clipping itself still executes on every optimizer update, and
FP32 master weights are still synchronized back to the FP16 model after
every optimizer step.

## Decision

Diagnostic CPU synchronization should follow `log_interval` rather than
the optimizer cadence when possible.

A `1000`-step benchmark confirmed a throughput improvement from
`2.0038 s/step` to `1.9307 s/step`, corresponding to a `1.038×` speedup
and `3.65%` lower total runtime.

The throughput impact is validated for the tested configuration.
Checkpoint-quality validation remains required before the optimized
trainer is treated as the final training reference.

------------------------------------------------------------------------

# Final Blocking Issue — Full-Training Model Quality

## Problem

After the final two-epoch Dense pretraining run, the DirectML-trained checkpoint
still produced incoherent generation. The same behavior was already visible in
the epoch-1 checkpoint.

By contrast, the official upstream `pretrain_768.pth` checkpoint produced
coherent text when evaluated on both CPU and DirectML.

## Evidence

The following candidates were checked and did not explain the discrepancy:

- tokenizer encode/decode behavior;
- pretraining dataset readability and content;
- checkpoint loading and evaluation path;
- DirectML inference itself, because the official checkpoint generates
  coherently on DirectML.

Earlier DirectML fixes solved real execution and numerical problems, including
FP16 optimizer precision, cross-entropy reduction, MoE routing, attention-mask
compatibility, and synchronization overhead. Nevertheless, those fixes did not
produce a locally trained checkpoint with acceptable generation quality.

## Decision

The project no longer treats successful DirectML execution, finite losses, or
matching diagnostic losses as sufficient evidence of a valid training backend.

The unresolved full-training quality gap is considered a blocking DirectML
training issue for this project. Further DirectML training work is stopped, the
existing implementation is preserved for documentation and regression, and the
active training path moves to **ROCm**.
