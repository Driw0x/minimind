# MiniMind --- DirectML Technical Issues

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

The validated project default for the DirectML FP16 path is `1e-4`.

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
