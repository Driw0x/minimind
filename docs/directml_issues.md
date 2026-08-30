# MiniMind — DirectML Technical Issues

This document records technical compatibility issues encountered while adapting MiniMind to DirectML.

Benchmark and performance-related observations are documented separately in [`directml_benchmarks.md`](directml_benchmarks.md).

---

# DirectML Device Displayed as `privateuseone:0`

## Problem

After initializing DirectML, PyTorch reports:

```text
privateuseone:0
```

instead of a device name such as `directml:0`.

## Cause

`torch-directml` integrates DirectML into PyTorch through the `PrivateUse1` backend.

Therefore, `privateuseone:0` is the expected internal PyTorch representation of the DirectML device.

## Solution

No code change was required.

Device placement was verified by checking model parameters, inputs, and outputs.

## Decision

`privateuseone:0` is treated as expected DirectML behavior.

The user-facing device option remains:

```text
directml
```

---

# DirectML Not Available as an Original Device Option

## Problem

Upstream MiniMind did not provide DirectML as an execution device.

## Cause

MiniMind was not originally designed around `torch-directml`, which requires explicit initialization through:

```python
import torch_directml

device = torch_directml.device()
```

## Solution

`directml` was added as an explicit device option.

The resulting device continues to use standard PyTorch operations:

```python
model.to(device)
tensor.to(device)
```

## Decision

DirectML should reuse the existing MiniMind execution pipeline instead of introducing a separate training implementation.

---

# AdamW CPU Fallback

## Problem

During an AdamW optimizer step, DirectML reports:

```text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

## Cause

DirectML does not implement every PyTorch operation used internally by `torch.optim.AdamW`.

`torch-directml` automatically executes the unsupported operation on CPU.

## Solution

The complete training step was validated despite the warning:

```text
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

## Decision

The fallback is accepted because it does not prevent correct training.

It remains documented because it may affect performance.

---

# Checkpoint and Model Architecture Mismatch

## Problem

`eval_llm.py` failed when loading a validation checkpoint with strict state-dictionary matching.

## Cause

The evaluation model architecture did not match the architecture used during training.

Checkpoint compatibility depends on parameters such as:

```text
hidden_size
num_hidden_layers
vocab_size
```

The validation checkpoint used:

```text
hidden_size = 128
num_hidden_layers = 2
```

## Solution

Evaluation must instantiate a compatible model.

For the validation configuration:

```powershell
--hidden_size 128 `
--num_hidden_layers 2
```

## Decision

Checkpoint-loading failures should not automatically be attributed to DirectML.

Architecture compatibility must be verified first.

---

# Reward Model Fails on DirectML

## Problem

During GRPO training, the main MiniMind model works on DirectML while the reward model fails when forced onto the same device.

## Cause

The reward-model inference path uses operations that are not reliably supported by DirectML.

The original workflow also assumed that participating models could share the same execution device.

## Solution

The trainable model remains on DirectML while the reward model executes on CPU:

```text
Trainable model → DirectML
Reward model    → CPU
```

Reward-model inputs are moved to CPU before inference.

Reward values are transferred back to the training device when required.

## Decision

The reward model intentionally remains on CPU under DirectML.

Correctness and stability take priority over forcing every component onto the GPU.

---

# Duplicated Device Handling Across Trainers

## Problem

As DirectML support expanded, device-specific handling began to appear across individual trainers.

This risked duplicated and inconsistent backend logic.

## Cause

DirectML introduces component-specific device requirements that do not fit the original assumption of one shared execution device.

## Solution

Device and backend compatibility handling was centralized in shared trainer utilities.

These utilities provide a common location for:

* device resolution;
* DirectML initialization;
* model placement;
* component-specific device requirements;
* future backend compatibility rules.

## Decision

Individual trainers should remain focused on their training algorithm.

Backend compatibility belongs in shared trainer infrastructure.

---

# Empty Token Handling During Generation

## Problem

Generation-based training could produce an empty token sequence.

Passing the empty result downstream could create an invalid training sample.

## Cause

The generation path assumed that generated output would always contain usable token content.

There was no shared protection against an empty sequence.

## Solution

Empty-token handling was added to shared trainer utilities.

Generated sequences are validated before downstream processing, and a safe fallback is used when necessary.

## Decision

Shared generation utilities must not return unusable empty token sequences.

The protection belongs in common utilities rather than individual GRPO or PPO implementations.

---

# Checkpoint Compatibility Across Training Stages

## Problem

DirectML modifications could potentially affect checkpoint interoperability between MiniMind training stages.

Relevant stages include:

```text
Pretraining
SFT
DPO
GRPO
PPO
```

## Cause

The training stages depend on compatible architectures and state dictionaries.

Device handling must remain independent from checkpoint representation.

## Solution

Checkpoint compatibility is tested independently from DirectML execution.

## Decision

DirectML must not introduce a backend-specific checkpoint format.

Existing MiniMind checkpoint semantics are preserved.

---

# Multiple GPUs and DirectML Device Selection

## Problem

The development machine exposes two graphics adapters:

```text
Integrated GPU (iGPU)
Dedicated GPU (dGPU)
```

This made it unclear which physical GPU DirectML was using during training and benchmarking.

## Cause

The integrated GPU and dedicated GPU are separate DirectML-compatible graphics adapters.

Therefore:

```text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

Implicit device selection does not provide sufficient certainty about the physical GPU executing the workload.

## Solution

DirectML adapter selection must explicitly target the intended GPU.

For MiniMind training, the dedicated graphics card should be selected.

Conceptually:

```text
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

GPU utilization can then be verified against the corresponding adapter in Windows Task Manager.

## Decision

The dedicated GPU is the intended MiniMind training device.

Performance benchmarks should only be compared when they were obtained using the same explicitly selected physical GPU.
