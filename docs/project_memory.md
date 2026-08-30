# MiniMind — Project Memory

This document records the technical problems encountered while adapting MiniMind to DirectML.

Problems are grouped by the milestone during which they were encountered.

Each entry focuses on:

* the problem;
* its cause;
* the implemented solution;
* the resulting technical decision when relevant.

Development progress and completed features are tracked separately in `update_log.md`.

---

# M1 — DirectML Foundation

## DirectML Device Displayed as `privateuseone:0`

### Problem

After initializing DirectML, PyTorch reports the device as:

```text
privateuseone:0
```

instead of a device name such as:

```text
directml:0
```

This initially made it unclear whether the model was actually running on DirectML.

### Cause

`torch-directml` integrates DirectML into PyTorch through the `PrivateUse1` backend.

A DirectML device is therefore internally exposed by PyTorch as:

```text
privateuseone:0
```

This is expected behavior and does not mean that execution is falling back to CPU.

### Solution

No code change was required.

Device placement was verified by checking the model, inputs, and outputs:

```python
print(next(model.parameters()).device)
print(input_ids.device)
print(logits.device)
```

Expected output:

```text
privateuseone:0
```

### Decision

`privateuseone:0` is treated as the expected internal representation of a DirectML device.

The user-facing device name remains:

```text
directml
```

---

## DirectML Not Available as an Original Device Option

### Problem

The original MiniMind scripts did not provide DirectML as an explicit device option.

This prevented DirectML from being selected through the existing device configuration.

### Cause

Upstream MiniMind was not designed with `torch-directml` as one of its execution backends.

DirectML also requires explicit initialization through:

```python
torch_directml.device()
```

rather than the standard CUDA device path.

### Solution

Added:

```text
directml
```

as an explicit device option.

When selected, the DirectML device is initialized through:

```python
import torch_directml

device = torch_directml.device()
```

The resulting device can then be used with standard PyTorch device-independent operations:

```python
model.to(device)
tensor.to(device)
```

### Decision

DirectML support should reuse the existing MiniMind execution pipeline instead of introducing separate DirectML-specific training or evaluation implementations.

Backend differences should be handled through device resolution whenever possible.

---

## AdamW CPU Fallback

### Problem

During the AdamW optimizer step, PyTorch reports:

```text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

The affected operation is:

```text
aten::lerp.Scalar_out
```

### Cause

DirectML does not implement every PyTorch operation used internally by:

```python
torch.optim.AdamW
```

When an unsupported operation has a CPU implementation available, `torch-directml` can automatically fall back to CPU execution.

### Solution

The complete optimizer and training step were tested despite the warning:

```text
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

The fallback does not prevent training and therefore does not currently require a workaround.

### Decision

CPU fallbacks are acceptable when:

* training remains correct;
* gradients remain valid;
* execution does not crash.

Fallbacks should still be documented because they may affect training performance.

---

# M2 — Pretraining and Evaluation Integration

## Checkpoint and Model Architecture Mismatch

### Problem

`eval_llm.py` failed while loading a checkpoint using strict state-dictionary matching:

```python
model.load_state_dict(
    torch.load(ckp, map_location="cpu"),
    strict=True
)
```

The error reported incompatible model parameters.

### Cause

The model instantiated during evaluation did not initially match the architecture used to create the checkpoint.

MiniMind checkpoints depend on model configuration parameters such as:

```text
hidden_size
num_hidden_layers
vocab_size
```

For example, the validation checkpoint was produced using:

```text
hidden_size = 128
num_hidden_layers = 2
```

Loading it into a differently configured model produces missing keys, unexpected keys, or incompatible tensor dimensions.

The issue was therefore unrelated to DirectML.

### Solution

The evaluation model must use an architecture compatible with the model used during training.

For the validation model:

```powershell
--hidden_size 128 `
--num_hidden_layers 2
```

must be supplied when evaluating the corresponding checkpoint.

The required relationship is:

```text
Training configuration
        ↓
Checkpoint architecture
        ↓
Evaluation configuration
```

### Decision

Checkpoint-loading errors should not automatically be attributed to DirectML.

Model configuration compatibility must be verified first.

---

## Isolated DirectML Tests Were Not Sufficient

### Problem

Successful forward, backward, and optimizer tests confirmed that basic MiniMind operations worked with DirectML, but they did not guarantee that the complete MiniMind workflow was compatible.

Potential failures could still occur during:

```text
Training
Checkpoint creation
Checkpoint loading
Evaluation
Generation
```

### Cause

DirectML compatibility depends on all PyTorch operations used throughout the complete workflow.

A minimal training step only exercises part of the actual MiniMind execution path.

### Solution

A deliberately small MiniMind model was used as an end-to-end validation configuration:

```text
hidden_size = 128
num_hidden_layers = 2
```

The complete pipeline was tested:

```text
Training
   ↓
DirectML
   ↓
Model weights
   ↓
Checkpoint loading
   ↓
eval_llm.py
   ↓
DirectML
   ↓
Text generation
```

### Decision

The small `128 / 2-layer` configuration is kept as a lightweight validation configuration for future DirectML changes.

New backend changes should be validated end-to-end rather than only through isolated PyTorch operations.

---

# M3 — Training Pipeline Compatibility

## Reward Model Fails on DirectML

### Problem

During GRPO training, the reward model works correctly on CPU but fails when executed on DirectML.

The main MiniMind model can run on DirectML, but forcing the reward model onto the same device causes the training workflow to fail.

Conceptually:

```text
Trainable model → DirectML
Reward model    → DirectML
                      ↓
             incompatible operation
                      ↓
                    crash
```

### Cause

The reward-model inference path uses operations that are not reliably supported by the DirectML backend.

The original device handling also assumed that all models participating in the training workflow could share the same execution device.

This assumption does not hold for the DirectML GRPO pipeline.

### Solution

The reward model is kept on CPU while the trainable MiniMind model remains on DirectML:

```text
Trainable MiniMind model
          ↓
       DirectML

Reward model
          ↓
         CPU
```

Reward-model inputs are placed on the reward model's device before inference.

The resulting reward values are transferred to the training device when required by the GRPO computation.

Conceptually:

```text
Generated samples
       ↓
Move reward inputs to CPU
       ↓
Reward model inference
       ↓
Reward scores
       ↓
Move scores to training device
       ↓
GRPO computation on DirectML
```

### Decision

The reward model intentionally remains on CPU when the main training backend is DirectML.

This is acceptable because the reward model is inference-only and is not updated by the optimizer.

Correctness and stability take priority over forcing every model onto DirectML.

---

## Duplicated Device Handling Across Trainers

### Problem

As DirectML support expanded across the training pipeline, device-specific logic was distributed across individual training scripts.

The reward-model CPU requirement also introduced cases where different components of the same training workflow needed different devices.

Handling this separately in each trainer would duplicate logic and could produce inconsistent behavior.

### Cause

The original workflow could largely assume a common execution device.

DirectML introduces more complex device requirements:

```text
Trainable model → DirectML
Reward model    → CPU
Other tensors   → appropriate component device
```

Embedding these rules directly into individual trainers would tightly couple training algorithms to backend-specific compatibility code.

### Solution

Shared device-handling logic was centralized in the trainer utilities.

The shared utilities handle backend/device compatibility while individual trainers remain focused on their training-stage logic.

The intended separation is:

```text
Training script
      ↓
Training-stage logic

trainer/utils
      ↓
Device and backend compatibility
```

This provides a common location for:

* device resolution;
* DirectML initialization;
* model placement;
* component-specific device handling;
* future backend compatibility rules.

### Decision

Backend compatibility should be centralized in shared trainer infrastructure whenever possible.

Individual trainers should not duplicate DirectML-specific device-management logic.

---

## Empty Token Handling During Generation

### Problem

During generation-based training, a generated result could produce an empty token sequence.

Passing an empty generated sequence to downstream processing could result in an invalid or unusable training sample.

This is particularly relevant for training stages that generate outputs before performing additional computations such as reward evaluation.

### Cause

The generation path assumed that generated outputs would always contain usable token content.

No shared protection guaranteed that the token sequence remained valid after generation and subsequent extraction or processing.

Conceptually:

```text
Model generation
      ↓
Generated tokens
      ↓
Empty sequence
      ↓
Downstream processing
      ↓
Invalid sample
```

### Solution

Empty-token handling was added to the shared trainer utilities.

Generated token sequences are checked before being passed to downstream training logic.

If generation results in an empty sequence, a safe fallback is used instead of allowing the invalid sequence to propagate.

Conceptually:

```text
Model generation
      ↓
Generated tokens
      ↓
Empty?
   ↙       ↘
 Yes       No
  ↓         ↓
Fallback   Continue
   ↘       ↙
Valid token sequence
      ↓
Downstream training
```

### Decision

Shared generation utilities must not return an unusable empty token sequence to downstream training logic.

The protection belongs in the common trainer utilities rather than in GRPO, PPO, or another individual trainer.

This prevents multiple generation-based training stages from implementing separate fixes for the same edge case.

---

## Checkpoint Compatibility Across Training Stages

### Problem

MiniMind uses multiple training stages, including pretraining, SFT, DPO, GRPO, and PPO.

Changes introduced while adding DirectML support could unintentionally affect model initialization or checkpoint loading and break compatibility between stages.

A training stage working correctly on DirectML does not by itself guarantee that its checkpoints remain compatible with the rest of the MiniMind pipeline.

### Cause

Training stages depend on compatible model architectures and state dictionaries.

DirectML support modifies device initialization and model placement, but those changes must remain independent from checkpoint structure.

The following concerns must remain separated:

```text
Device handling
      ↓
must not modify
      ↓
Model architecture
Checkpoint format
State dictionary semantics
```

### Solution

Checkpoint compatibility is validated independently from DirectML execution.

Compatibility tests ensure that DirectML-related changes do not alter the checkpoint format or expected state dictionaries used across the training stages.

### Decision

DirectML must not introduce a backend-specific checkpoint format.

Existing MiniMind checkpoint semantics are preserved so that checkpoints remain interoperable between training stages regardless of the execution device.

---

# General Decisions

## Keep DirectML Changes Close to Upstream

DirectML support should modify as little of the original MiniMind training logic as possible.

Prefer:

```python
device = ...
model.to(device)
tensor.to(device)
```

and shared compatibility utilities over DirectML-specific copies of existing pipelines.

This reduces divergence from upstream MiniMind and makes future maintenance easier.

---

## Separate Backend Compatibility From Training Logic

Backend-specific behavior belongs in shared infrastructure when possible.

The desired architecture is:

```text
Training algorithms
        ↓
Generic training logic

Shared trainer utilities
        ↓
Device / backend compatibility

PyTorch
        ↓
CPU / CUDA / DirectML
```

This became particularly important during M3 when different components required different execution devices.

---

## Accept CPU Fallbacks When Necessary

Using DirectML does not require every operation involved in training to execute on the GPU.

A mixed execution path is acceptable when required for compatibility:

```text
Main training → DirectML
Unsupported optimizer operation → CPU fallback
Reward model → CPU
```

The priority is:

```text
Correctness
    ↓
Stability
    ↓
Compatibility
    ↓
Performance optimization
```

CPU execution should only be treated as a problem when it unnecessarily prevents GPU acceleration or creates a significant performance bottleneck.

---

## Preserve MiniMind Checkpoint Semantics

DirectML is an execution backend and should not affect the logical representation of a MiniMind model.

Therefore:

```text
CPU checkpoint
DirectML checkpoint
CUDA checkpoint
```

should not become separate checkpoint formats.

Model architecture and training stage determine checkpoint compatibility, not the device used to execute the model.

---

## Batch Size and Sequence Length Constraints on DirectML

### Problem

During real training validation on DirectML, some combinations of `batch_size` and `max_seq_len` became extremely slow or could not complete within a reasonable amount of time.

For example, increasing both parameters can significantly increase the computational and memory requirements of a training step:

```text
batch_size ↑
     +
max_seq_len ↑
     ↓
Higher memory usage
Higher computation cost
Longer training steps
```

A configuration may therefore technically start on DirectML while still being unusable in practice because a single training step takes too long.

This also made automatic configuration testing inefficient: if a relatively lightweight configuration already fails or times out, testing a strictly heavier configuration may provide no useful information.

### Cause

`batch_size` and `max_seq_len` both directly affect the amount of data processed by the model during each training step.

Sequence length is particularly expensive for transformer models because attention computation grows rapidly as the sequence becomes longer.

DirectML also introduces additional performance constraints compared with the execution backends targeted by the original MiniMind defaults, including possible CPU fallbacks for unsupported operations.

As a result, upstream default training settings cannot be assumed to be practical on the current DirectML environment.

### Solution

Training configuration validation was changed to test progressively larger combinations of:

```text
batch_size
max_seq_len
```

instead of immediately assuming that the upstream defaults are usable.

Each candidate configuration can be tested with a limited number of training steps and a timeout.

Conceptually:

```text
Candidate configuration
        ↓
Run short training test
        ↓
Completes within timeout?
     ↙              ↘
   Yes               No
    ↓                 ↓
Record success     Record failure
    ↓
Try larger configuration
```

The benchmark progression also takes configuration dominance into account.

If a configuration fails, configurations that are strictly more demanding do not need to be tested when the failure already demonstrates that they cannot reasonably improve the situation.

For example:

```text
batch=8, seq_len=128
        ↓
      fails
        ↓
batch=16, seq_len=128
        ↓
      skip
```

because increasing the batch size while keeping the same sequence length cannot reduce the workload.

However, configurations that trade one dimension for another may still be useful to test:

```text
batch=8,  seq_len=256
        ↓
timeout / too expensive
        ↓
batch=16, seq_len=128
        ↓
may still be tested
```

because the second configuration reduces sequence length while increasing batch size and therefore represents a different memory/performance trade-off.

### Observed Non-Monotonic Behavior

During DirectML configuration testing, an unexpected behavior was observed when comparing small batch sizes and similar sequence lengths.

In particular:

```text
batch_size = 1
max_seq_len = 300
→ failed / was not viable
```

while:

```text
batch_size = 2
max_seq_len = 340
→ succeeded
```

At first glance, this appears counterintuitive because the second configuration processes both a larger batch and a longer sequence.

This observation shows that DirectML viability cannot be predicted solely from the apparent size of a configuration.

### Possible Explanation

DirectML execution depends on more than the theoretical workload represented by:

```text
batch_size × max_seq_len
```

Different tensor shapes may result in different:

* kernel execution paths;
* memory allocation patterns;
* operator implementations;
* internal DirectML behavior;
* CPU fallback behavior.

A larger tensor shape may therefore execute successfully while a theoretically smaller shape fails.

The exact internal cause of the `1 × 300` versus `2 × 340` behavior has not been established, so it should not be attributed to a specific DirectML optimization without further investigation.

### Consequence

A successful or failed configuration cannot always be used to infer the result of every apparently larger or smaller configuration.

For example, the observation:

```text
1 × 300 → FAIL
2 × 340 → PASS
```

means that the benchmark should not assume:

```text
smaller configuration = always safer
larger configuration  = always less viable
```

Configuration pruning is still useful when a configuration is clearly dominated by another one, but unusual DirectML behavior means that representative configurations should be tested empirically rather than rejected solely from their theoretical workload.

### Decision

DirectML training settings should be selected empirically rather than copied directly from the upstream MiniMind defaults.

The configuration search should:

* start with conservative values;
* progressively approach the upstream settings;
* use short training runs for validation;
* enforce a timeout for configurations that are too slow;
* stop exploring configurations that are strictly heavier than an already failing configuration;
* continue testing configurations that represent a different `batch_size` / `max_seq_len` trade-off.

Once a stable configuration is identified, the normal training workflow can be launched directly with the selected:

```text
batch_size
max_seq_len
```

values instead of repeating the benchmark.

#### DirectML batch size / sequence length validation

A compatibility benchmark was added to determine which training configurations can complete several real training steps on DirectML without crashing.

Pretraining benchmark results:

| Batch size | Sequence length | Result |
|---:|---:|---|
| 1 | 64 | PASS |
| 1 | 128 | PASS |
| 1 | 256 | PASS |
| 1 | 340 | FAIL (`0xC0000409`) |
| 2 | 128 | PASS |
| 2 | 256 | PASS |
| 2 | 340 | PASS |
| 4 | 128 | PASS |
| 4 | 256 | TIMEOUT |
| 8 | 128 | TIMEOUT |

Observations:

- DirectML stability is not strictly monotonic with batch size and sequence length.
- `1 × 340` caused a native Windows/DirectML process crash, while `2 × 340` completed successfully.
- Larger configurations such as `4 × 256` and `8 × 128` did not crash but exceeded the benchmark timeout.
- `2 × 256` was selected as a conservative configuration for validating the remaining training pipeline.
- These results apply to `train_pretrain.py` and should not be assumed to represent every trainer.