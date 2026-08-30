# MiniMind — Project Memory

## Purpose

This repository is a fork of MiniMind adapted to experiment with and train small language models on Windows using DirectML.

The main goals are to:

* understand the complete training pipeline of a small language model;
* experiment with MiniMind locally;
* add DirectML support for GPU acceleration on Windows;
* keep the fork as close as possible to the original MiniMind project;
* document important technical changes, limitations, and decisions.

---

## Current Status

DirectML support has been successfully validated on a minimal MiniMind training step.

### Working

* PyTorch with `torch-directml`
* DirectML device detection
* Model execution on DirectML
* Input tensors on DirectML
* Forward pass
* Loss computation
* Backward pass
* AdamW optimizer step
* `zero_grad()`
* Complete minimal training step
* Initial `--device directml` support

### Current Focus

Integrate and validate DirectML support in the actual MiniMind training and evaluation scripts while keeping modifications to the original project minimal.

---

## Environment

Current development environment:

```text
OS: Windows
PyTorch: 2.4.1+cpu
GPU backend: DirectML
DirectML package: torch-directml
PyTorch device: privateuseone:0
```

DirectML devices are created with:

```python
import torch_directml

device = torch_directml.device()
```

Although the project refers to the backend as `directml`, PyTorch exposes the device as:

```text
privateuseone:0
```

This is expected behavior.

---

## DirectML Validation

Before modifying the full MiniMind training pipeline, a minimal training test was used to verify DirectML compatibility.

The following operations were successfully validated:

```text
Model → DirectML
Inputs → DirectML
Forward
Loss
Backward
AdamW step
zero_grad
```

Observed result:

```text
PyTorch: 2.4.1+cpu
Device: privateuseone:0
Input device: privateuseone:0
Model device: privateuseone:0
Logits shape: torch.Size([1, 16, 6400])
Logits device: privateuseone:0
Loss: 8.839616775512695

DirectML forward pass: OK
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

This confirms that the fundamental PyTorch operations required to train MiniMind can execute through DirectML.

---

## DirectML Integration

The fork introduces explicit DirectML device selection through:

```text
--device directml
```

Example:

```powershell
python eval_llm.py `
  --device directml `
  --weight pretrain_128 `
  --hidden_size 128 `
  --num_hidden_layers 2 `
  --max_new_tokens 32
```

The expected device flow is:

```text
--device directml
        ↓
torch_directml.device()
        ↓
privateuseone:0
        ↓
model.to(device)
        ↓
tensor.to(device)
```

DirectML-specific changes should remain limited to device selection and compatibility handling whenever possible.

---

## Test Configuration

The initial DirectML tests use a deliberately small MiniMind configuration:

```text
hidden_size: 128
num_hidden_layers: 2
weight: pretrain_128
```

Evaluation tests use:

```text
max_new_tokens: 32
```

This small configuration is intended for development and validation rather than model quality.

The priority is:

```text
Correctness
    ↓
Full pipeline validation
    ↓
Stability
    ↓
Performance
    ↓
Larger models
```

---

## Known DirectML Limitations

### AdamW CPU Fallback

DirectML does not currently implement every PyTorch operator used by AdamW.

The following operator has been observed:

```text
aten::lerp.Scalar_out
```

PyTorch reports:

```text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

The fallback occurs during the optimizer step.

### Current Impact

The fallback:

* does not crash training;
* does not prevent the optimizer step;
* allows the complete training step to finish;
* may reduce training performance.

CPU fallbacks are currently acceptable as long as they do not affect correctness or prevent training.

Performance optimization will be considered after the complete pipeline has been validated.

---

## Checkpoint Compatibility

MiniMind checkpoints must be loaded with a model architecture compatible with the configuration used during training.

For example, a checkpoint created using:

```text
hidden_size = 128
num_hidden_layers = 2
```

must be loaded using the corresponding architecture.

Checkpoint loading uses strict parameter matching:

```python
model.load_state_dict(
    torch.load(ckp, map_location="cpu"),
    strict=True
)
```

A mismatch can result in:

* missing parameters;
* unexpected parameters;
* tensor shape mismatches.

When checkpoint loading fails, the model configuration should therefore be checked before assuming the issue comes from DirectML.

The original MiniMind checkpoint/output organization is preserved.

---

## Technical Decisions

### DirectML as the Windows GPU Backend

DirectML is used to provide GPU acceleration on Windows without requiring CUDA.

---

### Keep the Fork Close to Upstream

Changes to MiniMind should remain minimal.

DirectML support should preferably be implemented through generic device handling rather than duplicating training or evaluation logic.

For example:

```python
device = ...
model.to(device)
```

is preferred over creating separate DirectML-specific versions of the training pipeline.

---

### Preserve Existing Project Structure

The original MiniMind directory organization and checkpoint handling should remain unchanged unless a modification becomes technically necessary.

DirectML support should not introduce unrelated structural changes.

---

### CPU Fallbacks Are Temporarily Acceptable

Unsupported DirectML operations may fall back to CPU.

These fallbacks should be documented and monitored, but they do not need to be eliminated before the full pipeline works.

---

### Small Models First

Development and debugging should use small configurations before attempting larger MiniMind models.

This makes DirectML compatibility issues faster and easier to isolate.

---

## Current Pipeline

The pipeline currently being validated is:

```text
Dataset
   ↓
Tokenizer
   ↓
MiniMind Model
   ↓
DirectML
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Optimizer
   ↓
Checkpoint
   ↓
Evaluation
   ↓
Text Generation
```

The fundamental DirectML operations have been validated independently.

The next goal is to confirm that they work correctly inside the actual MiniMind pipeline.

---

## Next Steps

* Continue adapting MiniMind scripts to accept DirectML where necessary.
* Validate DirectML in the actual pretraining script.
* Run a small real pretraining session.
* Verify that training completes correctly.
* Verify checkpoint generation using the existing MiniMind structure.
* Load the generated checkpoint with `eval_llm.py`.
* Validate complete text generation on DirectML.
* Record any additional unsupported DirectML operators.
* Evaluate the performance impact of CPU fallbacks.
* Progressively test larger MiniMind configurations.

---

## Update Log

### DirectML Initial Validation

* Configured PyTorch with `torch-directml`.
* Confirmed DirectML device detection.
* Confirmed that DirectML appears as `privateuseone:0`.
* Validated model execution on DirectML.
* Validated forward pass.
* Validated loss computation.
* Validated backward pass.
* Validated AdamW optimizer step.
* Validated `zero_grad()`.
* Validated a complete minimal training step.
* Identified the `aten::lerp.Scalar_out` CPU fallback.

### MiniMind Integration

* Added initial DirectML device handling.
* Added support for selecting `directml` as a device.
* Started validating the existing MiniMind evaluation pipeline with DirectML.
* Tested a small 128 hidden-size / 2-layer configuration.
* Identified checkpoint/model architecture compatibility as an important consideration.
