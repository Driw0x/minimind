# MiniMind — Project Memory

This document records technical problems encountered while adapting MiniMind to DirectML, their causes, and the solutions or decisions applied.

---

## DirectML Device Displayed as `privateuseone:0`

### Problem

After initializing DirectML, PyTorch reports the device as:

```text id="ivzjnw"
privateuseone:0
```

instead of a device name such as `directml:0`.

This initially made it unclear whether the model was actually running on DirectML.

### Cause

`torch-directml` integrates DirectML into PyTorch through the `PrivateUse1` backend.

Therefore, a DirectML device is internally exposed by PyTorch as:

```text id="9csh9d"
privateuseone:0
```

This is expected behavior and does not mean that execution is falling back to CPU.

### Solution

No code change is required.

Device placement can be verified by checking the model, inputs, and outputs:

```python id="v5f6ju"
print(next(model.parameters()).device)
print(input_ids.device)
print(logits.device)
```

Expected output:

```text id="gk1gnv"
privateuseone:0
```

---

## AdamW CPU Fallback

### Problem

During the AdamW optimizer step, PyTorch reports:

```text id="6uw86k"
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

The affected operator is:

```text id="lqvf6g"
aten::lerp.Scalar_out
```

### Cause

DirectML does not currently implement every PyTorch operation used internally by `torch.optim.AdamW`.

When an unsupported operation is encountered, `torch-directml` automatically executes it on the CPU when a fallback implementation is available.

### Solution

No workaround is currently required.

The complete optimizer step was tested successfully despite the fallback:

```text id="6jwg3p"
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

The fallback is therefore accepted for now.

It should be reconsidered only if it becomes a significant training performance bottleneck.

---

## Checkpoint and Model Architecture Mismatch

### Problem

`eval_llm.py` failed while loading a checkpoint using:

```python id="nqzxox"
model.load_state_dict(
    torch.load(ckp, map_location="cpu"),
    strict=True
)
```

The error indicated incompatible model parameters.

### Cause

The model instantiated during evaluation did not initially match the architecture used to create the checkpoint.

MiniMind checkpoints depend on parameters such as:

```text id="6yxzqs"
hidden_size
num_hidden_layers
vocab_size
```

With `strict=True`, PyTorch requires the checkpoint and instantiated model to have compatible parameters and tensor dimensions.

The issue was therefore caused by model configuration rather than DirectML.

### Solution

Use the same model architecture during evaluation as during training.

For the validation model:

```text id="yxic44"
hidden_size = 128
num_hidden_layers = 2
```

the corresponding evaluation command must include:

```powershell id="hnfs5k"
--hidden_size 128 `
--num_hidden_layers 2
```

The relationship to preserve is:

```text id="rkn36l"
Training configuration
        ↓
Checkpoint
        ↓
Evaluation configuration
```

---

## DirectML Not Available as an Original Device Option

### Problem

The original MiniMind scripts did not provide `directml` as an explicit device option.

This prevented DirectML from being selected in the same way as the existing supported devices.

### Cause

The upstream MiniMind project was not designed with `torch-directml` as one of its execution backends.

DirectML also requires initialization through:

```python id="62pbkl"
torch_directml.device()
```

rather than the standard CUDA device path.

### Solution

Add:

```text id="69kicv"
directml
```

as an explicit device option.

The device is initialized when DirectML is selected:

```python id="uf3x1d"
import torch_directml

device = torch_directml.device()
```

The resulting device can then be used through the existing PyTorch device-independent operations:

```python id="4as6fb"
model.to(device)
tensor.to(device)
```

This avoids creating separate DirectML-specific training or evaluation pipelines.

---

## DirectML End-to-End Validation

### Problem

Successful isolated forward and backward passes were not enough to guarantee that the actual MiniMind workflow would function correctly with DirectML.

Potential failures could still occur during training, checkpoint generation, checkpoint loading, evaluation, or generation.

### Cause

DirectML compatibility depends on all PyTorch operations used throughout the complete workflow, not only the model's forward and backward operations.

### Solution

A small MiniMind configuration was used to validate the complete workflow:

```text id="prpkv2"
hidden_size = 128
num_hidden_layers = 2
```

The following pipeline was successfully tested:

```text id="f0bzd0"
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

This configuration should remain useful as a lightweight regression test when future DirectML changes are introduced.
