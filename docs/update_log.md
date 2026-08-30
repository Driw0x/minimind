# MiniMind — Update Log

This document provides a chronological record of the main changes made to the MiniMind DirectML fork.

For technical problems, causes, and solutions, see [`directml_issues.md`](directml_issues.md).

For durable architectural decisions and technical lessons, see [`project_memory.md`](project_memory.md).

---

# M1 — DirectML Foundation

* Configured PyTorch 2.4.1 with `torch-directml`.
* Added DirectML device detection and initialization.
* Added DirectML as an explicit MiniMind execution device.
* Integrated DirectML into the existing MiniMind device handling.
* Validated model and tensor placement on DirectML.
* Validated forward pass and loss computation.
* Validated backward pass and gradient computation.
* Validated AdamW optimizer execution.
* Validated a complete minimal training step.
* Confirmed that the known AdamW DirectML CPU fallback does not prevent training.
* Added automated DirectML compatibility tests.
* Validated installation from a clean virtual environment.

---

# M2 — MiniMind Training on DirectML

* Audited CUDA-specific code in the MiniMind training pipeline.
* Added DirectML-compatible device handling to the training workflow.
* Adapted CUDA-specific utilities where necessary.
* Added DirectML support to `eval_llm.py`.
* Validated a lightweight training configuration using:

```text
hidden_size = 128
num_hidden_layers = 2
```

* Ran minimal MiniMind pretraining on DirectML.
* Validated training loss computation.
* Validated model weight generation.
* Saved and reloaded a training checkpoint successfully.
* Validated evaluation using a DirectML-trained checkpoint.
* Validated text generation on DirectML.
* Confirmed the complete DirectML workflow:

```text
Training
   ↓
Checkpoint
   ↓
Checkpoint loading
   ↓
Evaluation
   ↓
Text generation
```

* Identified a checkpoint-loading failure caused by model architecture mismatch rather than DirectML.
* Retained the `128 / 2-layer` model as a lightweight DirectML regression configuration.

---

# M3 — Full Training Pipeline

* Extended device-aware handling across the MiniMind training utilities.
* Removed remaining assumptions that model execution always uses CUDA.
* Centralized backend and device compatibility handling in shared trainer utilities.
* Extended DirectML compatibility across the main MiniMind training stages.
* Validated pretraining.
* Validated supervised fine-tuning (SFT).
* Validated LoRA training.
* Validated DPO.
* Validated GRPO.
* Validated PPO.
* Validated Agent RL.
* Validated knowledge distillation.
* Added device-aware reward-model handling for alignment training workflows.
* Kept the trainable MiniMind model on DirectML while executing the incompatible reward model on CPU.
* Ensured reward-model inputs are moved to the correct execution device.
* Added shared handling for empty generated token sequences.
* Added safe handling for empty token selections during distillation.
* Added checkpoint compatibility tests across supported training stages.
* Validated checkpoint save/load compatibility without introducing DirectML-specific checkpoint semantics.
* Documented known unsupported and partially supported DirectML operations.
* Added deterministic test fixtures for lightweight training validation.
* Added development utilities for DirectML compatibility auditing.
* Added a PowerShell utility for sequential execution of the main training stages.
* Added a DirectML compatibility benchmark for `batch_size` and `max_seq_len`.
* Identified inconsistent initial benchmark results caused by uncontrolled physical GPU selection.
* Added explicit DirectML adapter selection for the intended dedicated GPU.
* Reran the compatibility benchmark on the explicitly selected dedicated GPU.
* Established a reference compatibility matrix on `directml:1`.
* Validated all tested configurations up to `16 × 340`.
* Validated `32 × 128`, while `32 × 256` failed.
* Confirmed that benchmark success should not be interpreted as guaranteed full-training stability.
* Confirmed that the main training pipeline is ready for longer performance and stability validation.

---

# M4 — Performance & Stability

Current milestone.

Initial validation:

* Started real-data pretraining validation on DirectML.
* Confirmed sustained pretraining execution using the real MiniMind pretraining dataset.
* Measured approximately 5.9 seconds per training step with the tested configuration.
* Observed 158780 steps per epoch, corresponding to an estimated runtime of approximately 10.9 days per epoch.
* Confirmed that a technically compatible DirectML configuration may still be impractical for full-scale training.
* Adopted bounded real-data training runs using `--max_steps` for practical DirectML validation.

Planned work includes:

* GPU memory measurement;
* CPU fallback identification and measurement;
* evaluation of CPU fallback performance impact;
* longer bounded training runs;
* stability and error-handling improvements;
* consolidation of known DirectML limitations.