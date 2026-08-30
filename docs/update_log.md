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

* Started real-data pretraining performance and stability validation on DirectML.
* Added bounded real-data training runs using `--max_steps` to make DirectML benchmarking practical.
* Confirmed sustained pretraining execution using the real MiniMind pretraining dataset.
* Benchmarked training on the explicitly selected `directml:1` adapter.
* Tested real training with the full `max_seq_len = 340` configuration.
* Validated real-data execution with `batch_size = 8` and `max_seq_len = 340`.
* Used gradient accumulation to preserve a larger effective batch size while keeping the physical batch size compatible with DirectML memory constraints.
* Added warmup steps before performance measurement to avoid including initialization overhead in steady-state timing.
* Observed approximately 5–6 seconds per training step after warmup with the tested configuration.
* Confirmed that full-epoch training remains impractical at the measured DirectML throughput.
* Established bounded step-based runs as the preferred method for DirectML performance and stability validation.
* Compared the synthetic DirectML compatibility benchmark with real MiniMind training behavior.
* Confirmed that configurations passing the synthetic compatibility benchmark are not guaranteed to remain practical or stable with real training data.
* Identified `batch_size` and `max_seq_len` as interacting memory constraints rather than independent compatibility limits.
* Confirmed that smaller physical batches provide substantially more headroom for full sequence lengths.
* Identified DirectML CPU fallback during AdamW execution for unsupported operations such as `aten::lerp.Scalar_out`.
* Confirmed that the fallback does not prevent training but introduces a potential performance bottleneck.
* Distinguished DirectML device compatibility from actual end-to-end training performance.
* Confirmed that explicit physical adapter selection remains necessary to obtain reproducible benchmark results on multi-GPU systems.
* Documented the difference between synthetic compatibility results and real-data training observations.
* Consolidated DirectML benchmark methodology and results for future regression testing.

Current conclusions:

```text
DirectML compatibility
        ↓
Synthetic benchmark
        ↓
Real-data bounded training
        ↓
Performance / fallback analysis
        ↓
Practical configuration
```

The current practical reference configuration for real-data DirectML validation is:

```text
Device:              directml:1
Batch size:          8
Max sequence length: 340
Gradient accumulation: 8
```

This configuration is intended as a **validation and benchmarking baseline**, not as evidence that full-scale MiniMind training on DirectML is computationally practical.
