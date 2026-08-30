# MiniMind — Project Memory

This document provides a concise technical memory of the MiniMind DirectML adaptation.

Detailed technical issues are documented in [`directml_issues.md`](directml_issues.md).

Benchmark results and performance experiments are documented in [`directml_benchmarks.md`](directml_benchmarks.md).

Development progress is tracked separately in [`update_log.md`](update_log.md).

---

# M1 — DirectML Foundation

DirectML support was introduced as an explicit MiniMind execution backend.

The main observations were:

* DirectML is exposed internally by PyTorch as `privateuseone:0`.
* Standard `.to(device)` operations can be reused for DirectML.
* Forward, backward, loss computation, and optimizer steps were validated.
* AdamW uses an unsupported DirectML operation (`aten::lerp.Scalar_out`) that automatically falls back to CPU.
* The fallback does not prevent training.

### Decision

DirectML support should remain integrated into the existing MiniMind execution pipeline rather than introducing separate DirectML-specific implementations.

---

# M2 — Pretraining and Evaluation Integration

DirectML support was validated through a complete MiniMind workflow rather than only isolated PyTorch operations.

A lightweight validation model was used:

```text
hidden_size = 128
num_hidden_layers = 2
```

The following pipeline was successfully validated:

```text
Training
   ↓
DirectML
   ↓
Checkpoint
   ↓
Checkpoint loading
   ↓
Evaluation
   ↓
Text generation
```

A checkpoint-loading failure was also identified as an architecture mismatch rather than a DirectML issue.

### Decision

The `128 / 2-layer` configuration is kept as a lightweight DirectML regression configuration.

Checkpoint architecture compatibility must be checked independently from backend compatibility.

---

# M3 — Training Pipeline Compatibility

DirectML support was extended to the remaining MiniMind training pipeline.

Several compatibility issues were identified:

* the GRPO reward model cannot reliably execute on DirectML and remains on CPU;
* device handling was centralized in shared trainer utilities;
* empty generated token sequences are protected by shared generation utilities;
* checkpoint compatibility across training stages was validated;
* practical `batch_size` and `max_seq_len` limits require hardware-specific benchmarking;
* the development machine exposes both an integrated and dedicated GPU, requiring explicit DirectML adapter selection.

### Decision

Backend compatibility belongs in shared trainer infrastructure rather than individual training algorithms.

Different components may intentionally use different devices when required:

```text
Trainable model → DirectML
Reward model    → CPU
```

DirectML must not modify MiniMind checkpoint semantics.

Performance limits must be determined empirically on the explicitly selected physical GPU.

---

# General Decisions

## Keep DirectML Changes Close to Upstream

DirectML support should modify as little of the original MiniMind training logic as possible.

Shared compatibility utilities are preferred over DirectML-specific copies of existing pipelines.

## Separate Backend Compatibility From Training Logic

Training algorithms should remain backend-independent whenever possible.

Device resolution, DirectML initialization, model placement, and compatibility exceptions belong in shared infrastructure.

## Accept CPU Fallbacks When Necessary

CPU execution is acceptable when required for compatibility and when correctness is preserved.

Current examples include:

```text
AdamW unsupported operation → CPU fallback
Reward model → CPU
```

## Preserve MiniMind Checkpoint Semantics

DirectML is an execution backend and must not introduce a separate checkpoint format.

Checkpoint compatibility depends on model architecture and training stage, not the execution device.
