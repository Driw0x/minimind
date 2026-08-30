# MiniMind — Project Memory

This document provides a concise technical memory of the MiniMind DirectML adaptation.

It records the main architectural decisions, technical lessons, and validation references that should remain useful throughout the project.

Detailed technical issues are documented in [`directml_issues.md`](directml_issues.md).

Benchmark results and performance experiments are documented in [`directml_benchmarks.md`](directml_benchmarks.md).

Current limitations and fallbacks are documented in [`directml_limitations.md`](directml_limitations.md).

Development progress is tracked separately in [`update_log.md`](update_log.md).

---

# Architecture Decisions

## Keep DirectML Changes Close to Upstream

DirectML support should modify as little of the original MiniMind training logic as possible.

The existing MiniMind execution pipeline should be reused whenever possible:

```text
Model
  ↓
Standard PyTorch operations
  ↓
Resolved execution device
  ↓
CPU / CUDA / DirectML
```

DirectML-specific copies of existing training pipelines should be avoided.

### Decision

Shared compatibility utilities are preferred over separate DirectML implementations.

---

## Separate Backend Compatibility From Training Logic

Training algorithms should remain backend-independent whenever possible.

Backend-specific responsibilities include:

* device resolution;
* DirectML initialization;
* model placement;
* component-specific device requirements;
* backend compatibility rules.

These responsibilities belong in shared trainer infrastructure rather than individual training algorithms.

### Decision

Individual trainers should focus on their training algorithm.

Backend compatibility should be centralized in shared trainer utilities.

---

## Allow Component-Specific Device Placement

Not every component of a training workflow must execute on the same device.

Some models or operations may require CPU execution when DirectML does not support their complete inference path.

The current alignment-training configuration may therefore use:

```text
Trainable model → DirectML
Reward model    → CPU
```

Data is transferred between devices when required.

### Decision

Correctness and stability take priority over forcing every component onto DirectML.

---

## Preserve MiniMind Checkpoint Semantics

DirectML is an execution backend and must not introduce a separate checkpoint format.

Checkpoint compatibility depends on factors such as:

```text
Model architecture
Training stage
State dictionary structure
```

and not on whether the model was executed using CPU, CUDA, or DirectML.

### Decision

Existing MiniMind checkpoint semantics must remain unchanged.

Backend compatibility and checkpoint compatibility should be validated independently.

---

# Important Technical Lessons

## DirectML Device Representation

`torch-directml` integrates DirectML through PyTorch's `PrivateUse1` backend.

A DirectML device therefore appears internally as:

    privateuseone:<index>

The adapter index is preserved between the user-facing DirectML device and PyTorch's internal representation.

For example:

    directml:0 → privateuseone:0
    directml:1 → privateuseone:1

This is expected behavior.

The user-facing MiniMind device options remain:

    directml
    directml:<index>

### Lesson

Do not treat `privateuseone` as a separate device backend.

It is PyTorch's internal representation of the DirectML device selected through `torch-directml`.

When multiple GPUs are available, the device index is significant and should be preserved when documenting or comparing runs.

---

## CPU Fallbacks Are Acceptable When Necessary

DirectML does not support every PyTorch operation used by MiniMind and its dependencies.

Some unsupported operations can automatically fall back to CPU.

Other components may need to remain explicitly on CPU.

Examples currently include:

```text
AdamW unsupported operation → automatic CPU fallback
Reward model                → CPU execution
```

### Lesson

A CPU fallback is not automatically a compatibility failure.

Fallbacks are acceptable when correctness is preserved, but their performance impact must be evaluated separately.

Current fallbacks are tracked in [`directml_limitations.md`](directml_limitations.md).

---

## Physical GPU Selection Must Be Explicit

The development machine exposes multiple graphics adapters, including an integrated GPU and a dedicated GPU.

Both may be visible to DirectML.

Therefore:

```text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

Implicit DirectML device selection can make benchmark results difficult to interpret because separate runs may execute on different physical GPUs.

### Lesson

The intended physical DirectML adapter should be explicitly selected before training or benchmarking.

Reference performance measurements must only be compared when they use the same physical GPU.

---

## Benchmark Compatibility Is Not Full-Training Stability

Short compatibility benchmarks are useful for identifying configurations that fail immediately.

However:

```text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

A full training run executes for much longer and may encounter different memory, performance, or backend conditions.

### Lesson

Benchmark results should be treated as compatibility indicators rather than performance recommendations.

Final training configurations must be validated with actual real-data training runs.

Real pretraining confirmed that a configuration can execute successfully on DirectML while still being impractical for full-scale training because of performance or memory constraints.

DirectML validation should therefore use bounded real-data training runs with `--max_steps` when full-dataset execution is not practical.

Detailed performance measurements belong in [`directml_benchmarks.md`](directml_benchmarks.md).

---

# Regression Configuration

A lightweight MiniMind model is used for fast DirectML compatibility validation:

```text
hidden_size = 128
num_hidden_layers = 2
```

This configuration has been used to validate the complete workflow:

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

### Decision

The `128 / 2-layer` configuration is retained as a lightweight DirectML regression configuration.

It is intended for compatibility validation rather than representative model performance testing.

---

# Validation Principles

## Test Backend Compatibility Independently

DirectML execution should be validated independently from unrelated model or checkpoint problems.

A failure should not automatically be attributed to DirectML.

Relevant checks include:

```text
Device placement
Model architecture
Checkpoint compatibility
Dataset validity
Backend operation support
```

---

## Prefer Real Training Validation

Synthetic and short-running tests are useful for development and regression testing.

They do not replace real training validation.

The validation strategy therefore combines:

```text
Unit tests
    +
Smoke tests
    +
Compatibility benchmarks
    +
Real training runs
```

Each layer answers a different compatibility question.

---

# Documentation Responsibilities

Project documentation is intentionally separated by purpose:

```text
roadmap.md
    → planned milestones and remaining work

update_log.md
    → chronological development history

project_memory.md
    → durable architectural decisions and lessons

directml_issues.md
    → problems, causes, solutions, and decisions

directml_limitations.md
    → current unsupported operations and fallbacks

directml_benchmarks.md
    → experimental compatibility and performance results

development-tools.md
    → development and validation utilities

directml_audit.md
    → automatically generated compatibility audit
```

This separation should be preserved as the project evolves.
