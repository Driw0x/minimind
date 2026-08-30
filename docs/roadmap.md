# MiniMind DirectML Roadmap

This document tracks the development and evaluation of the MiniMind DirectML adaptation.

The project distinguishes between **functional compatibility** and **practical usability**.
A training stage may execute correctly on DirectML while still being impractical for real-world training because of performance limitations, CPU fallbacks, memory constraints, or backend overhead.

---

## M1 — DirectML Foundation ✅

Establish and validate the basic DirectML environment.

* [x] Set up a Windows + DirectML environment
* [x] Replace the CUDA-oriented PyTorch dependency with `torch-directml`
* [x] Initialize a DirectML device
* [x] Move MiniMind models and tensors to DirectML
* [x] Validate the forward pass
* [x] Validate loss computation
* [x] Validate the backward pass
* [x] Validate the AdamW optimizer step
* [x] Add DirectML compatibility tests with Pytest
* [x] Validate installation from a clean virtual environment

---

## M2 — MiniMind Training Compatibility ✅

Adapt the MiniMind training pipeline so that its training operations can execute on DirectML.

* [x] Audit CUDA-specific code in the training pipeline
* [x] Introduce DirectML-compatible device handling
* [x] Adapt mixed precision and CUDA-specific utilities where necessary
* [x] Run a minimal pretraining job on DirectML
* [x] Verify that the training loss evolves correctly
* [x] Save a model checkpoint
* [x] Reload the checkpoint successfully
* [x] Run inference using the trained checkpoint

> M2 validates **functional compatibility**, not practical training performance.

---

## M3 — Full Training Pipeline Compatibility ✅

Validate the main MiniMind training stages with DirectML.

* [x] Validate pretraining
* [x] Validate supervised fine-tuning (SFT)
* [x] Validate LoRA training
* [x] Validate additional training stages supported by MiniMind
* [x] Validate checkpoint compatibility between training stages
* [x] Document unsupported or partially supported DirectML operations
* [x] Validate device consistency across training stages

> The training pipeline is functionally compatible with DirectML, but this does not guarantee that full-scale training is practically usable.

---

## M4 — DirectML Viability & Performance Investigation 🚧

Determine whether DirectML is practically usable for MiniMind training and identify the main performance bottlenecks.

### Baseline benchmarking

* [x] Benchmark supported batch size / sequence length combinations
* [x] Identify configurations that fail because of memory or backend limitations
* [x] Confirm that synthetic compatibility benchmarks do not necessarily represent real training performance

### Real training performance

* [x] Measure real pretraining throughput
* [x] Measure iteration time and samples/tokens processed per second
* [x] Measure GPU memory usage during real training
* [x] Compare synthetic benchmark results with real dataset training
* [x] Estimate realistic training duration

### DirectML bottleneck investigation

* [ ] Identify CPU fallback operations
* [ ] Measure the performance impact of CPU fallbacks
* [ ] Investigate DirectML synchronization and execution overhead
* [ ] Identify operations responsible for unexpectedly slow training
* [ ] Determine whether the bottlenecks can be mitigated

### Practical viability

* [x] Test optimized batch size / sequence length configurations
* [x] Test longer training runs when performance permits
* [ ] Evaluate stability during sustained training
* [ ] Determine whether DirectML training is practically viable
* [ ] Document known performance limitations

> Current observations indicate that DirectML can execute the MiniMind training pipeline but may be **too slow for practical full-scale training**. M4 therefore focuses on determining whether this is caused by fixable implementation issues or fundamental DirectML/backend limitations.

---

## M5 — Project Finalization

Finalize the project according to the conclusions of the DirectML viability investigation.

### If DirectML training is practically viable

* [ ] Apply identified performance optimizations
* [ ] Finalize installation documentation
* [ ] Document supported training workflows
* [ ] Add troubleshooting documentation
* [ ] Clean up remaining CUDA assumptions
* [ ] Validate the project from a fresh clone
* [ ] Publish a reproducible DirectML-ready configuration

### If DirectML training remains impractical

* [ ] Clearly document the identified DirectML bottlenecks
* [ ] Document tested configurations and measured performance
* [ ] Define which workflows remain usable with DirectML
* [ ] Document recommended alternatives for full training
* [ ] Preserve the DirectML compatibility layer and tests
* [ ] Publish the project as a documented DirectML compatibility and feasibility study

---

## Project Goal

The objective of this project is not only to make MiniMind **run** on DirectML, but to determine whether DirectML provides a practical training backend for MiniMind on Windows.

A successful outcome can therefore be either:

1. a practically usable DirectML training pipeline, or
2. a reproducible technical evaluation demonstrating the current limitations of DirectML for MiniMind training.
