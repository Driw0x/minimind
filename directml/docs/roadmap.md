# MiniMind Backend Roadmap

> **Project direction update — 2026-09-21**
>
> The DirectML work is now retained as a compatibility and feasibility study,
> not as the active MiniMind training backend. The upstream official
> `pretrain_768.pth` checkpoint generates coherent text on both CPU and
> DirectML, while the locally trained DirectML checkpoints remained incoherent
> after epoch 1 and epoch 2 despite apparently healthy loss and checkpoint
> diagnostics. This isolates the unresolved problem to the custom DirectML
> training path rather than the tokenizer, dataset, checkpoint loader, or
> DirectML inference path.
>
> Because acceptable pretraining quality could not be obtained reliably with
> DirectML, the DirectML training track is **abandoned for this project**.
> Development is moving to **ROCm**. DirectML benchmarks, issues, workarounds,
> commands, and validation results below are preserved as historical technical
> evidence unless explicitly stated otherwise.

This document tracks the completed DirectML evaluation and the transition of
MiniMind training to ROCm.

The project distinguishes between **functional compatibility** and
**practical usability**. A training stage may execute correctly on
DirectML while still being impractical for real-world training because
of performance limitations, CPU fallbacks, memory constraints, or
backend overhead.

------------------------------------------------------------------------

## M1 --- DirectML Foundation ✅

Establish and validate the basic DirectML environment.

-   [x] Set up a Windows + DirectML environment
-   [x] Replace the CUDA-oriented PyTorch dependency with
    `torch-directml`
-   [x] Initialize a DirectML device
-   [x] Move MiniMind models and tensors to DirectML
-   [x] Validate the forward pass
-   [x] Validate loss computation
-   [x] Validate the backward pass
-   [x] Validate the AdamW optimizer step
-   [x] Add DirectML compatibility tests with Pytest
-   [x] Validate installation from a clean virtual environment

------------------------------------------------------------------------

## M2 --- MiniMind Training Compatibility ✅

Adapt the MiniMind training pipeline so that its training operations can
execute on DirectML.

-   [x] Audit CUDA-specific code in the training pipeline
-   [x] Introduce DirectML-compatible device handling
-   [x] Adapt mixed precision and CUDA-specific utilities where
    necessary
-   [x] Run a minimal pretraining job on DirectML
-   [x] Verify that the training loss evolves correctly
-   [x] Save a model checkpoint
-   [x] Reload the checkpoint successfully
-   [x] Run inference using the trained checkpoint

> M2 validates **functional compatibility**, not practical training
> performance.

------------------------------------------------------------------------

## M3 --- Full Training Pipeline Compatibility ✅

Validate the main MiniMind training stages with DirectML.

-   [x] Validate pretraining
-   [x] Validate supervised fine-tuning (SFT)
-   [x] Validate LoRA training
-   [x] Validate additional training stages supported by MiniMind
-   [x] Validate checkpoint compatibility between training stages
-   [x] Document unsupported or partially supported DirectML operations
-   [x] Validate device consistency across training stages

> The training pipeline is functionally compatible with DirectML, but
> this does not guarantee that full-scale training is practically
> usable.

------------------------------------------------------------------------

## M4 --- DirectML Viability & Performance Investigation ✅

Determine whether DirectML is practically usable for MiniMind training
and identify the main performance bottlenecks.

### Baseline benchmarking

-   [x] Benchmark supported batch size / sequence length combinations
-   [x] Identify configurations that fail because of memory or backend
    limitations
-   [x] Confirm that synthetic compatibility benchmarks do not
    necessarily represent real training performance

### Real training performance

-   [x] Measure real pretraining throughput
-   [x] Measure iteration time and samples/tokens processed per second
-   [x] Measure GPU memory usage during real training
-   [x] Compare synthetic benchmark results with real dataset training
-   [x] Estimate realistic training duration

### DirectML bottleneck investigation

-   [x] Identify CPU fallback operations
-   [x] Measure the performance impact of CPU fallbacks
-   [x] Investigate DirectML synchronization and execution overhead
-   [x] Identify operations responsible for unexpectedly slow training
-   [x] Determine whether the bottlenecks can be mitigated

### Practical viability

-   [x] Test optimized batch size / sequence length configurations
-   [x] Test longer training runs when performance permits
-   [x] Evaluate stability during sustained training
-   [x] Determine whether DirectML training is practically viable
-   [x] Document known performance limitations
-   [x] Validate Dense and MoE DirectML training paths
-   [x] Validate the complete cross-trainer smoke runner (`9/9`)

> M4 is complete as a historical DirectML compatibility/performance study.
> Its execution, throughput, memory, and smoke-test results remain valid as
> engineering evidence, but later full-training quality checks showed that the
> locally trained DirectML checkpoints were not acceptable. The M4 practical
> viability conclusion is therefore superseded.
>
> This M4 baseline is retained as historical validation evidence.
> M5 later replaced direct FP16 parameter updates with FP16 compute and
> FP32 master-weight optimization after long-run checkpoint-quality
> investigation.

------------------------------------------------------------------------

## M5 --- DirectML Final Evaluation ✅

Finalize the DirectML investigation with model-quality validation and decide
whether it should remain the active training backend.

### Historical finalization work

-   [x] Apply identified performance optimizations
-   [ ] Finalize installation documentation
-   [x] Document supported training workflows
-   [ ] Add troubleshooting documentation
-   [ ] Clean up remaining CUDA assumptions
-   [ ] Validate the project from a fresh clone
-   [ ] Publish a reproducible DirectML-ready configuration
-   [x] Document periodic checkpoint and resume workflow
-   [x] Correct DirectML cross-entropy normalization with ignored padding
-   [x] Validate FP16 + FP32-master pretraining through two full epochs
-   [x] Reject direct FP16 AdamW through long-run checkpoint diagnostics
-   [x] Benchmark reduced diagnostic CPU synchronization
-   [ ] Validate Log-Sync checkpoint quality

### Final outcome

-   [x] Clearly document the identified DirectML bottlenecks
-   [x] Document tested configurations and measured performance
-   [x] Define which workflows remain usable with DirectML
-   [x] Select ROCm as the next full-training backend
-   [x] Preserve the DirectML compatibility layer and tests
-   [x] Retain the DirectML work as a documented compatibility and feasibility study

------------------------------------------------------------------------

## M6 --- ROCm Migration 🚧

Move MiniMind training away from the custom DirectML path and establish a clean
ROCm baseline for the AMD GPU.

-   [x] Select ROCm as the replacement training backend
-   [x] Validate the basic ROCm precision path with FP16 autocast, FP32 weights,
    and FP32 gradients
-   [ ] Reproduce the official MiniMind pretraining configuration on ROCm
-   [ ] Validate short-run loss behavior and gradient finiteness
-   [ ] Train a bounded checkpoint and compare it with the official checkpoint
-   [ ] Run qualitative generation checks before committing to a full epoch
-   [ ] Complete full Dense pretraining only after checkpoint quality is confirmed
-   [ ] Continue to Full SFT from the validated ROCm pretraining checkpoint

> DirectML inference may still be useful, but DirectML is no longer the active
> training backend. All new training validation should target ROCm.

------------------------------------------------------------------------

## Project Goal

The project first evaluated whether DirectML could provide a practical
MiniMind training backend on Windows. That study is now complete: the backend
could execute the pipeline, but the resulting locally trained checkpoints did
not reach acceptable generation quality. The active objective is now to obtain
a correct and reproducible MiniMind training pipeline with ROCm.

The DirectML phase is considered a successful feasibility study because it
produced a reproducible record of compatibility, performance, and model-quality
limitations. The next success criterion is a ROCm-trained checkpoint whose
quantitative diagnostics and qualitative generation both match expected
MiniMind behavior.
