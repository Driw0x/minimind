# MiniMind Backend Roadmap

> **Historical DirectML archive — updated 2026-09-24**
>
> DirectML is no longer the active MiniMind training backend. M5 final validation
> isolated a numerical divergence specific to the tested DirectML FP16 path.
> Historical results below are retained for reproducibility and engineering
> reference; ROCm development is documented separately.

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

Finalize the DirectML investigation with full-training model-quality validation
and determine whether DirectML should remain a MiniMind training backend.

### Finalization work

-   [x] Apply identified performance optimizations
-   [x] Document supported DirectML training workflows
-   [x] Document periodic checkpoint and resume workflow
-   [x] Correct DirectML cross-entropy normalization with ignored padding
-   [x] Validate FP16 + FP32-master pretraining through two full epochs
-   [x] Reject direct FP16 AdamW through long-run checkpoint diagnostics
-   [x] Benchmark reduced diagnostic CPU synchronization
-   [x] Evaluate final DirectML checkpoints qualitatively
-   [x] Compare DirectML- and ROCm-trained checkpoints on the same evaluation set
-   [x] Re-evaluate the DirectML checkpoint under ROCm
-   [x] Compare the same DirectML checkpoint under DirectML/ROCm in FP16 and FP32
-   [x] Isolate the remaining blocker to the tested DirectML FP16 numerical path

### Final evidence

On the same `1,000` samples (`204,327` valid tokens), evaluated under ROCm:

```text
DirectML-trained: loss 6.880233, perplexity 972.8529
ROCm-trained:     loss 1.846587, perplexity   6.3382
```

Using the same DirectML-trained checkpoint and one fixed batch (`873` valid
tokens):

```text
DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML FP16 without SDPA produced non-finite logits, while DirectML and ROCm
matched to approximately `1e-6` in FP32.

### Final outcome

-   [x] DirectML execution compatibility is documented
-   [x] DirectML performance and limitations are documented
-   [x] Full-training checkpoint quality is rejected
-   [x] The remaining blocker is classified as a DirectML FP16 numerical divergence
-   [x] DirectML training is closed for this project
-   [x] DirectML code, tests and evidence are retained as a completed feasibility study

> This roadmap intentionally stops at M5. ROCm development is tracked in a
> separate roadmap outside the DirectML archive.

------------------------------------------------------------------------

## Project Goal

The DirectML project evaluated whether MiniMind could be trained reliably on an
AMD Radeon GPU under Windows through `torch-directml`.

The study is complete. DirectML could execute the training pipeline and pass
extensive compatibility, performance and smoke validation, but the final trained
checkpoints did not reach acceptable model quality. Cross-backend validation
isolated the remaining failure to the tested DirectML FP16 numerical path.

The final deliverable is therefore the preserved feasibility study, not an active
DirectML training backend.
