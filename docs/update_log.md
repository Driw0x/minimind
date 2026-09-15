# MiniMind --- Update Log

This document provides a chronological record of the main changes made
to the MiniMind DirectML fork.

For technical problems, causes, and solutions, see
[`directml_issues.md`](directml_issues.md).

For durable architectural decisions and technical lessons, see
[`project_memory.md`](project_memory.md).

------------------------------------------------------------------------

# M1 --- DirectML Foundation

-   Configured PyTorch 2.4.1 with `torch-directml`.
-   Added DirectML device detection and initialization.
-   Added DirectML as an explicit MiniMind execution device.
-   Integrated DirectML into the existing MiniMind device handling.
-   Validated model and tensor placement on DirectML.
-   Validated forward pass and loss computation.
-   Validated backward pass and gradient computation.
-   Validated AdamW optimizer execution.
-   Validated a complete minimal training step.
-   Confirmed that the known AdamW DirectML CPU fallback does not
    prevent training.
-   Added automated DirectML compatibility tests.
-   Validated installation from a clean virtual environment.

------------------------------------------------------------------------

# M2 --- MiniMind Training on DirectML

-   Audited CUDA-specific code in the MiniMind training pipeline.
-   Added DirectML-compatible device handling to the training workflow.
-   Adapted CUDA-specific utilities where necessary.
-   Added DirectML support to `eval_llm.py`.
-   Validated a lightweight training configuration using:

``` text
hidden_size = 128
num_hidden_layers = 2
```

-   Ran minimal MiniMind pretraining on DirectML.
-   Validated training loss computation.
-   Validated model weight generation.
-   Saved and reloaded a training checkpoint successfully.
-   Validated evaluation using a DirectML-trained checkpoint.
-   Validated text generation on DirectML.
-   Confirmed the complete DirectML workflow:

``` text
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

-   Identified a checkpoint-loading failure caused by model architecture
    mismatch rather than DirectML.
-   Retained the `128 / 2-layer` model as a lightweight DirectML
    regression configuration.

------------------------------------------------------------------------

# M3 --- Full Training Pipeline

-   Extended device-aware handling across the MiniMind training
    utilities.
-   Removed remaining assumptions that model execution always uses CUDA.
-   Centralized backend and device compatibility handling in shared
    trainer utilities.
-   Extended DirectML compatibility across the main MiniMind training
    stages.
-   Validated pretraining.
-   Validated supervised fine-tuning (SFT).
-   Validated LoRA training.
-   Validated DPO.
-   Validated GRPO.
-   Validated PPO.
-   Validated Agent RL.
-   Validated knowledge distillation.
-   Added device-aware reward-model handling for alignment training
    workflows.
-   Kept the trainable MiniMind model on DirectML while executing the
    incompatible reward model on CPU.
-   Ensured reward-model inputs are moved to the correct execution
    device.
-   Added shared handling for empty generated token sequences.
-   Added safe handling for empty token selections during distillation.
-   Added checkpoint compatibility tests across supported training
    stages.
-   Validated checkpoint save/load compatibility without introducing
    DirectML-specific checkpoint semantics.
-   Documented known unsupported and partially supported DirectML
    operations.
-   Added deterministic test fixtures for lightweight training
    validation.
-   Added development utilities for DirectML compatibility auditing.
-   Added a PowerShell utility for sequential execution of the main
    training stages.
-   Added a DirectML compatibility benchmark for `batch_size` and
    `max_seq_len`.
-   Identified inconsistent initial benchmark results caused by
    uncontrolled physical GPU selection.
-   Added explicit DirectML adapter selection for the intended dedicated
    GPU.
-   Reran the compatibility benchmark on the explicitly selected
    dedicated GPU.
-   Established a reference compatibility matrix on `directml:1`.
-   Validated all tested configurations up to `16 × 340`.
-   Validated `32 × 128`, while `32 × 256` failed.
-   Confirmed that benchmark success should not be interpreted as
    guaranteed full-training stability.
-   Confirmed that the main training pipeline is ready for longer
    performance and stability validation.

------------------------------------------------------------------------

# M4 --- Performance & Stability

-   Started real-data pretraining performance and stability validation
    on DirectML.
-   Added bounded real-data training runs using `--max_steps` to make
    DirectML benchmarking practical.
-   Confirmed sustained pretraining execution using the real MiniMind
    pretraining dataset.
-   Benchmarked training on the explicitly selected `directml:1`
    adapter.
-   Tested real training with the full `max_seq_len = 340`
    configuration.
-   Validated real-data execution with `batch_size = 8` and
    `max_seq_len = 340`.
-   Validated a 100-step bounded real-training run with `batch_size = 8`
    and `max_seq_len = 340`.
-   Confirmed that `32 × 340` fails because of insufficient GPU memory
    during real training.
-   Confirmed that `16 × 340` can complete an accumulation cycle but
    later fails because of insufficient GPU memory.
-   Established `8 × 340` with gradient accumulation `8` as the current
    real-training reference configuration.
-   Measured approximately `11.62 GB` peak dedicated GPU memory during
    the reference benchmark.
-   Used gradient accumulation to preserve a larger effective batch size
    while keeping the physical batch size compatible with DirectML
    memory constraints.
-   Added warmup steps before performance measurement to avoid including
    initialization overhead in steady-state timing.
-   Initial real-training measurement reported approximately
    `5.280 s/iteration`, `1.52 samples/s`, and
    `304.81 effective tokens/s`.
-   Investigated the unexpectedly slow real-training measurement.
-   Identified unnecessary DirectML-to-CPU synchronization in the
    training measurement path.
-   Removed unnecessary synchronization from the critical training and
    measurement path.
-   Reran the same `8 × 340` reference workload for 100 steps after the
    correction.
-   Measured approximately `0.482 s/iteration`, `16.61 samples/s`, and
    `3341.37 effective tokens/s`.
-   Measured approximately `5647.46 padded tokens/s`.
-   Reduced measured iteration time by approximately `10.95×` compared
    with the initial benchmark.
-   Reduced the estimated epoch duration from approximately `9.70 days`
    to `21.24 hours` (`0.89 days`).
-   Confirmed that the performance improvement was obtained without
    reducing the model architecture, batch size, sequence length, or
    gradient accumulation configuration.
-   Confirmed that the AdamW `aten::lerp.Scalar_out` CPU fallback
    remains present but does not explain the full initial slowdown.
-   Established bounded step-based runs as the preferred method for
    DirectML performance and stability validation.
-   Compared the synthetic DirectML compatibility benchmark with real
    MiniMind training behavior.
-   Confirmed that configurations passing the synthetic compatibility
    benchmark are not guaranteed to remain practical or stable with real
    training data.
-   Identified `batch_size` and `max_seq_len` as interacting memory
    constraints rather than independent compatibility limits.
-   Confirmed that smaller physical batches provide substantially more
    headroom for full sequence lengths.
-   Identified DirectML CPU fallback during AdamW execution for
    unsupported operations such as `aten::lerp.Scalar_out`.
-   Confirmed that the fallback does not prevent training but introduces
    a potential performance bottleneck.
-   Distinguished DirectML device compatibility from actual end-to-end
    training performance.
-   Confirmed that explicit physical adapter selection remains necessary
    to obtain reproducible benchmark results on multi-GPU systems.
-   Documented the difference between synthetic compatibility results
    and real-data training observations.
-   Consolidated DirectML benchmark methodology and results for future
    regression testing.

Current conclusions:

``` text
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

The current practical reference configuration for real-data DirectML
validation is:

``` text
Device:              directml:1
Batch size:          8
Max sequence length: 340
Gradient accumulation: 8
```

This configuration is the current **validation and benchmarking
baseline**.

Final M4 validation additionally:

-   introduced a stable DirectML FP16 path using static loss scale
    `1024` and AdamW epsilon `1e-4`;
-   confirmed that the default AdamW epsilon `1e-8` becomes non-finite
    after the first optimizer update in the tested pure-FP16 path;
-   validated the complete bounded FP16 compatibility matrix through the
    first optimizer-update boundary;
-   completed a sustained `1000`-step pretraining run with finite losses
    throughout;
-   completed `125` optimizer updates during that run;
-   measured approximately `7,769.62 MB` peak dedicated VRAM in FP16;
-   reduced peak dedicated VRAM by approximately `33.15%` compared with
    the earlier FP32 measurement;
-   propagated DirectML FP16 handling and global `--max_steps` support
    across Pretrain, Full SFT, LoRA, Distillation, GRPO, Agent RL, and
    PPO;
-   added `tests/test_all_trainers.py` for explicit sequential DirectML
    trainer smoke validation;
-   validated all trainers included in that smoke suite successfully;
-   concluded that the tested DirectML configuration is practically
    viable.
-   extended the smoke workflow with Dense and MoE pretraining / Full
    SFT coverage;
-   identified unsupported scatter behavior in the upstream sparse MoE
    routing path on DirectML;
-   added a DirectML-specific scatter-free MoE compatibility path while
    preserving the original CPU/CUDA sparse routing;
-   validated MoE Pretrain, MoE Full SFT, and Dense-student/MoE-teacher
    Distillation;
-   fixed Agent RL DataLoader multiprocessing on Windows by moving
    `collate_fn` to module scope;
-   isolated Agent FP16 NaNs to policy/reference full-sequence
    recomputation using the attention mask;
-   avoided the problematic attention-mask path for the validated
    right-padded Agent recomputation and retained FP32 for numerically
    sensitive RL calculations;
-   completed the final DirectML trainer smoke suite successfully:

``` text
All trainer smoke tests passed
Passed: 9/9
```

M4 is complete. The project now moves to **M5 --- Project
Finalization**.

------------------------------------------------------------------------

# M5 --- Project Finalization

-   Diagnosed a long-run Dense pretraining collapse in which the
    previous pure-FP16 optimizer path remained finite but converged
    toward token self-copying.
-   Added FP32 master weights for Dense DirectML FP16 pretraining and
    validated the corrected path through global step `1100` with mean
    diagnostic loss `6.7506`, Top-1 accuracy `6.30%`, and repeat rate
    `0.89%`.

-   Added trainer-specific batch-size, sequence-length, and accumulation
    settings to the sequential training workflow, while keeping the M4
    `8 × 340` configuration documented as the sustained DirectML
    pretraining baseline.
-   Added automatic resume checkpoint refresh every 100 iterations and
    documented interrupted-training recovery with `--from_resume 1`.
-   Identified incorrect DirectML mean reduction for
    `cross_entropy(..., ignore_index=-100)` on padded causal-LM batches.
-   Replaced implicit mean reduction with explicit valid-token loss
    normalization.
-   Moved the targeted DirectML loss-validation utilities from
    `scripts/` to `tests/`.
-   Validated a fresh corrected FP16 + FP32-master pretraining run at
    step `100` with loss `7.3925`, Top-1 accuracy `3.26%`, and repeat
    rate `3.26%`.
-   Validated the same run at step `1000` with loss `6.7785`, Top-1
    accuracy `6.76%`, and repeat rate `0.76%`.

-   Restarted Dense pretraining from scratch with the corrected FP16
    compute + FP32 master-weight + valid-token loss path and upstream
    pretraining parameters.
-   Completed the first full Dense pretraining epoch successfully with
    physical `batch_size = 32`.
-   Validated the epoch-1 checkpoint on samples `0–255` with
    token-weighted global loss `6.1017`, Top-1 accuracy `15.07%`,
    Top-1 repeat rate `0.83%`, and mean entropy `5.3160`.
-   Confirmed exact agreement between model loss and independent manual
    token-normalized cross-entropy (`6.48799419`, difference `0` on the
    reference batch).
-   Identified the previously reported aggregate diagnostic loss around
    `12–13` as a `diagnose_pretrain.py` aggregation issue rather than a
    training divergence.
-   Began correcting `diagnose_pretrain.py` to use the real batched
    training-loss path and token-weighted aggregation.

-   Re-tested the DirectML causal-LM loss with identical logits using
    `sum / valid`, per-token `none → FP32 mean`, and CPU FP32 reference
    reductions.
-   Confirmed that the initial `reduction="sum" / valid_tokens`
    workaround is also incorrect on the tested DirectML FP16 path.
-   Replaced it with `reduction="none"` followed by FP32 averaging over
    valid non-padding tokens.
-   Validated the final batch-32 loss path with model loss `6.30274916`
    versus CPU FP32 reference `6.30530691` (difference approximately
    `0.00256`).
-   Reclassified the completed two-epoch / earlier full-epoch
    pretraining checkpoints as intermediate investigation artifacts
    because their backward pass used the superseded `sum / valid`
    reduction.
-   Final Dense pretraining must restart from scratch with FP32 master
    weights and the final per-token FP32-valid-mean cross-entropy path.
-   Validated the final DirectML FP16 + FP32-master + per-token
    cross-entropy path through step `1200`.
-   Training-path and manual token losses remained within approximately
    `0.003` across two independent 256-sample evaluation regions.
-   At step `1200`, train-path loss reached `6.3752` and `6.3568`,
    Top-1 accuracy reached `8.66%` and `8.91%`, and Top-1 repeat rate
    remained limited to `1.96%` and `2.08%`.
-   Confirmed no recurrence of the historical self-copying collapse;
    no recurrence was observed through the subsequent full epoch.
-   Completed epoch 1 of the final Dense DirectML pretraining run using
    FP16 compute, FP32 master weights, and per-token FP32-valid-mean
    cross-entropy.
-   On samples `0–255`, validated train loss `6.0612`, manual loss
    `6.0637`, Top-1 accuracy `13.71%`, repeat rate `0.81%`, and mean
    entropy `5.3487`.
-   On samples `100000–100255`, validated train loss `6.0177`, manual
    loss `6.0201`, Top-1 accuracy `13.80%`, repeat rate `0.71%`, and mean
    entropy `5.3436`.
-   Confirmed train/manual loss agreement within approximately `0.0025`
    after a complete epoch and no recurrence of the historical
    self-copying collapse.
-   Continued the same validated run into epoch 2; final two-epoch
    validation remains pending.

