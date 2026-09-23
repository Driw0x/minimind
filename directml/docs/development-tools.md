# Development Tools

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

This document describes the development utilities that were added to support
and validate the historical DirectML adaptation of MiniMind. They remain useful
for regression, diagnosis, and documenting the DirectML feasibility study, but
new training work now targets ROCm.

## DirectML Compatibility Audit

The DirectML audit scans the project for code that may rely on
CUDA-specific behavior.

Run from the project root:

``` bash
python scripts/audit_directml.py
```

The generated report is written to:

``` text
docs/directml_audit.md
```

The audit detects potential compatibility concerns including:

-   CUDA-specific references;
-   device placement assumptions;
-   CUDA utilities;
-   automatic mixed precision (AMP);
-   distributed training code.

The report is intended as a static analysis aid. A reported occurrence
does not necessarily represent an incompatibility and must be reviewed
manually.

------------------------------------------------------------------------

## Test Fixture Generation

Small deterministic datasets are used by the automated test suite to
validate MiniMind training stages without requiring the full training
datasets.

Generate the fixtures with:

``` bash
python scripts/generate_test_fixtures.py
```

The generated datasets are stored in:

``` text
tests/fixtures/
```

They provide lightweight inputs for testing training workflows such as:

-   pretraining;
-   supervised fine-tuning;
-   DPO;
-   Agent RL;
-   other supported training stages.

These fixtures are intended for compatibility and smoke testing, not for
actual model training.

------------------------------------------------------------------------

## Sequential Training

The `train_all.ps1` PowerShell script runs the main MiniMind training
stages sequentially on Windows.

Run it from the project root:

``` powershell
.\scripts\train_all.ps1
```

The script executes each training stage one after another. A new stage
starts only after the previous one has completed successfully.

This avoids running multiple DirectML training processes in parallel,
which could increase GPU memory usage and reduce stability.

If a training stage fails, the script stops immediately instead of
continuing with the remaining stages.

The script keeps the standard MiniMind output structure:

``` text
out/            # Generated model weights
checkpoints/    # Training resume checkpoints
```

This utility was intended to simplify sequential validation and execution
of the DirectML training pipeline on Windows. It is now retained for historical
reproduction and regression only; it is not the recommended full-training path.

`train_all.ps1` uses trainer-specific batch-size, sequence-length, and
gradient-accumulation settings instead of applying one shared
configuration to every stage. Checkpoint intervals are trainer-specific and are documented in
`training_commands.md`; short validation runs may intentionally override
them.

See [`training_commands.md`](training_commands.md) for the complete
per-trainer configuration table and the `--from_resume 1` recovery
commands.

------------------------------------------------------------------------

## Real Training Benchmark

The `benchmark_training.py` utility measures real MiniMind pretraining
performance using the actual pretraining dataset instead of a synthetic
compatibility workload.

Run it from the project root:

``` bash
python scripts/benchmark_training.py
```

A typical DirectML reference run can be started with:

``` powershell
python scripts/benchmark_training.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --warmup_steps 8 `
    --steps 100 `
    --model_dtype float16
```

The benchmark performs the main operations of a real pretraining step:

-   dataset loading;
-   forward pass;
-   loss computation;
-   backward pass;
-   gradient accumulation;
-   gradient clipping;
-   AdamW optimizer steps.

Warmup iterations are excluded from the measured results so
initialization overhead does not distort steady-state performance.

The utility reports metrics including:

``` text
Average iteration time
Samples / second
Effective tokens / second
Padded tokens / second
Estimated steps / epoch
Estimated epoch duration
```

This benchmark is intended to evaluate practical DirectML training
throughput and sustained execution behavior. It complements the shorter
compatibility benchmark: a configuration that passes a short
compatibility test is not necessarily stable during longer real-data
training.

------------------------------------------------------------------------

## Log-Sync Trainer Benchmark

The `benchmark_log_sync.ps1` utility compares the stable Dense pretraining
trainer with the log-synchronized variant over the same bounded workload.

Run it from the project root:

``` powershell
.\scripts\benchmark_log_sync.ps1
```

The current 1000-step comparison measured:

``` text
Stable:    2.0038 s / step
Log-Sync:  1.9307 s / step
Speedup:   1.038×
Reduction: 3.65%
```

The benchmark validates the throughput improvement. Checkpoint-quality
validation remains required before Log-Sync becomes the final training
reference.

------------------------------------------------------------------------

## Bounded Training Runs

The DirectML-compatible model trainers support bounded runs through the
`--max_steps` argument.

This allows training workflows to be exercised for a fixed number of
iterations without requiring a complete epoch. The shared bounded-run
behavior is used by pretraining, Full SFT, LoRA, distillation, GRPO,
Agent RL, and PPO.

Example:

``` powershell
python trainer/train_pretrain.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --max_steps 100
```

Bounded runs are useful for:

-   validating real-data training stability;
-   comparing batch-size and sequence-length configurations;
-   reproducing DirectML memory failures;
-   investigating CPU fallback behavior;
-   running longer stability checks without committing to a full
    training epoch.

For DirectML performance work, bounded real-data runs should be used
together with `benchmark_training.py`: the trainer validates the actual
training workflow, while the benchmark provides more focused throughput
measurements.

## DirectML Loss Validation Tests

Targeted regression utilities for the DirectML cross-entropy
investigation are stored in `tests/`:

``` text
test_directml_cross_entropy.py
test_directml_loss_reductions.py
test_batch_loss_consistency.py
test_train_diagnostic_loss.py
```

They validate the DirectML `cross_entropy` reduction behavior,
including the retained per-token `reduction="none"` → FP32 valid-token
mean path, and compare training-path loss with token-normalized
diagnostic loss. These
tests are kept separate from `scripts/`, while
`scripts/diagnose_pretrain.py` remains the checkpoint-analysis utility.

------------------------------------------------------------------------

## Trainer Smoke Test Runner

The `tests/test_all_trainers.py` utility executes the DirectML FP16
trainer smoke tests sequentially from a single Python entry point.

Run it explicitly from the project root:

``` bash
python tests/test_all_trainers.py
```

The runner currently validates:

``` text
Dense Pretrain
Dense Full SFT
MoE Pretrain
MoE Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

Each trainer uses a bounded configuration through `--max_steps`. The
test runner launches every trainer from the `trainer/` working directory
so the standard MiniMind relative paths remain valid.

The smoke suite stops on the first failure and reports the number of
completed trainers. The complete M4 validation run passed for all
trainers included in this suite:

``` text
Passed: 9/9
```

This runner is intentionally executed separately from the normal
`pytest -q` suite because it launches real DirectML training workloads
and is substantially heavier than unit and lightweight regression tests.
