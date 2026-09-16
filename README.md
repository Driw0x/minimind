# MiniMind DirectML

A Windows and DirectML adaptation of
[MiniMind](https://github.com/jingyaogong/minimind), designed to run and
train MiniMind models on DirectX 12 compatible GPUs without requiring
CUDA.

This repository is based on the original MiniMind project by
[jingyaogong](https://github.com/jingyaogong).

> M4 performance and stability validation is complete. Project
> finalization is in progress.

------------------------------------------------------------------------

## About this fork

The original MiniMind project primarily targets CUDA-based environments.

This fork aims to provide a Windows-compatible alternative using PyTorch
DirectML, allowing MiniMind to run on GPUs supported by DirectX 12.

The main goals are:

-   support Windows + DirectML environments;
-   remove the dependency on CUDA for GPU acceleration;
-   adapt MiniMind training and inference code where necessary;
-   identify PyTorch operations not natively supported by DirectML;
-   provide tests to validate DirectML compatibility;
-   document differences from the upstream project.

------------------------------------------------------------------------

## Current status

MiniMind training and inference are functional on Windows with DirectML.

**M1–M4 are complete. M5 — Project Finalization is currently in
progress.**

The DirectML adaptation supports the main MiniMind training pipeline,
including pretraining, SFT, LoRA, DPO, GRPO, PPO, Agent RL, knowledge
distillation, and Dense/MoE workflows.

During M5, long-run validation identified two silent training-quality
issues specific to the tested DirectML path:

- direct FP16 AdamW updates could remain finite while converging toward
  a degenerate self-copying model;
- DirectML cross-entropy mean reduction with ignored padding tokens did
  not normalize over valid tokens as expected.

Dense DirectML pretraining now uses FP16 compute with FP32 master
weights and per-token cross-entropy followed by FP32 averaging over
valid non-padding tokens.

Targeted reduction tests showed that both the DirectML default mean
reduction and the initial `sum / valid_tokens` workaround were
incorrect on the tested FP16 path. The retained implementation matches
the CPU FP32 reference within approximately `0.003` on the batch-32
validation case.

Earlier full-epoch checkpoints produced before this final loss
correction are retained as investigation artifacts rather than final
training bases. Final Dense pretraining was rerun from scratch with the
corrected loss path and completed both epochs successfully.

The final checkpoint was additionally evaluated across 4,096 samples
from four dataset regions without recurrence of the historical
self-copying collapse. Dense pretraining is complete; Full SFT is the
next full-training stage.

See the [project roadmap](docs/roadmap.md) for the complete development
plan.

------------------------------------------------------------------------

## Tested configuration

DirectML compatibility has currently been validated on the following
configuration:

  Component                       Configuration
  ------------------------------- ----------------------------
  OS                              Windows
  GPU                             AMD Radeon RX 7800 XT
  RAM                             32 GB
  Python                          3.10
  PyTorch                         2.4.1+cpu
  Backend                         torch-directml
  DirectML adapter                directml:1 (dedicated GPU)
  PyTorch device representation   privateuseone:1

Other hardware and software configurations may work but have not yet
been validated.

Benchmark results are hardware-specific and should not be interpreted as
universal DirectML limits.

See [DirectML benchmarks](docs/directml_benchmarks.md) for the tested
compatibility configurations.

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Driw0x/minimind
cd minimind
```

### 2. Create a virtual environment

``` bash
py -3.10 -m venv .venv
```

### 3. Activate the environment

PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

``` cmd
.venv\Scripts\activate
```

### 4. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies of this fork include `torch-directml` for DirectML
support.

------------------------------------------------------------------------

## DirectML compatibility tests

The repository includes an automated Pytest suite covering the DirectML
adaptation and the main MiniMind training pipeline.

Run the complete test suite from the project root:

``` bash
pytest -q
```

For the heavier end-to-end DirectML trainer smoke validation, run:

``` bash
python tests/test_all_trainers.py
```

This explicitly launches bounded DirectML FP16 runs for Dense Pretrain,
Dense Full SFT, MoE Pretrain, MoE Full SFT, LoRA, Distillation, GRPO,
Agent RL, and PPO. The final M4 run passed all `9/9` trainers included
in this suite. It is kept separate from
the normal `pytest -q` workflow because it performs real training
workloads.

The Pytest suite covers:

-   DirectML device handling;
-   model and tensor placement;
-   forward and backward passes;
-   optimizer execution;
-   dataset fixtures;
-   training smoke tests;
-   checkpoint compatibility between training stages.

A warning related to `aten::lerp.Scalar_out` may appear because this
operation is not currently supported natively by the DirectML backend
and falls back to CPU execution.

See [DirectML limitations](docs/directml_limitations.md) for known
fallbacks and compatibility constraints.

------------------------------------------------------------------------

## Project structure

``` text
minimind/
│
├── checkpoints/             # Training resume checkpoints (not tracked)
├── dataset/                 # Dataset loading and training data
│
├── docs/                    # Project and DirectML documentation
│   ├── original/            # Original MiniMind documentation
│   ├── development-tools.md
│   ├── directml_audit.md
│   ├── directml_benchmarks.md
│   ├── directml_issues.md
│   ├── directml_limitations.md
│   ├── project_memory.md
│   ├── roadmap.md
│   ├── training_commands.md
│   └── update_log.md
│
├── images/                  # Project images and resources
├── model/                   # MiniMind model implementation
├── out/                     # Generated model weights (not tracked)
├── scripts/                 # Development, evaluation and API utilities
├── tests/                   # Automated compatibility tests
│   └── fixtures/            # Small deterministic test datasets
│
├── trainer/                 # DirectML-compatible training pipeline
│
├── .gitignore
├── eval_llm.py              # Local model evaluation and inference
├── requirements.txt
└── README.md
```

Generated model weights and training checkpoints are local runtime
artifacts and are not intended to be tracked by Git.

------------------------------------------------------------------------

## Documentation

Detailed technical information about the DirectML adaptation is
maintained separately in `docs/`.

### DirectML

-   [DirectML limitations](docs/directml_limitations.md) --- current
    unsupported operations, compatibility constraints, and CPU
    fallbacks.
-   [DirectML issues](docs/directml_issues.md) --- technical problems
    encountered during development, their causes, solutions, and
    decisions.
-   [DirectML benchmarks](docs/directml_benchmarks.md) --- experimental
    compatibility and performance results for DirectML training
    configurations.
-   [DirectML audit](docs/directml_audit.md) --- automatically generated
    audit of CUDA-specific code and potential DirectML compatibility
    concerns.

### Development

-   [Roadmap](docs/roadmap.md) --- development milestones and remaining
    work.
-   [Development tools](docs/development-tools.md) --- DirectML audit,
    deterministic test fixture generation, sequential training with
    `train_all.ps1`, real-training benchmarking, bounded runs, and the
    Python cross-trainer smoke runner.
-   [Training commands](docs/training_commands.md) --- per-trainer
    training configurations, periodic checkpoints, and resume commands.

### Project history and memory

-   [Project memory](docs/project_memory.md) --- durable architectural
    decisions, technical lessons, and validation principles.
-   [Update log](docs/update_log.md) --- chronological record of the
    main changes made to the DirectML adaptation.

------------------------------------------------------------------------

## Original documentation

The original MiniMind README files are preserved in:

``` text
docs/original/
```

-   Chinese documentation:
    [docs/original/README.md](docs/original/README.md)
-   English documentation:
    [docs/original/README_en.md](docs/original/README_en.md)

These files contain the original project documentation, examples, and
usage instructions.

------------------------------------------------------------------------

## Differences from upstream

This fork focuses specifically on Windows and DirectML support.

Changes include:

-   DirectML device detection and initialization;
-   replacement or guarding of CUDA-specific device handling;
-   DirectML-compatible tensor and model placement;
-   adaptations to the MiniMind training pipeline;
-   component-specific device placement where required;
-   handling and documentation of unsupported DirectML operators;
-   DirectML-specific compatibility and training tests;
-   deterministic fixtures for lightweight training validation;
-   DirectML compatibility and real-training benchmarking;
-   DirectML FP16 Dense pretraining with static loss scaling and FP32
    master-weight optimizer updates;
-   per-token cross-entropy with FP32 valid-token averaging for DirectML;
-   bounded `--max_steps` validation across the main trainable
    workflows;
-   explicit cross-trainer DirectML smoke validation;
-   DirectML-compatible MoE routing fallback for unsupported scatter
    behavior;
-   Windows and DirectML FP16 compatibility fixes for Agent RL;
-   development utilities for DirectML compatibility auditing;
-   sequential training utilities for Windows and DirectML;
-   dependency changes for a CUDA-free Windows environment.

The adaptation is being implemented progressively, so not every MiniMind
feature should currently be assumed to work with DirectML.

------------------------------------------------------------------------

## Upstream project

This repository is a fork of
[MiniMind](https://github.com/jingyaogong/minimind).

MiniMind and the original source code are developed by their respective
authors.

This fork provides additional modifications aimed at Windows and
DirectML compatibility.

------------------------------------------------------------------------

## License

This fork retains the license of the original MiniMind project.

See [LICENSE](LICENSE) for the applicable license terms.
