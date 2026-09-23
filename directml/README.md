# MiniMind DirectML — Archived Experiment

> **Archived / historical**
>
> This directory contains the former Windows + DirectML adaptation of MiniMind.
> It is no longer the active training backend.
>
> Active development has moved to **ROCm on Windows**:
> [return to the main README](../README.md).

This experiment investigated whether the CUDA-oriented MiniMind training pipeline could be adapted to run on an AMD Radeon GPU under Windows through `torch-directml`.

The goal was to replace or bypass CUDA-specific execution assumptions while preserving the original MiniMind training logic as much as possible.

The experiment achieved broad execution compatibility, but final model-quality validation showed that the locally trained DirectML checkpoints were not reliable. The implementation is therefore preserved as an engineering archive rather than maintained as an active backend.

---

## Experiment outcome

The DirectML adaptation could:

- initialize and use the dedicated Radeon GPU through `torch-directml`;
- place models and tensors on DirectML devices;
- execute forward and backward passes;
- run optimizer steps;
- complete bounded training runs;
- complete full pretraining epochs;
- save and reload checkpoints;
- run the main MiniMind training workflows through DirectML smoke tests.

However, execution success did not translate into correct trained checkpoints.

The decisive validation result was:

```text
Official upstream pretrain_768.pth
    CPU inference       -> coherent
    DirectML inference  -> coherent

Locally trained DirectML checkpoints
    epoch 1             -> incoherent
    epoch 2             -> incoherent
```

Tokenizer and dataset checks were clean, and the official upstream checkpoint generated coherently through the same evaluation path.

The unresolved problem was therefore isolated to the custom DirectML training path used in this experiment.

**DirectML training was abandoned for this project.**

---

## Why this archive is kept

The DirectML work remains useful as a documented compatibility study.

It records:

- which CUDA assumptions had to be replaced or guarded;
- how `torch-directml` exposed the AMD GPU to PyTorch;
- which operators required special handling or CPU fallbacks;
- how FP16 training was adapted;
- how training correctness was tested;
- why finite loss and successful epochs are not sufficient validation;
- why checkpoint-quality testing must happen early.

The code is retained for reproducibility, historical comparison and engineering reference.

It should not be used as the default MiniMind training implementation.

---

## Historical tested configuration

The DirectML experiment was tested with:

| Component | Configuration |
| --- | --- |
| OS | Windows |
| GPU | AMD Radeon RX 7800 XT |
| RAM | 32 GB |
| Python | 3.10 |
| PyTorch | 2.4.1+cpu |
| Backend | `torch-directml` |
| DirectML adapter | `directml:1` |
| PyTorch device representation | `privateuseone:1` |

These values describe the archived experiment only.

They are not the dependency or backend requirements of the active ROCm version of the repository.

---

## What was adapted

The archived implementation contains DirectML-specific changes for:

- device detection and initialization;
- replacement or guarding of CUDA-specific device handling;
- model and tensor placement;
- trainer compatibility;
- unsupported DirectML operators;
- CPU fallback behavior;
- deterministic training fixtures;
- cross-trainer smoke tests;
- real-training benchmarks;
- Dense FP16 pretraining;
- static loss scaling;
- FP32 master-weight optimizer updates;
- per-token cross-entropy with FP32 valid-token averaging;
- bounded `--max_steps` validation;
- MoE routing fallbacks for unsupported scatter behavior;
- Windows/DirectML compatibility fixes for Agent RL;
- sequential Windows training utilities;
- DirectML compatibility auditing.

These modifications belong to this historical implementation and should not be copied into the active ROCm code unless a current ROCm issue independently requires an equivalent change.

---

## Historical installation

The following commands reproduce the environment model used by the archived DirectML experiment.

They are **not** the installation instructions for the active ROCm project.

### 1. Enter the archived implementation

From the repository root:

```powershell
cd directml
```

### 2. Create a Python 3.10 environment

```powershell
py -3.10 -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the historical dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The active ROCm environment is maintained separately at the repository root.

---

## Historical validation

The DirectML implementation included an automated test suite for the compatibility layer and training pipeline.

Run the archived tests from this directory:

```powershell
pytest -q
```

The heavier trainer validation used:

```powershell
python tests/test_all_trainers.py
```

That smoke runner exercised bounded DirectML training across the main workflows and was used to prove execution compatibility.

Passing these tests did **not** prove trained model quality. That distinction became the main lesson of the experiment.

---

## Known DirectML constraints

The experiment encountered backend-specific limitations and unsupported operations.

Some operations required CPU fallback or alternative implementations. A known example was the fallback associated with `aten::lerp.Scalar_out`.

The detailed findings are preserved in:

- [DirectML limitations](docs/directml_limitations.md)
- [DirectML issues](docs/directml_issues.md)
- [DirectML benchmarks](docs/directml_benchmarks.md)
- [DirectML audit](docs/directml_audit.md)

These documents describe the historical DirectML environment and should not be interpreted as ROCm limitations.

---

## Archived project structure

```text
directml/
│
├── checkpoints/             # Historical/local training checkpoints
├── dataset/                 # Dataset code used by the experiment
│
├── docs/
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
├── logs/
│   └── directml/            # Preserved training logs
│
├── model/                   # Historical model implementation
├── out/                     # Historical/local generated weights
├── scripts/                 # DirectML development and benchmark utilities
├── tests/                   # DirectML compatibility tests
├── trainer/                 # DirectML-adapted training pipeline
│
├── eval_llm.py
├── requirements.txt
└── README.md
```

Local checkpoints and generated artifacts may not be tracked by Git depending on the repository configuration.

---

## Documentation

### DirectML compatibility

- [DirectML limitations](docs/directml_limitations.md) — unsupported operations, compatibility constraints and CPU fallbacks.
- [DirectML issues](docs/directml_issues.md) — problems encountered during development, investigations and decisions.
- [DirectML benchmarks](docs/directml_benchmarks.md) — experimental compatibility and performance results.
- [DirectML audit](docs/directml_audit.md) — audit of CUDA-specific code and DirectML compatibility concerns.

### Historical development material

- [Development tools](docs/development-tools.md) — audit, fixture generation, training and benchmarking utilities.
- [Training commands](docs/training_commands.md) — historical DirectML training configurations and resume commands.
- [Project memory](docs/project_memory.md) — architectural decisions and technical lessons from the experiment.
- [Update log](docs/update_log.md) — chronological record of the DirectML adaptation.
- [Roadmap](docs/roadmap.md) — historical roadmap from the DirectML development phase.

---

## Original MiniMind documentation

The upstream README files are preserved outside this archive:

```text
../docs/original/
```

- [Original Chinese README](../docs/original/README.md)
- [Original English README](../docs/original/README_en.md)

The original project is available at:

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

## Main lesson

The most important result of the DirectML experiment was methodological:

```text
training runs
+ finite losses
+ successful checkpoints
≠ validated model training
```

Backend validation must also include the behavior of the resulting model.

The official MiniMind checkpoint worked through the same inference path on CPU and DirectML, while the locally trained DirectML checkpoints did not produce coherent output after the tested epochs.

For that reason, further training work moved away from DirectML instead of adding more compatibility workarounds.

---

## Current project

For the active Windows + AMD implementation, see:

**[MiniMind ROCm Windows](../README.md)**

---

## License

This archived adaptation remains part of the MiniMind fork and follows the repository license.

See [../LICENSE](../LICENSE).
