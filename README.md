# MiniMind DirectML

A Windows and DirectML adaptation of [MiniMind](https://github.com/jingyaogong/minimind), designed to run and train MiniMind models on DirectX 12 compatible GPUs without requiring CUDA.

This repository is based on the original MiniMind project by [jingyaogong](https://github.com/jingyaogong).

> The DirectML adaptation is currently a work in progress.

---

## About this fork

The original MiniMind project primarily targets CUDA-based environments.

This fork aims to provide a Windows-compatible alternative using PyTorch DirectML, allowing MiniMind to run on GPUs supported by DirectX 12.

The main goals are:

- support Windows + DirectML environments;
- remove the dependency on CUDA for GPU acceleration;
- adapt MiniMind training and inference code where necessary;
- identify PyTorch operations not natively supported by DirectML;
- provide tests to validate DirectML compatibility;
- document differences from the upstream project.

---

## Current status

MiniMind training and inference are functional on Windows with DirectML.

The main training pipeline has been validated with DirectML, including:

- pretraining;
- supervised fine-tuning (SFT);
- LoRA;
- DPO;
- GRPO;
- PPO;
- Agent RL;
- knowledge distillation;
- checkpoint compatibility between training stages.

Some operations still require CPU fallback or have limited DirectML support.

See [DirectML limitations](docs/directml-limitations.md) for currently known compatibility limitations.

The project is currently moving into **M4 — Performance & Stability**.

See the [project roadmap](docs/roadmap.md) for the complete development plan.

---

## Tested configuration

DirectML compatibility has currently been validated on the following configuration:

| Component | Configuration |
|---|---|
| OS | Windows |
| GPU | AMD Radeon RX 7800 XT |
| RAM | 32 GB |
| Python | 3.10 |
| PyTorch | 2.4.1+cpu |
| Backend | torch-directml |
| DirectML device | privateuseone:0 |

Other hardware and software configurations may work but have not yet been validated.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Driw0x/minimind
cd minimind
```

### 2. Create a virtual environment

```bash
py -3.10 -m venv .venv
```

### 3. Activate the environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies of this fork include `torch-directml` for DirectML support.

---

## DirectML compatibility tests

The repository includes an automated Pytest suite covering the DirectML adaptation and the main MiniMind training pipeline.

Run the complete test suite from the project root:

```bash
pytest -q
```

The test suite covers:

- DirectML device handling;
- model and tensor placement;
- forward and backward passes;
- optimizer execution;
- dataset fixtures;
- training smoke tests;
- checkpoint compatibility between training stages.

A warning related to `aten::lerp.Scalar_out` may appear because this operation is not currently supported natively by the DirectML backend and falls back to CPU execution.

See [DirectML limitations](docs/directml-limitations.md) for details about known fallbacks and compatibility constraints.

---

## Project structure

```text
minimind/
│
├── checkpoints/             # Training resume checkpoints (not tracked)
├── dataset/                 # Dataset loading and training data
│
├── docs/                    # Project and DirectML documentation
│   ├── original/            # Original MiniMind documentation
│   ├── development-tools.md
│   ├── directml_audit.md
│   ├── directml-limitations.md
│   ├── project_memory.md
│   ├── roadmap.md
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

Generated model weights and training checkpoints are local runtime artifacts and are not intended to be tracked by Git.

---

## Documentation

Detailed technical information about the DirectML adaptation is maintained separately in `docs/`.

### DirectML

- [DirectML limitations](docs/directml-limitations.md) — known limitations, unsupported operations, and CPU fallbacks.
- [DirectML audit](docs/directml_audit.md) — generated audit of CUDA-specific code and potential DirectML compatibility concerns.

### Development

- [Roadmap](docs/roadmap.md) — development milestones and current project progression.
- [Development tools](docs/development-tools.md) — DirectML audit, deterministic test fixture generation, and sequential training with `train_all.ps1`.

### Project history

- [Project memory](docs/project_memory.md) — technical problems encountered during development, their causes, and implemented solutions.
- [Update log](docs/update_log.md) — chronological record of the main changes made to the DirectML adaptation.

---

## Original documentation

The original MiniMind README files are preserved in:

```text
docs/original/
```

- Chinese documentation: [docs/original/README.md](docs/original/README.md)
- English documentation: [docs/original/README_en.md](docs/original/README_en.md)

These files contain the original project documentation, examples, and usage instructions.

---

## Differences from upstream

This fork focuses specifically on Windows and DirectML support.

Changes include:

- DirectML device detection and initialization;
- replacement or guarding of CUDA-specific device handling;
- DirectML-compatible tensor and model placement;
- adaptations to the MiniMind training pipeline;
- handling and documentation of unsupported DirectML operators;
- DirectML-specific compatibility and training tests;
- deterministic fixtures for lightweight training validation;
- development utilities for DirectML compatibility auditing;
- sequential training utilities for Windows and DirectML;
- dependency changes for a CUDA-free Windows environment.

The adaptation is being implemented progressively, so not every MiniMind feature should currently be assumed to work with DirectML.

---

## Upstream project

This repository is a fork of [MiniMind](https://github.com/jingyaogong/minimind).

MiniMind and the original source code are developed by their respective authors.

This fork provides additional modifications aimed at Windows and DirectML compatibility.

---

## License

This fork retains the license of the original MiniMind project.

See [LICENSE](LICENSE) for the applicable license terms.