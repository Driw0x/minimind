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

The automated test suite currently passes successfully:

```text
34 passed
```

Some operations still require CPU fallback or have limited DirectML support.

See [`docs/directml-limitations.md`](docs/directml-limitations.md) for details.

The project is currently moving into **M4 — Performance & Stability**.

---

## Roadmap

The DirectML adaptation is developed incrementally to ensure that each part of the MiniMind pipeline is validated on Windows before moving to the next stage.

### M1 — DirectML Foundation ✅

Establish and validate the basic DirectML environment.

- [x] Set up a Windows + DirectML environment
- [x] Replace the CUDA-oriented PyTorch dependency with `torch-directml`
- [x] Initialize a DirectML device
- [x] Move MiniMind models and tensors to DirectML
- [x] Validate the forward pass
- [x] Validate loss computation
- [x] Validate the backward pass
- [x] Validate the AdamW optimizer step
- [x] Add DirectML compatibility tests with Pytest
- [x] Validate installation from a clean virtual environment

### M2 — MiniMind Training on DirectML 🚧

Adapt the actual MiniMind training pipeline to run on DirectML.

- [x] Audit CUDA-specific code in the training pipeline
- [x] Introduce DirectML-compatible device handling
- [x] Adapt mixed precision and CUDA-specific utilities where necessary
- [x] Run a minimal pretraining job on DirectML
- [x] Verify that the training loss evolves correctly
- [x] Save a model checkpoint
- [x] Reload the checkpoint successfully
- [x] Run inference using the trained checkpoint

### M3 — Full Training Pipeline

Validate the main MiniMind training stages with DirectML.

- [x] Validate pretraining
- [x] Validate supervised fine-tuning (SFT)
- [x] Validate LoRA training
- [x] Validate additional training stages supported by MiniMind
- [x] Validate checkpoint compatibility between training stages
- [x] Document unsupported or partially supported DirectML operations

### M4 — Performance & Stability

Evaluate the practical usability of MiniMind with DirectML.

- [ ] Measure GPU memory usage
- [ ] Measure training throughput
- [ ] Identify CPU fallback operations
- [ ] Evaluate the performance impact of CPU fallbacks
- [ ] Test longer training runs
- [ ] Improve stability and error handling
- [ ] Document known limitations

### M5 — DirectML-ready Release

Prepare the fork for reproducible use by other Windows users.

- [ ] Finalize installation documentation
- [ ] Document the supported training workflows
- [ ] Add troubleshooting documentation
- [ ] Clean up remaining CUDA assumptions
- [ ] Validate the project from a fresh clone
- [ ] Publish the first stable DirectML-ready version

---

## Tested Configuration

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

## DirectML compatibility test

The repository includes a Pytest test suite for the basic DirectML execution pipeline.

Run:

```bash
pytest tests/test_directml.py -v
```

The tests currently verify:

```text
DirectML device
      ↓
MiniMind model on DirectML
      ↓
Input tensor on DirectML
      ↓
Forward pass
      ↓
Loss computation
      ↓
Backward pass
      ↓
AdamW optimizer step
```

A successful execution should currently report:

```text
6 passed
```

A warning related to `aten::lerp.Scalar_out` may also appear because this operation falls back to CPU execution.

---

## Project structure

```text
minimind/
│
├── checkpoints/             # Training resume checkpoints (not tracked)
├── dataset/                 # Dataset loading and training data
│
├── docs/                    # Project and DirectML documentation
│   └── original/            # Original MiniMind documentation
│   ├── development-tools.md
│   ├── directml_audit.md
│   ├── directml-limitations.md
│   ├── project_memory.md
│   └── update_log.md
│
├── images/                  # Project images and resources
├── model/                   # MiniMind model implementation
├── out/                     # Generated model weights (not tracked)
├── scripts/                 # Utility, evaluation and API scripts
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

## Development documentation

Additional documentation is available for the DirectML adaptation and development tools:

- [DirectML limitations](docs/directml-limitations.md) — known DirectML limitations, unsupported operations, and CPU fallbacks.
- [DirectML audit](docs/directml_audit.md) — generated compatibility audit of CUDA-specific code and potential DirectML issues.
- [Development tools](docs/development-tools.md) — usage of the DirectML audit and deterministic test fixture generation utilities.
- [Project memory](docs/project_memory.md) — technical issues encountered during each milestone, their causes, and implemented solutions.
- [Update log](docs/update_log.md) — chronological overview of the DirectML adaptation progress.

---

## Original documentation

The original MiniMind README files are preserved in:

```text
docs/original/
```

- Chinese documentation: [`docs/original/README.md`](docs/original/README.md)
- English documentation: [`docs/original/README_en.md`](docs/original/README_en.md)

These files contain the original project documentation, examples, and usage instructions.

---

## Differences from upstream

This fork focuses specifically on Windows and DirectML support.

Changes may include:

- DirectML device detection and initialization;
- replacement of CUDA-specific device handling;
- DirectML-compatible tensor and model placement;
- adaptations to training scripts;
- handling or documentation of unsupported DirectML operators;
- DirectML-specific compatibility tests;
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

See [`LICENSE`](LICENSE) for the applicable license terms.