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

Basic MiniMind execution and training operations have been successfully tested with DirectML.

Currently validated:

- DirectML device initialization;
- model transfer to DirectML;
- input tensors on DirectML;
- forward pass;
- loss computation;
- backward pass;
- AdamW optimizer step;
- gradient reset;
- basic training step.

The current DirectML test suite passes successfully:

```text
6 passed, 1 warning
```

The remaining warning is caused by an unsupported DirectML operation inside `AdamW`:

```text
aten::lerp.Scalar_out
```

DirectML automatically falls back to the CPU for this operation. This does not prevent training from running, but may have performance implications.

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
├── dataset/                  # Dataset loading and processing
│   ├── __init__.py
│   ├── dataset.md
│   └── lm_dataset.py
│
├── docs/
│   └── original/             # Original MiniMind README files
│       ├── README.md
│       └── README_en.md
│
├── images/                   # Project images and resources
│
├── model/                    # MiniMind model implementation
│
├── scripts/                  # Utility and execution scripts
│
├── tests/                    # Tests including DirectML compatibility
│   ├── __init__.py
│   └── test_directml.py
│
├── trainer/                  # Training scripts
│
├── .gitignore
├── CODE_OF_CONDUCT.md
├── eval_llm.py
├── LICENSE
├── requirements.txt
└── README.md                 # Documentation for this fork
```

The structure may evolve as DirectML support is progressively integrated into the different parts of MiniMind.

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