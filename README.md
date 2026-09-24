# MiniMind ROCm Windows

A Windows + AMD ROCm adaptation of [MiniMind](https://github.com/jingyaogong/minimind).

This fork keeps the original MiniMind project structure as close to upstream as possible while validating native AMD GPU training on Windows through ROCm and PyTorch.

> **Active backend:** ROCm on Windows  
> **Primary target:** AMD Radeon RX 7800 XT (`gfx1101`)

---

## About this fork

The original MiniMind project is primarily developed around CUDA-oriented PyTorch environments.

This fork focuses on running and validating the same training pipeline on an AMD Radeon GPU under Windows using ROCm.

The objective is not to redesign MiniMind. The objective is to keep the upstream training logic intact where possible and make only the changes required for a correct ROCm/Windows execution path.

The current priorities are:

- validate MiniMind pretraining with ROCm on Windows;
- validate checkpoint quality early, not only successful execution;
- keep ROCm-specific changes minimal and isolated;
- preserve compatibility with the upstream MiniMind structure;
- validate Full SFT after pretraining is stable.

Project planning and reproducible training commands are documented separately
under [`docs/`](docs/).

---

## Legacy DirectML

The previous Windows + DirectML adaptation is preserved as a completed
compatibility and feasibility study under:

**[`directml/`](directml/README.md)**

It contains the historical implementation, benchmarks, tests, documentation and
final model-quality investigation. It is archived and is not part of the active
training path.

---

## ROCm and PyTorch device semantics

PyTorch on ROCm intentionally reuses the CUDA-facing Python API. An AMD GPU
running through ROCm is therefore addressed with `cuda` / `cuda:0`.

Do **not** replace the PyTorch device with `rocm`, `hip`, or a custom device
name. CUDA-named PyTorch interfaces are expected on the ROCm/HIP backend.

For the executable environment check, see
[Verify the ROCm environment](#verify-the-rocm-environment).

---

## Target environment

The active migration targets:

| Component | Target |
| --- | --- |
| Operating system | Windows |
| GPU | AMD Radeon RX 7800 XT |
| GPU architecture | `gfx1101` |
| Compute backend | AMD ROCm |
| Framework | PyTorch for ROCm |
| Training device | `cuda:0` through the PyTorch HIP backend |

ROCm, PyTorch and AMD driver versions must remain compatible with each other.

Before installing or updating the environment, check the official AMD compatibility information:

- [ROCm documentation](https://rocm.docs.amd.com/)
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Driw0x/minimind.git
cd minimind
```

### 2. Create the ROCm virtual environment

Use a Python version supported by the ROCm/PyTorch combination selected from AMD's compatibility matrix.

Example:

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### 3. Install ROCm-enabled PyTorch

The ROCm-specific PyTorch dependencies are kept separately in:

```text
requirements-rocm.txt
```

Install them first:

```powershell
pip install -r requirements-rocm.txt
```

This keeps the backend-specific PyTorch packages separate from the upstream
MiniMind dependencies.

### 4. Install MiniMind dependencies

Install the remaining project dependencies from:

```powershell
pip install -r requirements.txt
```

The recommended installation order is therefore:

```text
ROCm / PyTorch dependencies
        ↓
MiniMind dependencies
```

ROCm, PyTorch and AMD driver versions must remain compatible with each other.
When changing versions, verify the current AMD compatibility documentation.

---

## Verify the ROCm environment

Before running MiniMind, verify that PyTorch can access the Radeon GPU:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected characteristics:

```text
hip: <ROCm/HIP version>
cuda available: True
device: AMD Radeon RX 7800 XT
```

A minimal GPU computation can also be tested with:

```python
import torch

assert torch.cuda.is_available()
assert torch.version.hip is not None

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

print(y.device)
print(torch.cuda.get_device_name(0))
```

The tensor device will still be reported as:

```text
cuda:0
```

This is expected with PyTorch ROCm.

---

## Project documentation

Documentation is split by responsibility:

- [`docs/roadmap.md`](docs/roadmap.md) — milestones, mini-vs-full training rules,
  stage dependencies and validation criteria;
- [`docs/training_commands.md`](docs/training_commands.md) — reproducible trainer
  commands, checkpoint outputs and resume usage;
- [`docs/original/`](docs/original/) — preserved upstream MiniMind documentation.

---

## Project structure

```text
minimind/
│
├── dataset/                  # Dataset loading
│
├── docs/                     # Project documentation
│   ├── roadmap.md            # Active milestones and validation plan
│   ├── training_commands.md  # Reproducible training commands
│   └── original/             # Original MiniMind documentation
│
├── images/                   # Original project images/resources
├── model/                    # Active MiniMind model implementation
├── scripts/                  # Evaluation, conversion and serving utilities
├── trainer/                  # Active ROCm/Windows training code
│
├── eval_llm.py
├── requirements.txt          # MiniMind dependencies
├── requirements-rocm.txt     # ROCm/PyTorch dependencies
├── .gitignore
├── LICENSE
└── README.md
```

Generated checkpoints, virtual environments, caches and local training outputs are runtime artifacts and should not be committed unless they are intentionally preserved as experimental evidence.

---

## Original MiniMind documentation

The original upstream README files are preserved under:

```text
docs/original/
```

- [Original Chinese README](docs/original/README.md)
- [Original English README](docs/original/README_en.md)

For upstream usage, models, datasets and project details, refer to those documents or to the original repository:

[github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

## Differences from upstream

This fork currently focuses on Windows + AMD ROCm compatibility.

The intended differences from upstream are limited to what is required for that environment, such as:

- ROCm/Windows dependency management;
- validation of PyTorch HIP execution on Radeon hardware;
- removal of assumptions that are strictly NVIDIA-specific when they block ROCm;
- ROCm-compatible mixed-precision and training behavior where necessary;
- backend validation and checkpoint-quality checks;
- documentation of the Windows AMD setup.

Because PyTorch ROCm reuses the CUDA-facing Python API, CUDA-named PyTorch calls should not be changed merely because the physical GPU is AMD.

Changes should remain minimal and should not diverge from upstream training logic without a verified reason.

---

## Upstream project

This repository is based on:

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

MiniMind and the original source code are developed by their respective authors.

This fork focuses on adapting and validating that project for native Windows training on AMD Radeon GPUs through ROCm.

---

## License

This fork retains the license of the original MiniMind project.

See [LICENSE](LICENSE) for the applicable license terms.
