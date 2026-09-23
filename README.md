# MiniMind ROCm Windows

A Windows + AMD ROCm adaptation of [MiniMind](https://github.com/jingyaogong/minimind).

This fork keeps the original MiniMind project structure as close to upstream as possible while validating native AMD GPU training on Windows through ROCm and PyTorch.

> **Active backend:** ROCm on Windows  
> **Primary target:** AMD Radeon RX 7800 XT (`gfx1101`)  
> **Legacy backend:** DirectML — archived in [`directml/`](directml/README.md)

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
- validate Full SFT after pretraining is stable;
- document the previous DirectML attempt separately.

---

## Why ROCm?

A previous version of this fork attempted to replace the CUDA-oriented execution path with `torch-directml`.

That work reached broad functional compatibility: training could run, full epochs could complete, and the main DirectML smoke tests passed.

However, model-quality validation exposed a blocking issue:

```text
Official upstream pretrain_768.pth
    CPU inference       -> coherent
    DirectML inference  -> coherent

Locally trained DirectML checkpoints
    epoch 1             -> incoherent
    epoch 2             -> incoherent
```

Tokenizer and dataset checks were clean, and the official checkpoint generated coherently through the same evaluation path. The DirectML training path was therefore not retained.

The full experiment, implementation details, benchmarks, tests and limitations are preserved here:

**[DirectML archived experiment](directml/README.md)**

ROCm is now the active AMD GPU backend for this fork.

---

## ROCm and PyTorch device semantics

PyTorch on ROCm intentionally reuses the CUDA-facing Python API.

This means that an AMD GPU running through ROCm is still addressed with:

```python
device = "cuda"
```

or:

```python
device = "cuda:0"
```

Do **not** replace the PyTorch device with `rocm`, `hip`, or a custom device name.

Typical checks are:

```python
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA build:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

For a ROCm build:

```text
torch.cuda.is_available() -> True
torch.version.hip         -> non-empty
torch.version.cuda        -> None
```

Using `torch.cuda`, `.cuda()`, `cuda:0`, CUDA autocast APIs, and related PyTorch interfaces is therefore normal on ROCm.

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
py -3.12 -m venv .venv-rocm
```

Activate it:

```powershell
.\.venv-rocm\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### 3. Install ROCm-enabled PyTorch

Install the ROCm/PyTorch build recommended by AMD for the selected Windows, GPU and ROCm versions.

Use the official installation selector rather than copying an old wheel command from this repository:

[AMD — Install PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

### 4. Install project dependencies

`requirements-rocm.txt` is intentionally maintained manually for the active Windows ROCm environment.

Once the file has been curated:

```powershell
pip install -r requirements-rocm.txt
```

The legacy DirectML dependency set must not be reused as the ROCm dependency set.

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

## Training strategy

The migration is intentionally validated in small steps.

### Phase 1 — Environment validation

Validate:

```text
ROCm installation
    ↓
PyTorch HIP build
    ↓
RX 7800 XT detected
    ↓
basic GPU tensor operations
```

### Phase 2 — Pretraining smoke test

Run a short Dense pretraining test before a full training run.

The goal is to verify:

```text
dataset loading
    ↓
forward pass
    ↓
finite loss
    ↓
backward pass
    ↓
optimizer step
    ↓
stable GPU memory
    ↓
checkpoint save/load
    ↓
early generation-quality check
```

The active pretraining entry point is:

```text
trainer/train_pretrain.py
```

Use the ROCm device through:

```text
cuda:0
```

### Phase 3 — Full pretraining

Only start the full run after the bounded smoke test produces a valid checkpoint.

Checkpoint quality should be evaluated during training rather than waiting until the end of multiple epochs.

### Phase 4 — Full SFT

Once the ROCm pretraining path is validated, continue with:

```text
trainer/train_full_sft.py
```

The same principle applies: validate a bounded run and checkpoint quality before launching the complete stage.

---

## Validation principle

A successful training process is not sufficient evidence that the backend is correct.

This project distinguishes:

```text
execution correctness
```

from:

```text
model-quality correctness
```

A backend is considered usable only when it can:

1. execute the training loop;
2. keep losses finite and numerically stable;
3. save and reload checkpoints;
4. produce checkpoints whose generated output remains coherent;
5. preserve expected behavior across training stages.

This validation rule comes directly from the previous DirectML experiment.

---

## Project structure

```text
minimind/
│
├── dataset/                  # Dataset loading
│
├── directml/                 # Archived DirectML compatibility experiment
│
├── docs/
│   └── original/             # Original MiniMind documentation
│
├── images/                   # Original project images/resources
├── model/                    # Active MiniMind model implementation
├── scripts/                  # Evaluation, conversion and serving utilities
├── trainer/                  # Active ROCm/Windows training code
│
├── eval_llm.py
├── requirements-rocm.txt     # Manually curated ROCm/Windows dependencies
├── .gitignore
├── LICENSE
└── README.md
```

Generated checkpoints, virtual environments, caches and local training outputs are runtime artifacts and should not be committed unless they are intentionally preserved as experimental evidence.

---

## DirectML archive

The complete previous DirectML adaptation is intentionally isolated from the active ROCm codebase:

[`directml/README.md`](directml/README.md)

It documents:

- why DirectML was initially investigated;
- the CUDA-to-DirectML compatibility changes;
- device-placement work;
- FP16 training workarounds;
- unsupported operations and CPU fallbacks;
- DirectML benchmarks;
- training smoke tests;
- checkpoint-quality validation;
- the reason the DirectML training path was abandoned.

The DirectML directory should be treated as an archived engineering experiment, not as an active backend.

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
