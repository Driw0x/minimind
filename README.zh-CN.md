[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="minimind-rocm-windows"></a>
# MiniMind ROCm Windows

一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 Windows + AMD ROCm 适配版本。

该分支尽可能保持原始 MiniMind 项目结构与上游版本一致，同时通过 ROCm 和 PyTorch 验证在 Windows 上原生使用 AMD GPU 进行训练。

> **活跃后端：** ROCm 在 Windows 上   **主要目标硬件：** AMD Radeon RX 7800 XT (`gfx1101`)

---

<a id="about-this-fork"></a>
## 关于此分支

原始的 MiniMind 项目主要围绕基于 CUDA 的 PyTorch 环境展开开发。

本分支专注于在 Windows 系统上使用 ROCm 驱动的 AMD Radeon GPU 运行并验证相同的训练流程。

目标并非重设计 MiniMind。目标是尽可能保持上游训练逻辑不变，并仅对实现 ROCm/Windows 正确执行路径所必需的部分进行修改。

当前优先事项包括：

- 保持已验证的 Dense Pretrain 和 Full SFT 检查点作为参考基准；
- 早期验证检查点质量，而不仅仅是执行成功；
- 将 ROCm 相关的修改保持最小化和隔离；
- 保持与上游 MiniMind 项目结构的兼容性；
- 以 Dense 基线为参照验证 MoE 训练路径。

MoE Pretrain 和 MoE Full SFT 训练及其在 AMD ROCm 上的初步评估均已完成。保留的检查点为 `pretrain_768_moe.pth` 和 `full_sft_768_moe.pth`；详见[质量对比](docs/generation_quality.zh-CN.md)。
Hugging Face 仓库：[Driw0x/minimind-rocm-pytorch](https://huggingface.co/Driw0x/minimind-rocm-pytorch)。

项目规划和可复现的训练命令在 [`docs/`](docs/) 中单独记录。

---

<a id="legacy-directml"></a>
## 旧版 DirectML

之前的 Windows + DirectML 适配版本被保留为已完成的兼容性与可行性研究，位于：

**[`directml/`](directml/README.zh-CN.md)**

其中包含历史实现、基准测试、测试用例、文档以及最终模型质量分析。该版本已归档，不再属于当前训练路径的一部分。

---

<a id="rocm-and-pytorch-device-semantics"></a>
## ROCm 和 PyTorch 设备语义

PyTorch 在 ROCm 上有意复用面向 CUDA 的 Python API。因此，通过 ROCm 运行的 AMD GPU 会以 `cuda` / `cuda:0` 的方式被识别。

请**不要**将 PyTorch 设备替换为 `rocm`、`hip` 或自定义设备名称。在 ROCm/HIP 后端中，仍需使用 CUDA 命名的 PyTorch 接口。

有关可执行环境的检查，请参见 [验证 ROCm 环境](#verify-the-rocm-environment)。

---

<a id="target-environment"></a>
## 目标环境

当前迁移目标：

|组件|目标|
| --- | --- |
|操作系统|Windows|
|GPU|AMD Radeon RX 7800 XT|
|GPU 架构| `gfx1101` |
|计算后端|AMD ROCm|
|框架|PyTorch for ROCm|
|训练设备|`cuda:0` 通过 PyTorch HIP 后端|

ROCm、PyTorch 和 AMD 驱动版本之间必须保持相互兼容。

在安装或更新环境之前，请检查官方 AMD 兼容性信息：

- [ROCm 文档](https://rocm.docs.amd.com/)
- [ROCm 兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [安装 PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

---

<a id="installation"></a>
## 安装

<a id="1-clone-the-repository"></a>
### 1. 克隆仓库

```powershell
git clone https://github.com/Driw0x/minimind.git
cd minimind
```

<a id="2-create-the-rocm-virtual-environment"></a>
### 2. 创建 ROCm 虚拟环境

使用AMD兼容性矩阵中支持的ROCm/PyTorch组合所对应的Python版本。

示例：

```powershell
py -3.13 -m venv .venv
```

激活它：

```powershell
.\.venv\Scripts\Activate.ps1
```

升级pip：

```powershell
python -m pip install --upgrade pip
```

<a id="3-install-rocm-enabled-pytorch"></a>
### 3. 安装支持ROCm的PyTorch

ROCm专用的PyTorch依赖项单独存储在：

```text
requirements-rocm.txt
```

首先安装它们：

```powershell
pip install -r requirements-rocm.txt
```

这将使后端特定的PyTorch包与上游MiniMind依赖项分离。

<a id="4-install-minimind-dependencies"></a>
### 4. 安装MiniMind依赖项

从以下位置安装剩余的项目依赖项：

```powershell
pip install -r requirements.txt
```

因此推荐的安装顺序是：

```text
ROCm / PyTorch dependencies
        ↓
MiniMind dependencies
```

ROCm、PyTorch和AMD驱动版本之间必须相互兼容。在更改版本时，请验证当前的AMD兼容性文档。

---

<a id="verify-the-rocm-environment"></a>
## 验证ROCm环境

在运行MiniMind之前，请验证PyTorch能否访问Radeon GPU：

```powershell
python -c "import torch; print('torch:', torch.__version__); print('hip:', torch.version.hip); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

预期特性：

```text
hip: <ROCm/HIP version>
cuda available: True
device: AMD Radeon RX 7800 XT
```

也可以使用以下命令测试最小的GPU计算：

```python
import torch

assert torch.cuda.is_available()
assert torch.version.hip is not None

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

print(y.device)
print(torch.cuda.get_device_name(0))
```

张量设备仍将报告为：

```text
cuda:0
```

这是PyTorch ROCm的预期行为。

---

<a id="project-documentation"></a>
## 项目文档

文档按责任划分：

- [`docs/roadmap.md`](docs/roadmap.zh-CN.md) — 里程碑、迷你训练与完整训练规则、阶段依赖关系和验证标准；
- [`docs/training_commands.md`](docs/training_commands.zh-CN.md) — 可复现的训练命令、检查点输出和恢复使用说明；

- [`docs/development_tools.md`](docs/development_tools.zh-CN.md) — 诊断工具和参考结果；
- [`docs/generation_quality.md`](docs/generation_quality.zh-CN.md) — 检查点生成质量对比；
- `docs/memory.md`（本次仓库快照未包含该文档） — 运行时内存设计与评估计划；
- [`docs/original/`](docs/original/) — 保留上游MiniMind文档。

---

<a id="project-structure"></a>
## 项目结构

```text
minimind/
│
├── dataset/                  # Dataset loading
│
├── docs/                     # Project documentation
│   ├── roadmap.md            # Active milestones and validation plan
│   ├── training_commands.md  # Reproducible training commands
│   ├── development_tools.md  # Diagnostic tools and reference results
│   ├── generation_quality.md # Checkpoint generation-quality comparisons
│   ├── memory.md             # Runtime-memory design and evaluation plan
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

生成的检查点、虚拟环境、缓存和本地训练输出属于运行时产物，除非有意作为实验证据保留，否则不应提交到代码库。

---

<a id="original-minimind-documentation"></a>
## 原始MiniMind文档

原始上游README文件保存在：

```text
docs/original/
```

- [原始中文README](docs/original/README.zh-CN.md)
- [原始英文README](docs/original/README.md)

对于上游使用，模型、数据集和项目详情，请参考这些文档或原始仓库：

[github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

<a id="differences-from-upstream"></a>
## 与上游的差异

该分支目前专注于Windows + AMD ROCm的兼容性。

与上游的预期差异仅限于为该环境所必需的内容，例如：

- ROCm/Windows依赖管理；
- 验证PyTorch HIP在Radeon硬件上的执行；
- 移除严格针对NVIDIA的假设以避免阻碍ROCm；
- 必要时的ROCm兼容混合精度和训练行为；
- 后端验证和检查点质量检查；
- Windows AMD环境的设置说明。

由于PyTorch ROCm复用了面向CUDA的Python API，因此即使物理GPU是AMD，也不应仅仅因为这一点而更改CUDA命名的PyTorch调用。

变更应保持最小化，并且在没有经过验证的理由的情况下，不应偏离上游训练逻辑。

---

<a id="upstream-project"></a>
## 上游项目

本仓库基于：

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

MiniMind及其原始源代码由各自的作者开发。

本分支专注于将该项目适配并验证以实现原生Windows环境下在AMD Radeon GPU上通过ROCm进行训练。

---

<a id="license"></a>
## 许可

本分支保留了原始MiniMind项目的许可。

有关适用许可条款，请参见[LICENSE](LICENSE)。

<!-- multilingual-documentation -->

## 更多文档

- [dataset/dataset.md](dataset/dataset.zh-CN.md)
- [directml/dataset/dataset.md](directml/dataset/dataset.zh-CN.md)
