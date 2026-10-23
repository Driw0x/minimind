[English](README.md) | [Français](README.fr.md) | [简体中文](README.zh-CN.md)

<a id="minimind-directml--archived-experiment"></a>
# MiniMind DirectML — 已归档实验

> **归档 / 历史**
>
> 该目录包含曾经针对 Windows + DirectML 的 MiniMind 适配版本。它已不再作为活跃的训练后端。
>
> 活跃开发已迁移至 **ROCm 在 Windows 上**：[返回主 README](../README.zh-CN.md)。

本实验探究了是否可以将面向 CUDA 的 MiniMind 训练流程适配到通过 Windows 使用 AMD Radeon GPU 的环境中，通过 `torch-directml` 实现。

目标是尽可能保留原始 MiniMind 训练逻辑，同时替代或绕过与 CUDA 相关的执行假设。

实验实现了广泛的执行兼容性，但最终模型质量验证显示，本地训练的 DirectML 检查点不可靠。因此，该实现被保留为工程归档，而非作为活跃后端维护。

---

<a id="experiment-outcome"></a>
## 实验结果

DirectML 适配能够：

- 通过 `torch-directml` 初始化并使用专用的 Radeon GPU；
- 将模型和张量放置到 DirectML 设备上；
- 执行前向和反向传播；
- 运行优化器步骤；
- 完成有限的训练运行；
- 完成完整的预训练周期；
- 保存和重新加载检查点；
- 通过 DirectML 烟雾测试运行主 MiniMind 训练流程。

然而，执行成功并未转化为正确的训练检查点。

决定性的验证结果是：

```text
Official upstream pretrain_768.pth
    CPU inference       -> coherent
    DirectML inference  -> coherent

Locally trained DirectML checkpoints
    epoch 1             -> incoherent
    epoch 2             -> incoherent
```

分词器和数据集检查均正常，且通过相同评估路径生成的官方上游检查点表现一致。

最终的跨后端测试将失败进一步归因于测试的 DirectML FP16 数值路径。相同的 DirectML 训练检查点在 FP32 下与 ROCm 一致，但在 FP16 下存在显著差异。

**该项目已放弃 DirectML 训练。**

<a id="final-quantitative-validation"></a>
### 最终定量验证

在相同的 `1,000` 样本（`204,327` 有效标记）上，通过 ROCm 评估：

|检查点|损失|困惑度|生成|
| --- | ---: | ---: | --- |
|DirectML 训练| 6.880233 | 972.8529 |不连贯|
|ROCm 训练| 1.846587 | 6.3382 |连贯|

使用相同的 DirectML 训练检查点和一个固定批次（`873` 有效标记），DirectML/ROCm 损失在 FP16 下为 `5.733902 / 6.230469`，而在 FP32 下为 `6.231627 / 6.231628`。DirectML 在 FP16 下未使用 SDPA 时生成了非有限的对数概率。

这排除了检查点加载和通用推理作为最终解释，并将阻塞行为限定在测试的 DirectML FP16 数值路径上。

---

<a id="why-this-archive-is-kept"></a>
## 为何保留此归档

DirectML 的工作仍具有作为兼容性研究的文档价值。

它记录了：

- 哪些 CUDA 假设必须被替换或防护；
- `torch-directml` 如何使 AMD GPU 暴露于 PyTorch；
- 哪些操作符需要特殊处理或 CPU 降级；
- 如何适配 FP16 训练；
- 如何测试训练正确性；
- 为什么有限的损失和成功的训练周期不足以构成有效验证；
- 为什么检查点质量的测试必须在早期进行。

代码被保留用于可复现性、历史对比和工程参考。

它不应被用作MiniMind训练的默认实现。

---

<a id="historical-tested-configuration"></a>
## 历史测试配置

DirectML实验的测试内容包括：

|组件|配置|
| --- | --- |
|操作系统|Windows|
|GPU|AMD Radeon RX 7800 XT|
|内存|32 GB|
|Python| 3.10 |
|PyTorch|2.4.1+cpu|
|后端| `torch-directml` |
|DirectML适配器| `directml:1` |
|PyTorch设备表示| `privateuseone:1` |

这些值仅描述存档实验的情况。

它们不是当前ROCm版本仓库的依赖项或后端要求。

---

<a id="what-was-adapted"></a>
## 哪些内容被适配

存档实现中包含针对DirectML的特定修改，涉及：

- 设备检测和初始化；
- CUDA特定设备处理的替换或保护机制；
- 模型和张量的放置；
- 训练器兼容性；
- 不支持的DirectML操作符；
- CPU降级行为；
- 确定性训练环境配置；
- 跨训练器烟雾测试；
- 实际训练基准测试；
- 密集FP16预训练；
- 静态损失缩放；
- FP32主权重优化器更新；
- 每token交叉熵损失结合FP32有效token平均；
- 有界`--max_steps`验证；
- 不支持散列行为时的MoE路由降级机制；
- Agent RL在Windows/DirectML环境下的兼容性修复；
- Windows下顺序训练的辅助工具；
- DirectML兼容性审计。

这些修改属于该历史实现，除非当前ROCm版本独立存在需要等效变更，否则不应复制到活跃的ROCm代码中。

---

<a id="historical-installation"></a>
## 历史安装

以下命令可以重现存档DirectML实验所使用的环境模型。

它们**不是**当前ROCm项目的安装说明。

<a id="1-enter-the-archived-implementation"></a>
### 1. 进入存档实现

从仓库根目录：

```powershell
cd directml
```

<a id="2-create-a-python-310-environment"></a>
### 2. 创建Python 3.10环境

```powershell
py -3.10 -m venv .venv
```

<a id="3-activate-it"></a>
### 3. 激活该环境

```powershell
.\.venv\Scripts\Activate.ps1
```

<a id="4-install-the-historical-dependencies"></a>
### 4. 安装历史依赖项

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

当前ROCm环境在仓库根目录中独立维护。

---

<a id="historical-validation"></a>
## 历史验证

DirectML实现包含一个自动化测试套件，用于兼容层和训练流程的验证。

从该目录运行存档测试：

```powershell
pytest -q
```

较重的训练器验证使用了：

```powershell
python tests/test_all_trainers.py
```

该烟雾测试程序在主要工作流中验证了有界DirectML训练，并用于证明执行兼容性。

通过这些测试并不能证明训练模型的质量。这一区别成为了实验的主要教训。

---

<a id="known-directml-constraints"></a>
## 已知的 DirectML 限制

实验遇到了后端特定的限制和不支持的操作。

某些操作需要 CPU 降级或替代实现。一个已知的例子是与 `aten::lerp.Scalar_out` 相关联的降级。

详细发现保存在：

- [DirectML 限制](docs/directml_issues.zh-CN.md#current-directml-limitations)
- [DirectML 问题](docs/directml_issues.zh-CN.md)
- [DirectML 性能基准](docs/directml_benchmarks.zh-CN.md)
- [DirectML 审计](docs/directml_audit.zh-CN.md)

这些文档描述了历史上的 DirectML 环境，不应被解释为 ROCm 的限制。

---

<a id="archived-project-structure"></a>
## 归档项目结构

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

本地检查点和生成的产物可能因仓库配置而无法被 Git 跟踪。

---

<a id="documentation"></a>
## 文档

<a id="directml-compatibility"></a>
### DirectML 兼容性

- [DirectML 限制](docs/directml_issues.zh-CN.md#current-directml-limitations) — 不支持的操作、兼容性约束和 CPU 降级。
- [DirectML 问题](docs/directml_issues.zh-CN.md) — 开发过程中遇到的问题、调查和决策。
- [DirectML 性能基准](docs/directml_benchmarks.zh-CN.md) — 实验性的兼容性和性能结果。
- [DirectML 审计](docs/directml_audit.zh-CN.md) — 对 CUDA 特定代码和 DirectML 兼容性问题的审计。

<a id="historical-development-material"></a>
### 历史开发材料

- [开发工具](docs/development-tools.zh-CN.md) — 审计、测试用例生成、训练和性能基准工具。
- [训练命令](docs/training_commands.zh-CN.md) — 历史 DirectML 训练配置和恢复命令。
- [项目内存](docs/project_memory.zh-CN.md) — 实验中的架构决策和技术经验教训。
- [更新日志](docs/update_log.zh-CN.md) — DirectML 适配的按时间顺序记录。
- [路线图](docs/roadmap.zh-CN.md) — DirectML 开发阶段的历史路线图。

---

<a id="original-minimind-documentation"></a>
## 原始 MiniMind 文档

上游的 README 文件保存在本归档之外：

```text
../docs/original/
```

- [原始中文 README](../docs/original/README.md)
- [原始英文 README](../docs/original/README_en.md)

原始项目可在以下地址获取：

[MiniMind — jingyaogong/minimind](https://github.com/jingyaogong/minimind)

---

<a id="main-lesson"></a>
## 主要教训

DirectML 实验最重要的结果是方法论层面的：

```text
training runs
+ finite losses
+ successful checkpoints
≠ validated model training
```

后端验证也必须包括最终模型的行为。

官方的 MiniMind 检查点在 CPU 和 DirectML 上均通过了相同的推理路径，而本地训练的 DirectML 检查点则未能产生一致的输出。最终的 FP32 跨后端测试结果一致，而 DirectML FP16 则出现分歧。

因此，后续的训练工作转而使用 ROCm，而非增加更多的 DirectML 精度补救措施。

---

<a id="current-project"></a>
## 当前项目

有关活跃的 Windows + AMD 实现，请参见：

**[MiniMind ROCm Windows](../README.zh-CN.md)**

---

<a id="license"></a>
## 许可

本归档的适配版本仍属于 MiniMind 分支，并遵循仓库许可。

详见 [../LICENSE](../LICENSE)。
