[English](project_memory.md) | [Français](project_memory.fr.md) | [简体中文](project_memory.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="minimind-----project-memory"></a>
# MiniMind --- 项目记忆

> **历史 DirectML 归档 — 更新日期 2026-09-24**
>
> DirectML 已不再作为 MiniMind 训练的主后端。M5 最终验证发现了一种特定于测试的 DirectML FP16 路径的数值偏差。以下历史结果为可复现性和工程参考保留；ROCm 开发情况另作记录。

本文件提供了 MiniMind 对 DirectML 的适配过程以及最终决定停止使用 DirectML 进行训练并转向 ROCm 的简明技术记忆。

它记录了主要的架构决策、技术经验教训以及验证参考，这些内容在整个项目周期内都应保持实用。

详细的技术问题已在 [`directml_issues.md`](directml_issues.zh-CN.md) 中记录。

基准测试结果和性能实验已在 [`directml_benchmarks.md`](directml_benchmarks.zh-CN.md) 中记录。

当前限制及其解决方案已在 [`directml_issues.md`](directml_issues.zh-CN.md) 中总结。

开发进展在 [`update_log.md`](update_log.zh-CN.md) 中单独跟踪。

------------------------------------------------------------------------

<a id="architecture-decisions"></a>
# 架构决策

<a id="keep-directml-changes-close-to-upstream"></a>
## 将 DirectML 修改保持在上游接近位置

DirectML 支持应尽可能少地修改原始 MiniMind 训练逻辑。

应尽可能复用现有的 MiniMind 执行流程：

``` text
Model
  ↓
Standard PyTorch operations
  ↓
Resolved execution device
  ↓
CPU / CUDA / DirectML
```

应避免为 DirectML 创建现有训练流程的专用副本。

<a id="decision"></a>
### 决策

优先使用共享兼容性工具，而非单独的 DirectML 实现。

------------------------------------------------------------------------

<a id="separate-backend-compatibility-from-training-logic"></a>
## 将后端兼容性与训练逻辑分离

训练算法应尽可能保持与后端无关。

后端特定职责包括：

-   设备分辨率；
-   DirectML 初始化；
-   模型放置；
-   组件特定的设备要求；
-   后端兼容性规则。

这些职责应属于共享训练基础设施，而非各个训练算法本身。

<a id="decision-1"></a>
### 决策

各个训练器应专注于其训练算法本身。

后端兼容性应集中于共享训练工具中。

------------------------------------------------------------------------

<a id="allow-component-specific-device-placement"></a>
## 允许组件特定的设备放置

并非所有训练流程组件都必须在同一个设备上执行。

某些模型或操作在 DirectML 无法支持其完整推理路径时，可能需要在 CPU 上执行。

因此，当前的对齐训练配置可能采用：

``` text
Trainable model → DirectML
Reward model    → CPU
```

当需要时，数据将在不同设备之间传输。

<a id="decision-2"></a>
### 决策

正确性与稳定性优先于强制所有组件运行在 DirectML 上。

------------------------------------------------------------------------

<a id="preserve-minimind-checkpoint-semantics"></a>
## 保持 MiniMind 检查点语义

DirectML 仅作为执行后端，不得引入独立的检查点格式。

检查点兼容性取决于诸如：

``` text
Model architecture
Training stage
State dictionary structure
```

而非模型是通过 CPU、CUDA 或 DirectML 执行这一事实。

<a id="decision-3"></a>
### 决策

必须保持现有的 MiniMind 检查点语义不变。

后端兼容性和检查点兼容性应独立验证。

------------------------------------------------------------------------

<a id="important-technical-lessons"></a>
# 重要技术经验教训

<a id="directml-device-representation"></a>
## DirectML 设备表示

`torch-directml` 通过 PyTorch 的 `PrivateUse1` 后端集成了 DirectML。

因此，DirectML 设备在内部表现为：

    privateuseone:<index>

适配器索引在用户可见的 DirectML 设备和 PyTorch 内部表示之间保持一致。

例如：

    directml:0 → privateuseone:0
    directml:1 → privateuseone:1

这是预期行为。

用户可见的 MiniMind 设备选项保持为：

    directml
    directml:<index>

<a id="lesson"></a>
### 经验教训

不要将 `privateuseone` 视为一个独立的设备后端。

它是通过 `torch-directml` 选择的 DirectML 设备在 PyTorch 内部的表示。

当存在多个 GPU 时，设备索引是重要的，应在文档或比较运行时保留该索引。

------------------------------------------------------------------------

<a id="cpu-fallbacks-are-acceptable-when-necessary"></a>
## CPU 降级在必要时是可接受的

DirectML 不支持 MiniMind 及其依赖项中使用的所有 PyTorch 操作。

一些不支持的操作可以自动降级到 CPU。

其他组件可能需要明确保留在 CPU 上。

目前的示例包括：

``` text
AdamW unsupported operation → automatic CPU fallback
Reward model                → CPU execution
```

<a id="lesson-1"></a>
### 经验教训

CPU 降级并不自动意味着兼容性失败。

当正确性得以保留时，降级是可接受的，但其性能影响必须单独评估。

当前的降级情况记录在 [`directml_issues.md`](directml_issues.zh-CN.md) 中。

------------------------------------------------------------------------

<a id="physical-gpu-selection-must-be-explicit"></a>
## 必须明确选择物理 GPU

开发机器暴露了多个图形适配器，包括集成显卡和专用显卡。

两者都可能被 DirectML 识别。

因此：

``` text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

隐式选择 DirectML 设备会使基准测试结果难以解释，因为不同的运行可能在不同的物理 GPU 上执行。

<a id="lesson-2"></a>
### 经验教训

在训练或基准测试之前，应明确选择预期的物理 DirectML 适配器。

只有在使用相同物理 GPU 时，参考性能测量才可进行比较。

------------------------------------------------------------------------

<a id="benchmark-compatibility-is-not-full-training-stability"></a>
## 基准兼容性不等于完整训练稳定性

短兼容性基准测试有助于识别立即失败的配置。

然而：

``` text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

完整的训练运行持续时间更长，可能会遇到不同的内存、性能或后端条件。

<a id="lesson-3"></a>
### 经验教训

基准测试结果应被视为兼容性指标，而非性能建议。

最终的训练配置必须通过实际的实时数据训练运行进行验证。

实际预训练确认，某配置可在 DirectML 上成功执行，但由于性能或内存限制，仍不适合大规模训练。

直接ML验证应使用有界真实数据训练运行，并在全数据集执行不可行时使用`--max_steps`。

详细性能测量应包含在[`directml_benchmarks.md`](directml_benchmarks.zh-CN.md)中。

------------------------------------------------------------------------

<a id="training-measurement-can-introduce-major-synchronization-overhead"></a>
## 训练测量可能引入重大同步开销

最初报告的真实训练测量值约为：

``` text
5.280 s / iteration
1.52 samples / second
304.81 effective tokens / second
```

这最初表明预训练持续时间约为：

``` text
9.70 days / epoch
```

M4性能调查显示，测量路径本身在DirectML与CPU之间引入了显著的同步开销。

在移除关键训练和测量路径中不必要的同步后，相同的`8 × 340`参考工作负载结合梯度累积`8`完成了一个新的100步基准测试，结果为：

``` text
Average iteration time:      0.482 s
Samples / second:            16.61
Effective tokens / second:   3341.37
Padded tokens / second:      5647.46
Estimated epoch duration:    21.24 h
Estimated epoch duration:    0.89 days
```

在不减少模型架构、物理批大小、序列长度或梯度累积配置的情况下，迭代时间提升了约10.95倍。

<a id="lesson-4"></a>
### 经验教训

当性能监控强制在关键训练循环内进行设备到CPU的同步时，可能会显著扭曲DirectML的测量结果。

性能调查应隔离：

``` text
Model computation
        +
Required synchronization
        +
Optimizer overhead
        +
Logging / measurement overhead
```

在将意外差的运行时间归因于DirectML后端本身之前。

只有在确认测量路径未引入显著同步开销后，才应保留参考性能数值。

------------------------------------------------------------------------

<a id="regression-configuration"></a>
# 回归配置

为快速进行DirectML兼容性验证，使用了一个轻量级的MiniMind模型：

``` text
hidden_size = 128
num_hidden_layers = 2
```

该配置已被用于验证完整工作流：

``` text
Training
   ↓
Checkpoint
   ↓
Checkpoint loading
   ↓
Evaluation
   ↓
Text generation
```

<a id="decision-4"></a>
### 决策

保留`128 / 2-layer`配置作为轻量级DirectML回归配置。

该配置旨在用于兼容性验证，而非代表性的模型性能测试。

------------------------------------------------------------------------

<a id="validation-principles"></a>
# 验证原则

<a id="test-backend-compatibility-independently"></a>
## 独立测试后端兼容性

DirectML执行应独立于无关的模型或检查点问题进行验证。

不应自动将故障归因于DirectML。

相关检查包括：

``` text
Device placement
Model architecture
Checkpoint compatibility
Dataset validity
Backend operation support
```

------------------------------------------------------------------------

<a id="prefer-real-training-validation"></a>
## 优先使用真实训练验证

合成和短时测试对开发和回归测试很有用。

它们不能替代真实训练验证。

因此，验证策略结合了：

``` text
Unit tests
    +
Smoke tests
    +
Compatibility benchmarks
    +
Real training runs
```

每一层回答一个不同的兼容性问题。

------------------------------------------------------------------------

<a id="documentation-responsibilities"></a>
# 文档责任

项目文档按目的进行有意分离：

``` text
roadmap.md
    → planned milestones and remaining work

update_log.md
    → chronological development history

project_memory.md
    → durable architectural decisions and lessons

directml_issues.md
    → current limitations plus problems, causes, solutions, and decisions

directml_benchmarks.md
    → experimental compatibility, performance, and final M5 validation results

development-tools.md
    → development and validation utilities

directml_audit.md
    → automatically generated compatibility audit
```

随着项目的发展，应保持这种分离。

<a id="m4-final-decisions"></a>
# M4最终决策

<a id="directml-fp16-requires-explicit-numerical-handling"></a>
## DirectML FP16需要显式数值处理

最终的M4训练路径使用DirectML FP16并带有：

``` text
Static loss scale = 1024
AdamW epsilon     = 1e-4
```

默认的AdamW epsilon值`1e-8`在第一次优化器更新后立即导致训练产生非有限值。梯度检查显示在优化器步骤之前梯度是有限的，因此故障被定位在更新路径中。

<a id="lesson-5"></a>
### 经验教训

DirectML FP16 优化器设置必须独立于CUDA混合精度假设进行验证。

静态损失缩放和一个DirectML安全的AdamW epsilon被集中到共享训练工具中，而不是在每个训练算法中重复。

------------------------------------------------------------------------

<a id="sustained-validation-is-required-after-the-first-optimizer-step"></a>
## 首次优化器步骤之后必须持续进行验证

一个有界兼容性运行必须跨越优化器更新边界，才能提供有用的FP16验证。

因此最终使用的兼容性矩阵采用了一个包含9个步骤的有界运行，梯度累积为`8`。

为了实际可行性，该运行还补充了一个`1000`步的真实预训练运行。

持续参考运行成功完成，结果为：

``` text
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
dtype = float16
loss_scale = 1024
AdamW eps = 1e-4
```

<a id="lesson-6"></a>
### 经验教训

验证应区分：

``` text
Immediate execution
        ↓
First optimizer update
        ↓
Post-update execution
        ↓
Sustained training
```

仅通过第一阶段是不够的。

------------------------------------------------------------------------

<a id="fp16-improves-both-throughput-and-memory-headroom"></a>
## FP16同时提升了吞吐量和内存余量

最终FP16参考基准测量约为：

``` text
0.482 s / iteration
16.61 samples / second
3341.37 effective tokens / second
```

最终专用GPU内存监控测得峰值约为：

``` text
7,769.62 MB
```

相较于早期FP32测量中的约`11,623.12 MB`，减少了约`33.15%`。

<a id="decision-5"></a>
### 决策

针对测试硬件，实际的DirectML基准是FP16而非FP32。

------------------------------------------------------------------------

<a id="use-one-explicit-cross-trainer-smoke-runner"></a>
## 使用一个明确的跨训练器烟雾测试运行器

主要的DirectML FP16训练器使用有界`--max_steps`执行，并可通过以下方式依次验证：

``` text
tests/test_all_trainers.py
```

最终M4烟雾测试运行通过了：

``` text
Pretrain
Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

DPO仍属于之前已验证的流程，但未包含在本次M4 FP16烟雾测试运行结果中。

<a id="decision-6"></a>
### 决策

重型DirectML训练器烟雾验证使用Python显式运行，并与常规轻量级`pytest -q`工作流程分开。

------------------------------------------------------------------------

<a id="m4-practical-viability-conclusion--superseded"></a>
## M4实际可行性结论——已被取代

在M4结束时，测试的MiniMind DirectML配置从执行、吞吐量、内存和烟雾测试覆盖的角度来看被认为是实际可行的。但后续的完整训练模型质量验证取代了这一结论。

该结论基于修正后的吞吐量、降低的FP16内存使用、持续1000步的稳定性以及成功的跨训练器烟雾验证。

历史M4结论仅针对已验证的硬件和配置。它不能被解释为最终DirectML训练路径产生了可接受模型的证据。


------------------------------------------------------------------------

<a id="directml-moe-requires-a-compatibility-routing-path"></a>
## DirectML MoE需要一个兼容性路由路径

上游稀疏MoE路径依赖于路由操作，这些操作在测试的DirectML后端上触发了不支持的散列行为。

保留的架构是：

``` text
CPU / CUDA → original sparse MoE routing
DirectML   → scatter-free compatibility routing
```

DirectML路径计算所有专家，并通过选定的路由权重进行组合。

<a id="lesson-7"></a>
### 经验教训

当上游稀疏操作不被支持时，后端兼容性可能需要局部实现的降级方案。

该降级方案应仅限于DirectML，以确保上游CPU和CUDA行为得以保留。

------------------------------------------------------------------------

<a id="agent-rl-requires-windows-and-fp16-specific-compatibility-handling"></a>
## Agent RL需要Windows和FP16特定的兼容性处理

Windows DataLoader 工作线程需要在模块作用域内定义代理 `collate_fn`。

DirectML FP16 验证也表明，即使掩码的全序列策略/参考重新计算变为非有限，其 rollout 行为仍可保持有限。

对于已验证的右填充代理批次，问题性的全序列注意力掩码在策略/参考重新计算过程中被省略，而对数值敏感的对数概率和强化学习比率计算则使用 FP32。

<a id="lesson-8"></a>
### 经验教训

生成时的稳定性并不能保证全序列重新计算会遵循一条同样稳定的后端路径。

这两条路径应独立进行验证。

------------------------------------------------------------------------

<a id="final-cross-trainer-m4-validation"></a>
## 最终交叉训练器 M4 验证

最终显式 DirectML 烟雾测试程序完成：

``` text
All trainer smoke tests passed
Passed: 9/9
```

该测试程序覆盖：

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

DPO 仍保持先前验证状态，但不在此特定测试程序内。

<a id="decision-7"></a>
### 决策

`9/9` 的结果是最终 M4 交叉训练器烟雾参考。

------------------------------------------------------------------------

<a id="m5-training-execution"></a>
# M5 训练执行

完整的顺序工作流将上游训练器配置与持续的 DirectML 验证基准保持分离。

M4 验证的预训练基准保持为：

``` text
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
```

对于完整训练，`train_all.ps1` 使用在 [`training_commands.md`](training_commands.zh-CN.md) 中记录的训练器特定目标值。DirectML MoE 阶段保持保守的物理批次大小。

长训练阶段会根据在 [`training_commands.md`](training_commands.zh-CN.md) 中记录的训练器特定 `--save_interval` 值刷新其最新的恢复检查点。

中断阶段可以从该检查点继续执行：

``` text
--from_resume 1
```

<a id="decision-8"></a>
### 决策

上游训练器的默认配置不得被视为持续 DirectML 验证结果。长训练运行会保留周期性恢复检查点。

------------------------------------------------------------------------

<a id="dense-directml-pretraining-uses-fp32-master-weights"></a>
## 密集 DirectML 预训练使用 FP32 主权重

长运行检查点诊断显示，早期纯 FP16 DirectML 优化器路径虽然保持有限，却会产生退化的自我复制模型。

密集预训练现在保留 FP16 模型计算，但通过 FP32 主权重执行优化器更新：

``` text
FP16 compute
    ↓
FP32 gradient unscale
    ↓
AdamW FP32 update
    ↓
Copy master weights back to FP16 model
```

后续 DirectML 交叉熵调查表明，默认的均值归约在使用 `ignore_index=-100` 时错误地对填充批次进行了归一化。随后第一个显式的 `reduction="sum" / valid_tokens` 修复方案在测试的 DirectML FP16 路径上也被证明是错误的。

保留的实现方式按每个 token 计算 `reduction="none"` 损失，并在 FP32 中执行有效 token 的均值。在批次大小为 32 的参考测试中，模型损失 `6.30274916` 与 DirectML 每 token 计算结果完全一致，并且与 CPU FP32 参考值 `6.30530691` 保持在大约 `0.003` 的范围内。

早期第 100 步和第 1000 步运行仍具有有用的 FP32 主权重验证证据，但它们早于最终交叉熵归约修复。

<a id="decision-9"></a>
### 决策

有限的损失不足以证明长运行混合精度训练的正确性。

密集 DirectML 预训练在恢复检查点中保留 FP32 主权重，以确保优化器精度在中断期间得以保持。旧的没有主权重的恢复检查点不得与新路径一起使用。

DirectML因果语言模型的损失必须使用每token交叉熵，随后在有效非填充token上进行FP32平均。后端默认的均值归约以及测试过的FP16 `sum / valid_tokens`补救措施均未保留。



------------------------------------------------------------------------

<a id="intermediate-dense-pretraining-completed-epoch-1"></a>
## 中间密集预训练完成第1个周期

一个中间密集DirectML路径从头开始重启，使用上游预训练参数，包括物理`batch_size = 32`、FP32主权重以及初始`reduction="sum" / valid_tokens`损失补救措施。

第一个完整周期成功完成。

对第1周期模型进行检查点一致性测试确认：

``` text
Model loss:                  6.48799419
Manual token loss:           6.48799419
Difference:                  0.0000000000
Samples 0–255 global loss:   6.1017
Top-1 accuracy:              15.07%
Top-1 repeat rate:            0.83%
Mean entropy:                 5.3160
```

与修正后的第1000步检查点（`loss 6.7785`，Top-1 `6.76%`，重复`0.76%`）相比，第一个完整周期在不出现历史自复制崩溃的情况下，显著提升了下一个token的准确率。

后续针对特定减少方式的测试表明，该运行仍使用了错误的DirectML `reduction="sum" / valid_tokens`损失。因此该检查点仅保留用于调查，而非最终预训练基础。

<a id="decision-10"></a>
### 决策

最终密集预训练质量参考必须从头开始使用FP32主权重和保留的`reduction="none"` → FP32有效token均值损失来生成。

已完成的中间周期仍表明，物理`batch_size = 32`可通过当前内存路径完成一个完整周期；更早的OOM观察结果仍是来自更早训练路径的历史结果。

------------------------------------------------------------------------

<a id="final-directml-loss-path-completed-epoch-1"></a>
## 最终DirectML损失路径完成第1周期

保留的密集DirectML预训练路径使用FP16模型计算、FP32主权重、每token交叉熵（`reduction="none"`）和FP32有效token平均，从头完成了第一个完整周期。

两个独立的256样本诊断区域产生了训练/人工损失为`6.0612 / 6.0637`和`6.0177 / 6.0201`，Top-1准确率为`13.71%`和`13.80%`，重复率为`0.81%`和`0.71%`，平均熵为`5.3487`和`5.3436`。

模型及独立token损失在完成一个周期后仍保持在大约`0.0025`范围内，历史自复制崩溃未再次出现。

<a id="decision-11"></a>
### 决策

FP16计算 + FP32主权重 + 每token FP32有效均值路径现在是经过验证的完整周期密集DirectML预训练路径。

随后的同一修正运行完成了第2周期。在四个数据区域进行的4096样本更广泛评估中，训练/人工损失保持紧密一致，Top-1重复率介于`0.56%`和`0.96%`之间。定性生成检查也显示未出现历史重复token崩溃的回归。

<a id="final-decision"></a>
### 最终决策

已完成的第2周期检查点**不被保留作为全SFT密集预训练的基础**。尽管诊断数值上合理，但定性生成仍存在不连贯性。因此DirectML训练路径被放弃，下一训练工作转向ROCm。
------------------------------------------------------------------------

<a id="cuda-mixed-precision-and-directml-use-different-mechanisms"></a>
## CUDA混合精度与DirectML使用不同机制

上游CUDA预训练路径在保持模型参数为FP32的同时，使用BF16自动转换进行兼容性计算操作。

概念上：

``` text
Upstream CUDA:
FP32 model parameters
        ↓
BF16 autocast compute
        ↓
AdamW on FP32 parameters
```

当前DirectML栈不提供等效的已验证BF16自动转换路径。一个直接的BF16张量实验以`Invalid or unsupported data type BFloat16`失败。

因此保留的DirectML近似结果为：

``` text
DirectML:
FP16 model compute
        ↓
FP32 master parameters
        ↓
AdamW FP32 update
        ↓
copy FP32 master → FP16 model
```

<a id="lesson-9"></a>
### 经验教训

FP32主权重并非任意的额外副本。它们提供了FP32优化器状态/参数精度，CUDA通过将真实模型参数保留在FP32中并自动转换计算来自然获得这种精度。

------------------------------------------------------------------------

<a id="fast-direct-fp16-re-test-confirms-master-weights-are-required"></a>
## 快速Direct FP16重测确认主权重是必需的

一项新的上游类DirectML实验移除了FP32主权重，并直接将AdamW应用于FP16模型参数。

在样本`0–255`上的检查点诊断演进如下：

``` text
step 1000 → loss 9.2573, Top-1 0.31%, repeat 0.29%
step 2000 → loss 8.3102, Top-1 0.92%, repeat 1.20%
step 5000 → loss 7.3196, Top-1 1.80%, repeat 39.24%
```

真实数据的重复率保持在大约`0.29%`的水平。

因此损失的下降掩盖了预测行为的显著退化。这与之前长期证据一致，即直接FP16 AdamW更新在DirectML预训练中不可靠。

<a id="decision-12"></a>
### 决策

保留的Dense DirectML精度路径仍为：

``` text
FP16 forward / backward
        ↓
FP32 gradient conversion and unscale
        ↓
FP32 gradient clipping
        ↓
AdamW on FP32 master weights
        ↓
FP32 master → FP16 model synchronization
```

性能工作必须保留此架构，除非未来有独立验证的替代方案通过检查点诊断得到证实。

------------------------------------------------------------------------

<a id="limit-diagnostic-cpu-synchronization-to-logging-intervals"></a>
## 限制诊断CPU同步到日志间隔

FP32主权重和主权重到模型的同步是优化器路径中必需的部分，并保留在DirectML设备上。

相比之下，诸如：

``` text
loss.cpu().item()
grad_norm.cpu().item()
```

的标量诊断读取并非在每个训练步骤中都需要，且会强制异步DirectML执行与CPU之间的同步。

因此，优化后的训练器变体仅在`log_interval`边界上执行这些诊断读取，同时在每个优化器步骤中保持梯度裁剪和FP32主权重更新不变。

<a id="decision-13"></a>
### 决策

将`log_interval`作为CPU侧有限值检查和标量日志的正常周期。

一个`1000`步的对比测量显示：

``` text
Stable trainer:     2003.79 s total, 2.0038 s / step
Log-Sync trainer:   1930.75 s total, 1.9307 s / step
Speedup:             1.038×
Time reduction:      3.65%
```

因此，对于测试配置，吞吐量优势已得到验证。在日志同步训练器成为最终参考之前，仍需进行检查点质量验证。

------------------------------------------------------------------------

<a id="m5-final-diagnosis-and-backend-decision--2026-09-24"></a>
# M5最终诊断与后端决策 — 2026-09-24

最终证据将检查点质量与后端执行分开。

在相同的`1,000`评估样本（`204,327`有效标记）上，使用ROCm时：

```text
DirectML-trained checkpoint: loss 6.880233, perplexity 972.8529
ROCm-trained checkpoint:     loss 1.846587, perplexity   6.3382
```

相同的DirectML训练检查点也暴露出一个与精度相关的后端差异：

```text
same batch, 873 valid tokens

DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML FP16未使用SDPA产生NaN对数。在FP32中，DirectML和ROCm的值大约为`1e-6`。

<a id="final-lesson"></a>
### 最终经验教训

有限训练损失、检查点保存/加载成功以及甚至内部一致的损失计算均不足以验证后端。生成的检查点必须独立评估，当数值行为存疑时，必须直接测试混合精度后端等效性。

<a id="final-decision-1"></a>
### 最终决策

- 停止使用DirectML进行MiniMind训练。
- 将剩余障碍归因于测试的DirectML FP16数值路径，而非检查点加载、分词器、数据集或通用推理。
- 将DirectML代码和证据作为已完成的可行性研究予以保留。
- 关闭DirectML训练研究；未来ROCm工作将单独跟踪。
