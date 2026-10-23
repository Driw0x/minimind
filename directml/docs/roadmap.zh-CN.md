[English](roadmap.md) | [Français](roadmap.fr.md) | [简体中文](roadmap.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="minimind-backend-roadmap"></a>
# MiniMind 后端路线图

> **历史 DirectML 归档 — 更新日期：2026-09--24**
>
> DirectML 已不再作为 MiniMind 训练的主后端。M5 最终验证发现了一个特定于测试的 DirectML FP16 路径的数值偏差问题。以下历史结果为可复现性和工程参考保留；ROCm 开发情况另作记录。

本文件跟踪了 DirectML 评估的完成情况以及 MiniMind 训练向 ROCm 的过渡过程。

该项目区分了 **功能兼容性** 与 **实际可用性**。一个训练阶段可能在 DirectML 上正确执行，但由于性能限制、CPU 降级、内存约束或后端开销，仍然不适用于实际场景的训练。

------------------------------------------------------------------------

<a id="m1-----directml-foundation-"></a>
## M1 --- DirectML 基础构建 ✅

建立并验证基本的 DirectML 环境。

-   [x] 配置 Windows + DirectML 环境
-   [x] 将原本针对 CUDA 的 PyTorch 依赖项替换为 `torch-directml`
-   [x] 初始化 DirectML 设备
-   [x] 将 MiniMind 模型和张量移动到 DirectML
-   [x] 验证前向传播
-   [x] 验证损失计算
-   [x] 验证反向传播
-   [x] 验证 AdamW 优化器步骤
-   [x] 添加使用 Pytest 的 DirectML 兼容性测试
-   [x] 验证从干净虚拟环境安装的正确性

------------------------------------------------------------------------

<a id="m2-----minimind-training-compatibility-"></a>
## M2 --- MiniMind 训练流程兼容性 ✅

调整 MiniMind 训练流程，使其训练操作可在 DirectML 上执行。

-   [x] 检查训练流程中与 CUDA 相关的代码
-   [x] 引入支持 DirectML 的设备处理机制
-   [x] 在必要时调整混合精度和 CUDA 特定工具
-   [x] 在 DirectML 上运行一个最小预训练任务
-   [x] 验证训练损失的正确演化
-   [x] 保存一个模型检查点
-   [x] 成功重新加载检查点
-   [x] 使用训练后的检查点进行推理

> M2 验证的是 **功能兼容性**，而非实际训练性能。

------------------------------------------------------------------------

<a id="m3-----full-training-pipeline-compatibility-"></a>
## M3 --- 完整训练流程兼容性 ✅

验证 MiniMind 主要训练阶段在 DirectML 上的兼容性。

-   [x] 验证预训练
-   [x] 验证监督微调（SFT）
-   [x] 验证 LoRA 训练
-   [x] 验证 MiniMind 支持的其他训练阶段
-   [x] 验证训练阶段之间的检查点兼容性
-   [x] 记录不支持或部分支持的 DirectML 操作
-   [x] 验证训练阶段间的设备一致性

> 训练流程在功能上与 DirectML 兼容，但这并不保证大规模训练在实际中是可用的。

------------------------------------------------------------------------

<a id="m4-----directml-viability--performance-investigation-"></a>
## M4 --- DirectML 可用性与性能调查 ✅

判断 DirectML 是否适用于 MiniMind 训练，并识别主要的性能瓶颈。

<a id="baseline-benchmarking"></a>
### 基准测试

-   [x] 对支持的批量大小/序列长度组合进行基准测试
-   [x] 识别因内存或后端限制而失败的配置
-   [x] 确认合成兼容性基准测试并不必然代表真实的训练性能

<a id="real-training-performance"></a>
### 真实训练性能

-   [x] 测量真实的预训练吞吐量
-   [x] 测量迭代时间以及每秒处理的样本/令牌数量
-   [x] 测量真实训练过程中GPU内存的使用情况
-   [x] 将合成基准测试结果与真实数据集训练进行对比
-   [x] 估算合理的训练时长

<a id="directml-bottleneck-investigation"></a>
### DirectML瓶颈调查

-   [x] 识别CPU降级操作
-   [x] 测量CPU降级操作的影响
-   [x] 调查DirectML的同步与执行开销
-   [x] 识别导致训练异常缓慢的操作
-   [x] 确定瓶颈是否可以缓解

<a id="practical-viability"></a>
### 实际可行性

-   [x] 测试优化的批大小/序列长度配置
-   [x] 在性能允许的情况下测试更长的训练运行
-   [x] 评估长时间训练过程中的稳定性
-   [x] 确定DirectML训练的实际可行性
-   [x] 记录已知的性能限制
-   [x] 验证Dense和MoE的DirectML训练路径
-   [x] 验证完整的跨训练器烟雾测试（`9/9`）

> M4作为一项历史DirectML兼容性/性能研究已完整结束。其执行、吞吐量、内存使用和烟雾测试结果仍作为工程证据有效，但后续的完整训练质量检查显示，本地训练的DirectML检查点不可接受。因此，M4的实际可行性结论已被取代。
>
> 该M4基线被保留作为历史验证证据。M5在长期检查点质量研究后，将直接FP16参数更新替换为FP16计算和FP32主权重优化。

------------------------------------------------------------------------

<a id="m5-----directml-final-evaluation-"></a>
## M5 --- DirectML最终评估 ✅

通过完整的训练模型质量验证，最终完成DirectML调查，并确定DirectML是否应继续作为MiniMind训练后端。

<a id="finalization-work"></a>
### 最终化工作

-   [x] 应用已识别的性能优化
-   [x] 文档化支持的DirectML训练工作流
-   [x] 文档化定期检查点和恢复工作流
-   [x] 修正DirectML交叉熵归一化中对填充的忽略问题
-   [x] 验证FP16 + FP32主预训练通过两个完整周期
-   [x] 拒绝直接FP16 AdamW通过长期检查点诊断
-   [x] 基准测试减少的诊断CPU同步
-   [x] 对最终DirectML检查点进行定性评估
-   [x] 在相同评估集上对比DirectML与ROCm训练的检查点
-   [x] 重新评估DirectML检查点在ROCm下的表现
-   [x] 在FP16和FP32下对比相同DirectML检查点在DirectML与ROCm下的表现
-   [x] 将剩余瓶颈隔离到测试的DirectML FP16数值路径

<a id="final-evidence"></a>
### 最终证据

在相同的 `1,000` 样本（`204,327` 有效标记）上，基于 ROCm 进行评估：

```text
DirectML-trained: loss 6.880233, perplexity 972.8529
ROCm-trained:     loss 1.846587, perplexity   6.3382
```

使用相同的 DirectML 训练检查点和一个固定批次（`873` 有效标记）：

```text
DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML FP16 在未使用 SDPA 的情况下生成了非有限对数概率，而 DirectML 和 ROCm 在 FP32 上的数值结果则与 `1e-6` 相匹配。

<a id="final-outcome"></a>
### 最终结果

-   [x] DirectML 执行兼容性已记录
-   [x] DirectML 性能和限制已记录
-   [x] 全量训练检查点质量被拒绝
-   [x] 剩余障碍被归类为 DirectML FP16 数值发散
-   [x] 本项目中的 DirectML 训练已终止
-   [x] DirectML 代码、测试和证据被保留作为已完成的可行性研究

> 本路线图有意在 M5 处停止。ROCm 的开发在 DirectML 归档之外的另一份路线图中进行跟踪。

------------------------------------------------------------------------

<a id="project-goal"></a>
## 项目目标

DirectML 项目评估了 MiniMind 是否能够在 Windows 系统上通过 `torch-directml` 在 AMD Radeon GPU 上可靠地完成训练。

该研究已完成。DirectML 能够执行训练流程并通过广泛的兼容性、性能和烟雾测试，但最终训练的检查点未能达到可接受的模型质量。跨后端验证将剩余失败归因于测试中 DirectML FP16 数值路径。

因此，最终成果是保留的可行性研究，而非一个活跃的 DirectML 训练后端。
