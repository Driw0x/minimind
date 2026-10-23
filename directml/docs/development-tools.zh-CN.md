[English](development-tools.md) | [Français](development-tools.fr.md) | [简体中文](development-tools.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="development-tools"></a>
# 开发工具

> **历史 DirectML 归档 — 更新日期：2026-09-24**
>
> DirectML 已不再作为 MiniMind 的训练后端。M5 最终验证发现了一个特定于测试的 DirectML FP16 路径的数值差异。以下历史结果为可重现性和工程参考保留；ROCm 开发情况另作记录。

本文件描述了为支持和验证历史 DirectML 版本的 MiniMind 而添加的开发工具。这些工具仍可用于回归测试、故障诊断以及记录 DirectML 可行性研究，但新的训练工作现已转向 ROCm。

<a id="directml-compatibility-audit"></a>
## DirectML 兼容性审计

DirectML 审计会扫描项目代码，以检测可能依赖 CUDA 特定行为的部分。

从项目根目录运行：

``` bash
python scripts/audit_directml.py
```

生成的报告保存在：

``` text
docs/directml_audit.md
```

审计检测到的潜在兼容性问题包括：

-   CUDA 特定引用；
-   设备放置假设；
-   CUDA 工具；
-   自动混合精度（AMP）；
-   分布式训练代码。

该报告旨在作为静态分析辅助工具。报告中发现的项并不一定表示存在不兼容情况，必须手动审查。

------------------------------------------------------------------------

<a id="test-fixture-generation"></a>
## 测试 fixture 生成

自动化测试套件使用小型确定性数据集来验证 MiniMind 训练阶段，而无需使用完整的训练数据集。

使用以下命令生成 fixture：

``` bash
python scripts/generate_test_fixtures.py
```

生成的数据集存储在：

``` text
tests/fixtures/
```

它们为测试训练流程提供了轻量级输入，例如：

-   预训练；
-   监督微调；
-   DPO；
-   代理强化学习；
-   其他支持的训练阶段。

这些 fixture 仅用于兼容性与烟雾测试，不适用于实际模型训练。

------------------------------------------------------------------------

<a id="sequential-training"></a>
## 顺序训练

`train_all.ps1` PowerShell 脚本在 Windows 上依次运行 MiniMind 的主要训练阶段。

从项目根目录运行该脚本：

``` powershell
.\scripts\train_all.ps1
```

该脚本依次执行每个训练阶段。一个新阶段的启动仅在前一个阶段成功完成后才进行。

这避免了同时运行多个 DirectML 训练进程，从而可能增加 GPU 内存使用并降低稳定性。

如果某个训练阶段失败，脚本将立即停止，而不是继续执行剩余阶段。

该脚本保持标准的 MiniMind 输出结构：

``` text
out/            # Generated model weights
checkpoints/    # Training resume checkpoints
```

该工具旨在简化在 Windows 上对 DirectML 训练流水线的顺序验证和执行。目前仅保留用于历史重现和回归测试；它不再推荐作为完整的训练路径。

`train_all.ps1` 使用各训练器特定的批处理大小、序列长度和梯度累积设置，而非对每个阶段应用一个共享配置。检查点间隔为训练器特定，详情见 `training_commands.md`；短验证运行可能会有意覆盖这些设置。

参见 [`training_commands.md`](training_commands.zh-CN.md) 获取完整的每训练器配置表以及 `--from_resume 1` 恢复命令。

------------------------------------------------------------------------

<a id="real-training-benchmark"></a>
## 真实训练基准

`benchmark_training.py` 工具使用实际的预训练数据集而非合成兼容性工作负载来测量真实 MiniMind 预训练性能。

从项目根目录运行：

``` bash
python scripts/benchmark_training.py
```

一个典型的 DirectML 参考运行可以通过以下命令启动：

``` powershell
python scripts/benchmark_training.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --warmup_steps 8 `
    --steps 100 `
    --model_dtype float16
```

该基准执行了真实预训练步骤中的主要操作：

-   数据集加载；
-   前向传播；
-   损失计算；
-   反向传播；
-   梯度累积；
-   梯度裁剪；
-   AdamW 优化器步骤。

warmup 迭代被排除在测量结果之外，以避免初始化开销对稳态性能造成扭曲。

该工具报告的指标包括：

``` text
Average iteration time
Samples / second
Effective tokens / second
Padded tokens / second
Estimated steps / epoch
Estimated epoch duration
```

本基准旨在评估 DirectML 训练的实用吞吐量和持续执行行为。它补充了较短的兼容性基准：通过短兼容性测试的配置并不一定在长时间真实数据训练中保持稳定。

------------------------------------------------------------------------

<a id="log-sync-trainer-benchmark"></a>
## 日志同步训练基准

`benchmark_log_sync.ps1` 工具在相同受限工作负载下比较稳定密集预训练训练器与日志同步变体。

从项目根目录运行：

``` powershell
.\scripts\benchmark_log_sync.ps1
```

当前 1000 步的对比结果为：

``` text
Stable:    2.0038 s / step
Log-Sync:  1.9307 s / step
Speedup:   1.038×
Reduction: 3.65%
```

该基准验证了吞吐量的提升。在 Log-Sync 成为最终训练参考之前，仍需进行检查点质量验证。

------------------------------------------------------------------------

<a id="bounded-training-runs"></a>
## 受限训练运行

DirectML 兼容的模型训练器通过 `--max_steps` 参数支持受限运行。

这使得训练流程可以在不完整一个 epoch 的情况下进行固定迭代数的训练。共享的受限运行行为被用于预训练、完整 SFT、LoRA、蒸馏、GRPO、代理 RL 和 PPO 等场景。

示例：

``` powershell
python trainer/train_pretrain.py `
    --device directml:1 `
    --batch_size 8 `
    --max_seq_len 340 `
    --accumulation_steps 8 `
    --max_steps 100
```

受限运行在以下方面非常有用：

-   验证真实数据训练的稳定性；
-   比较批大小和序列长度配置；
-   重现 DirectML 内存故障；
-   研究 CPU 降级行为；
-   在不承诺完整训练 epoch 的情况下运行更长的稳定性检查。

对于 DirectML 性能工作，受限真实数据运行应与 `benchmark_training.py` 一起使用：训练器验证实际训练流程，而基准提供更聚焦的吞吐量测量。

<a id="directml-loss-validation-tests"></a>
## DirectML 损失验证测试

针对 DirectML 交叉熵调查的定向回归工具存储在 `tests/` 中：

``` text
test_directml_cross_entropy.py
test_directml_loss_reductions.py
test_batch_loss_consistency.py
test_train_diagnostic_loss.py
```

它们验证了 DirectML 的 `cross_entropy` 减少行为，包括保留的每 token `reduction="none"` → FP32 有效 token 均值路径，并将训练路径损失与 token 归一化诊断损失进行比较。这些测试与 `scripts/` 保持分离，而 `scripts/diagnose_pretrain.py` 仍为检查点分析工具。

------------------------------------------------------------------------

<a id="trainer-smoke-test-runner"></a>
## 训练器烟雾测试运行器

`tests/test_all_trainers.py` 标识符通过单个 Python 入口点依次执行 DirectML FP16 训练器烟雾测试。

从项目根目录显式运行它：

``` bash
python tests/test_all_trainers.py
```

当前运行器验证：

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

每个训练器通过 `--max_steps` 使用受限配置。测试运行器从 `trainer/` 工作目录启动每个训练器，因此标准的 MiniMind 相对路径保持有效。

烟雾测试在首次失败时停止，并报告已完成的训练器数量。该完整的 M4 验证运行对本套包含的所有训练器均已通过：

``` text
Passed: 9/9
```

该运行器故意与正常的 `pytest -q` 测试套件分开执行，因为它启动了真实的 DirectML 训练工作负载，其负载显著高于单元测试和轻量级回归测试。

------------------------------------------------------------------------

<a id="m5-backend-comparison-utilities"></a>
## M5 后端对比工具

M5 最终 DirectML 验证使用了三个小型跨后端工具：

```text
scripts/compare_pretrain_logs.py
    -> plots training loss, learning rate, ETA and relative loss reduction

scripts/compare_checkpoint_loss.py
    -> evaluates multiple checkpoints on the same fixed dataset subset

scripts/compare_single_batch_backend.py
    -> compares the same checkpoint and fixed batch across DirectML/ROCm
```

这些工具将训练日志行为与检查点质量及后端数值行为分离。它们仅用于关闭 M5 DirectML 诊断，且优先于仅通过训练损失判断后端正确性。
