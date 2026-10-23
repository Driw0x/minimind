[English](training_commands.md) | [Français](training_commands.fr.md) | [简体中文](training_commands.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="minimind-directml-----historical-training-commands-and-checkpoint-resume"></a>
# MiniMind DirectML --- 历史训练命令与检查点恢复

> **历史 DirectML 归档 — 更新日期：2026-09-24**
>
> DirectML 已不再作为 MiniMind 训练的主后端。M5 最终验证发现了一种特定于测试的 DirectML FP16 路径的数值偏差。以下历史结果为可复现性和工程参考保留；ROCm 开发情况另作文档说明。

本文件集中了可行性研究期间使用的 DirectML 训练命令，并解释了在该分支上自动检查点和训练恢复的工作原理。这些命令仅保留用于复现，**并非推荐用于新的完整训练**。

<a id="current-recommendation"></a>
## 当前推荐

请勿使用以下 DirectML 命令启动新的完整 MiniMind 训练运行。M5 最终检查点评估确认存在显著质量差距：在相同的 `204,327` 令牌上，DirectML 训练的检查点得分 `6.880233` 损失 / `972.8529` 混合度，而 ROCm 训练的检查点得分为 `1.846587` / `6.3382`。

这些命令仅保留用于历史复现。DirectML 路线图在 M5 版本已完成；任何 ROCm 命令均属于独立的活跃 ROCm 文档，而非本归档内容。

<a id="trainer-configurations"></a>
## 训练器配置

以下数值是**上游 MiniMind 训练器提出的默认值**。它们被用作历史顺序 DirectML 工作流中的目标配置。

|训练器|批处理大小|最大序列长度|累积步数|
| --- | ---: | ---: | ---: |
|密集预训练| 32 | 340 | 8 |
|密集全量 SFT| 16 | 768 | 1 |
|LoRA| 32 | 340 | 1 |
|DPO| 4 | 1024 | 1 |
|蒸馏| 32 | 340 | 1 |
|GRPO| 2 | 768 | 1 |
|PPO| 2 | 768 | 1 |
|代理强化学习| 2 | 1024 | 1 |

DirectML 特定的数值设置保持如下：

``` text
Device: directml:1
DType: float16
DirectML static loss scale: 1024
torch.compile: disabled
```

对于 DirectML FP16 训练，项目额外使用：

- FP32 主权重用于优化器更新；
- 按令牌交叉熵（`reduction="none"`）后，再进行 FP32 有效令牌平均处理。

这些调整避免了在长时间验证中发现的 FP16 优化器不稳定性和 DirectML 交叉熵错误减少问题。

密集预训练还使用 FP32 主权重与 AdamW `eps = 1e-8`。其他训练器特定的精度路径保持不变，除非有明确文档说明。

<a id="historical-m4-directml-baseline"></a>
### 历史 M4 DirectML 基线

持续的 M4 验证基线为：

``` text
Trainer: Dense Pretrain
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
loss_scale = 1024
AdamW eps = 1e-4
```

该配置仍可作为历史 M4 验证证据使用。

<a id="current-m5-dense-reference"></a>
### 当前 M5 密集参考

保留的密集预训练路径使用：

``` text
batch_size = 32
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
optimizer weights = float32 master weights
AdamW eps = 1e-8
loss = per-token cross-entropy → FP32 valid-token mean
```

该修正路径成功完成了全部预训练周期。

表中较大的训练器值仍作为独立验证目标，用于在 DirectML 上进行验证。

<a id="moe-directml-override"></a>
### MoE DirectML 重写

上游预训练和全量 SFT 无论在密集模式还是 MoE 模式下，均暴露相同的批处理默认值。对于该 DirectML 分支，MoE 故意保持保守的物理批处理大小，因为散列自由兼容路径会评估所有专家：

|MoE 训练器|DirectML 批处理大小|最大序列长度|累积|
| --- | ---: | ---: | ---: |
|MoE 预训练| 1 | 340 | 64 |
|MoE 完整 SFT| 1 | 768 | 64 |

这些 MoE 值是 DirectML 安全覆盖选项，而非上游默认值。

当前上游的预训练和完整 SFT 都暴露了 `--seed 42`。当前该项目使用的 DirectML 分支可能尚未暴露该 CLI 选项，因此以下命令不会强制启用 `--seed`；请保持分支当前的种子行为，直到本地 argparse 与上游同步。

从以下位置运行单个训练器命令：

``` powershell
cd trainer
```

从仓库根目录启动完整顺序流程的命令为：

``` powershell
.\scripts\train_all.ps1
```

<a id="checkpoint-behavior"></a>
## 检查点行为

以下每个命令在未被短时 DirectML 验证运行明确覆盖的情况下，均使用当前上游 `save_interval` 的默认设置。

训练器会在配置的间隔内以及每个周期的正常结束时写入一个恢复检查点。

会产生两类文件：

``` text
out/
    # Model weights intended for later stages / inference.

checkpoints/
    # Complete resume state used to continue interrupted training.
```

例如，密集预训练会产生：

``` text
out/pretrain_768.pth
checkpoints/pretrain_768.pth
checkpoints/pretrain_768_resume.pth
```

`_resume.pth` 文件是恢复训练的重要文件。它包含模型状态、优化器状态、周期、步数、世界规模以及适用的缩放器状态。配备调度器的训练器还会存储其调度器状态。PPO 还会存储批评者及其优化器/调度器状态。

当前的检查点实现为每个训练阶段保留**最新的检查点**。新的保存会覆盖相同权重名称的前一个检查点。因此，恢复距离大致由配置的 `save_interval` 决定；当前实现不会为每个检查点创建永久归档。

<a id="resume-an-interrupted-training"></a>
## 恢复中断的训练

要恢复，请重新运行**相同的训练命令**并添加：

``` text
--from_resume 1
```

在恢复时，请不要更改架构参数。特别是要保持相同的隐藏尺寸、层数、Dense/MoE 设置以及训练阶段。

训练器会自动在 `checkpoints/` 中查找匹配文件，恢复保存的状态，跳过当前周期中已完成的批次，并从下一个步骤继续。

示例：

``` text
Checkpoint saved at step 1200
Training interrupted at step 1267

Resume:
    --from_resume 1

The trainer reloads step 1200 and continues from step 1201.
The work from steps 1201–1267 must be recomputed.
```

------------------------------------------------------------------------

<a id="1-dense-pretrain"></a>
# 1. 密集预训练

完整训练：

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 32 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 8 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --from_weight none `
  --use_compile 0
```

恢复：

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 32 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 8 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --from_weight none `
  --from_resume 1 `
  --use_compile 0
```

恢复检查点：

``` text
checkpoints/pretrain_768_resume.pth
```

------------------------------------------------------------------------

<a id="2-dense-full-sft"></a>
# 2. 密集完整 SFT

需要 `out/pretrain_768.pth`。

完整训练：

``` powershell
python train_full_sft.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 16 `
  --learning_rate 1e-5 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight pretrain `
  --use_compile 0
```

恢复：使用相同的命令并添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/full_sft_768_resume.pth
```

------------------------------------------------------------------------

<a id="3-moe-pretrain"></a>
# 3. MoE 预训练

MoE 分支使用顺序训练脚本中采用的保守微批次配置。

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 1 `
  --learning_rate 5e-4 `
  --num_workers 8 `
  --accumulation_steps 64 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight none `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/pretrain_768_moe_resume.pth
```

------------------------------------------------------------------------

<a id="4-moe-full-sft"></a>
# 4. MoE 完整 SFT

需要 `out/pretrain_768_moe.pth`。

``` powershell
python train_full_sft.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 2 `
  --batch_size 1 `
  --learning_rate 1e-5 `
  --num_workers 8 `
  --accumulation_steps 64 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --use_moe 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight pretrain `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/full_sft_768_moe_resume.pth
```

------------------------------------------------------------------------

<a id="5-lora"></a>
# 5. LoRA

默认使用密集完整 SFT 模型。

``` powershell
python train_lora.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 10 `
  --batch_size 32 `
  --learning_rate 1e-4 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 10 `
  --save_interval 1000 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 340 `
  --use_moe 0 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

在默认 `--lora_name lora_medical` 下，恢复检查点为：

``` text
checkpoints/lora_medical_768_resume.pth
```

如果 `--lora_name` 发生更改，检查点名称也会相应更改。

------------------------------------------------------------------------

<a id="6-dpo"></a>
# 6. DPO

顺序流程将 DPO 保留在密集完整 SFT 分支上。

``` powershell
python train_dpo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 4 `
  --learning_rate 4e-8 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 100 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 1024 `
  --use_moe 0 `
  --beta 0.15 `
  --from_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

**重要：** DPO 位于最终 9/9 DirectML 训练器烟雾运行器之外。在依赖其进行长时间 DirectML FP16 运行之前，应重新检查其当前训练器。

------------------------------------------------------------------------

<a id="7-distillation-----dense-student--moe-teacher"></a>
# 7. 模型蒸馏 --- 密集学生 ← MoE 教师

要求：

``` text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

命令：

``` powershell
python train_distillation.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 6 `
  --batch_size 32 `
  --learning_rate 5e-6 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 100 `
  --save_interval 100 `
  --max_seq_len 340 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --student_hidden_size 768 `
  --student_num_layers 8 `
  --student_use_moe 0 `
  --from_student_weight full_sft `
  --teacher_hidden_size 768 `
  --teacher_num_layers 8 `
  --teacher_use_moe 1 `
  --from_teacher_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

使用默认的 `--save_weight full_dist` 时，恢复检查点为：

``` text
checkpoints/full_dist_768_resume.pth
```

------------------------------------------------------------------------

<a id="8-grpo"></a>
# 8. GRPO

``` powershell
python train_grpo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --max_gen_len 1024 `
  --use_moe 0 `
  --num_generations 6 `
  --beta 0.1 `
  --loss_type cispo `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/grpo_768_resume.pth
```

恢复状态也包含调度器状态。

------------------------------------------------------------------------

<a id="9-ppo"></a>
# 9. PPO

``` powershell
python train_ppo.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --critic_learning_rate 5e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 768 `
  --max_gen_len 1024 `
  --use_moe 0 `
  --mini_batch_size 2 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/ppo_actor_768_resume.pth
```

PPO 会恢复演员和批评家的训练状态。

------------------------------------------------------------------------

<a id="10-agent-rl"></a>
# 10. 自主体强化学习

``` powershell
python train_agent.py `
  --device directml:1 `
  --dtype float16 `
  --epochs 1 `
  --batch_size 2 `
  --learning_rate 3e-7 `
  --num_workers 8 `
  --accumulation_steps 1 `
  --grad_clip 1.0 `
  --log_interval 1 `
  --save_interval 10 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_seq_len 1024 `
  --max_gen_len 768 `
  --max_total_len 2500 `
  --use_moe 0 `
  --num_generations 4 `
  --beta 0.1 `
  --loss_type cispo `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --from_weight full_sft `
  --use_compile 0
```

恢复：添加：

``` text
--from_resume 1
```

恢复检查点：

``` text
checkpoints/agent_768_resume.pth
```

调度器状态作为恢复检查点的一部分被恢复。

------------------------------------------------------------------------

<a id="full-sequential-training"></a>
# 完整顺序训练

从仓库根目录开始：

``` powershell
.\scripts\train_all.ps1
```

顺序脚本应保留每个训练器的预期检查点间隔，除非 DirectML 验证运行特意覆盖该间隔。

如果某个阶段被中断，**不要**立即重启整个流程。应直接使用上述命令和 `--from_resume 1` 恢复该中断阶段。一旦该阶段完成，后续阶段可以单独启动，或者将顺序脚本调整为从下一个阶段开始启动。

<a id="quick-resume-rule"></a>
## 快速恢复规则

``` text
Normal start:
    --from_resume 0   (default)

After interruption:
    same command
    + --from_resume 1
```

在恢复之前，需验证预期的 `_resume.pth` 文件是否存在于 `checkpoints/` 中。

示例：

``` powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

当从 `trainer/` 运行时，或：

``` powershell
Get-ChildItem .\checkpoints\*_resume.pth
```

当从仓库根目录运行时。

<a id="important-notes"></a>
## 重要说明

-   `out/*.pth` 包含模型权重；仅恢复优化器进度是不够的。
-   `checkpoints/*_resume.pth` 是 `--from_resume 1` 使用的文件。
-   当希望恢复时，请保留 `out/` 和 `checkpoints/`。
-   在训练中断后，不要清理检查点目录。
-   恢复时应使用相同的架构和阶段配置。
-   每次保存都会替换检查点；当前实现不保留每 100 步保存的编号历史记录。
