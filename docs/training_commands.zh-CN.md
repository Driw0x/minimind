[English](training_commands.md) | [Français](training_commands.fr.md) | [简体中文](training_commands.zh-CN.md)

<a id="minimind-rocm--training-commands"></a>
# MiniMind ROCm — 训练命令

环境配置以及ROCm/PyTorch设备语义在根目录 [`README.md`](../README.zh-CN.md) 中有文档说明。

训练顺序、迷你版与完整版规则以及验证里程碑在 [`roadmap.md`](roadmap.zh-CN.md) 中有文档说明。

从以下位置运行训练命令：

```powershell
cd trainer
```

<a id="gpu-selection-with-integrated-graphics"></a>
### 使用集成显卡选择GPU

在CPU也提供集成AMD显卡的系统上，ROCm/PyTorch可能会将集成显卡暴露为`cuda:0`，而专用Radeon显卡暴露为另一个设备。

开始训练前，请检查检测到的设备：

```powershell
python -c "import torch; print('device_count:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

示例：

```text
0 AMD Radeon(TM) Graphics
1 AMD Radeon RX 7800 XT
```

如果专用GPU是设备`1`，则在启动MiniMind前选择它：

```powershell
$env:HIP_VISIBLE_DEVICES="1"
```

然后验证可见设备：

```powershell
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

预期结果：

```text
1
AMD Radeon RX 7800 XT
```

在`HIP_VISIBLE_DEVICES`过滤设备后，所选的专用GPU在Python进程中变为`cuda:0`。因此，当该环境变量生效时，训练命令应继续使用`--device cuda:0`。

`HIP_VISIBLE_DEVICES`适用于当前PowerShell会话。如果否则会选中集成显卡，则在打开新终端时需再次设置它。

---

<a id="upstream-datasets"></a>
## 上游数据集

官方MiniMind训练数据集由上游项目提供：

- [ModelScope — minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [Hugging Face — minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

仅需下载并放置所需的文件到：

```text
dataset/
```

主要数据集：

```text
pretrain_t2t_mini.jsonl   # Mini Pretrain
pretrain_t2t.jsonl        # Full Pretrain

sft_t2t_mini.jsonl        # Mini SFT
sft_t2t.jsonl             # Full SFT

dpo.jsonl                 # DPO
rlaif.jsonl               # GRPO / PPO
agent_rl.jsonl            # Agent RL
agent_rl_math.jsonl       # Agent RL math
```

为快速验证，使用`*_mini.jsonl`数据集。

为完整训练，使用`pretrain_t2t.jsonl`和`sft_t2t.jsonl`。

---

<a id="1-dense-pretrain"></a>
## 1. 密集预训练

默认数据集：

```text
../dataset/pretrain_t2t_mini.jsonl
```

<a id="mini-validation"></a>
### 迷你验证

```powershell
python train_pretrain.py --save_weight pretrain_mini
```

<a id="full-training"></a>
### 完整训练

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl
```

最终输出：

```text
out/pretrain_mini_768.pth
out/pretrain_768.pth
```

---

<a id="2-dense-full-sft"></a>
## 2. 密集全量SFT

需要：

```text
out/pretrain_768.pth
```

默认数据集：

```text
../dataset/sft_t2t_mini.jsonl
```

<a id="mini-validation-1"></a>
### 迷你验证

```powershell
python train_full_sft.py --save_weight full_sft_mini
```

<a id="full-training-1"></a>
### 完整训练

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl
```

最终输出：

```text
out/full_sft_mini_768.pth
out/full_sft_768.pth
```

---

<a id="3-moe-pretrain"></a>
## 3. MoE预训练

使用相同的预训练默认设置，启用MoE功能。

```powershell
python train_pretrain.py --use_moe 1 --data_path ../dataset/pretrain_t2t.jsonl
```

最终输出：

```text
out/pretrain_768_moe.pth
```

如果VRAM不足，请相应地减少`batch_size`并增加`accumulation_steps`。

---

<a id="4-moe-full-sft"></a>
## 4. MoE全量SFT

需要：

```text
out/pretrain_768_moe.pth
```

```powershell
python train_full_sft.py --use_moe 1 --data_path ../dataset/sft_t2t.jsonl
```

最终输出：

```text
out/full_sft_768_moe.pth
```

---

<a id="5-lora"></a>
## 5. LoRA

需要先完成密集全量SFT。

默认配置已使用：

```text
base weight: full_sft
dataset: lora_medical.jsonl
Dense model
```

<a id="training"></a>
### 训练

```powershell
python train_lora.py
```

最终输出：

```text
out/lora_medical_768.pth
```

<a id="evaluation"></a>
### 评估

LoRA将与基模型一起进行评估。

`--weight`的值必须与LoRA训练时使用的基模型一致。

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

另一个LoRA适配器的示例：

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

这在保留基模型通用能力的同时，应用了LoRA适配器学习到的领域特定行为。

<a id="merge"></a>
### 合并

LoRA适配器可选择性地合并回其基模型，以生成一个独立的完整模型检查点。

使用：

```text
scripts/convert_model.py
```

与：

```text
convert_merge_base_lora
```

<a id="full-fine-tuning-alternative"></a>
### 完整微调替代方案

在拥有足够领域数据的情况下，可以使用全参数SFT替代LoRA。

领域数据应与通用数据谨慎混合，以降低模型过度拟合特定领域并损害其通用能力的风险。

---

<a id="6-dpo"></a>
## 6. DPO

需要密集全量全量微调。

默认配置：

```text
base weight: full_sft
dataset: dpo.jsonl
beta: 0.15
```

```powershell
python train_dpo.py
```

最终输出：

```text
out/dpo_768.pth
```

---

<a id="7-distillation"></a>
## 7. 知识蒸馏

密集学生模型 ← MoE 教师模型。

需要：

```text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

默认配置：

```text
student: Dense Full SFT
teacher: MoE Full SFT
student hidden size: 768
teacher hidden size: 768
alpha: 0.5
temperature: 1.5
```

```powershell
python train_distillation.py --data_path ../dataset/sft_t2t.jsonl
```

最终输出：

```text
out/full_dist_768.pth
```

---

<a id="8-grpo"></a>
## 8. GRPO

需要密集全量全量微调。

默认配置：

```text
base weight: full_sft
dataset: rlaif.jsonl
loss: cispo
num_generations: 6
beta: 0.1
rollout engine: torch
```

```powershell
python train_grpo.py
```

最终输出：

```text
out/grpo_768.pth
```

---

<a id="9-ppo"></a>
## 9. PPO

需要密集全量全量微调。

默认配置：

```text
base weight: full_sft
dataset: rlaif.jsonl
actor learning rate: 3e-7
critic learning rate: 5e-7
mini batch size: 2
rollout engine: torch
```

```powershell
python train_ppo.py
```

最终输出：

```text
out/ppo_actor_768.pth
```

---

<a id="10-agent-rl"></a>
## 10. 代理强化学习

需要密集全量全量微调。

默认配置：

```text
base weight: full_sft
dataset: agent_rl.jsonl
loss: cispo
num_generations: 4
beta: 0.1
max sequence length: 1024
max generation length: 768
max total length: 2500
rollout engine: torch
```

```powershell
python train_agent.py
```

最终及输出：

```text
out/agent_768.pth
```

---

<a id="11-resume"></a>
## 11. 恢复训练

仅恢复同一阶段和数据集的中断运行。

使用相同的命令并添加：

```text
--from_resume 1
```

示例：

```powershell
python train_pretrain.py --from_resume 1
```

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl --from_resume 1
```

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl --from_resume 1
```

保持不变：

```text
dataset
hidden size
layer count
Dense / MoE
training stage
checkpoint name
```

恢复检查点：

|训练器|检查点|
| --- | --- |
|密集预训练| `checkpoints/pretrain_768_resume.pth` |
|密集全量微调| `checkpoints/full_sft_768_resume.pth` |
|MoE 预训练| `checkpoints/pretrain_768_moe_resume.pth` |
|MoE 全量微调| `checkpoints/full_sft_768_moe_resume.pth` |
|LoRA| `checkpoints/lora_medical_768_resume.pth` |
|知识蒸馏| `checkpoints/full_dist_768_resume.pth` |
|GRPO| `checkpoints/grpo_768_resume.pth` |
|PPO| `checkpoints/ppo_actor_768_resume.pth` |
|代理强化学习| `checkpoints/agent_768_resume.pth` |

通过以下命令查看可用的恢复检查点：

```powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

---
