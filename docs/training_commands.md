# MiniMind ROCm — Training Commands

Environment setup and ROCm/PyTorch device semantics are documented in the root
[`README.md`](../README.md).

Training order, mini-vs-full rules and validation milestones are documented in
[`roadmap.md`](roadmap.md).

Run trainer commands from:

```powershell
cd trainer
```

### GPU selection with integrated graphics

On systems where the CPU also provides an integrated AMD GPU, ROCm/PyTorch may expose
the integrated GPU as `cuda:0` and the dedicated Radeon GPU as another device.

Check the detected devices before starting training:

```powershell
python -c "import torch; print('device_count:', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Example:

```text
0 AMD Radeon(TM) Graphics
1 AMD Radeon RX 7800 XT
```

If the dedicated GPU is device `1`, select it before launching MiniMind:

```powershell
$env:HIP_VISIBLE_DEVICES="1"
```

Then verify the visible device:

```powershell
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Expected result:

```text
1
AMD Radeon RX 7800 XT
```

After `HIP_VISIBLE_DEVICES` filters the devices, the selected dedicated GPU becomes
`cuda:0` inside the Python process. Trainer commands should therefore continue to use
`--device cuda:0` when this environment variable is active.

`HIP_VISIBLE_DEVICES` applies to the current PowerShell session. Set it again when
opening a new terminal if the integrated GPU would otherwise be selected.

---

## Upstream datasets

Official MiniMind training datasets are provided by the upstream project:

- [ModelScope — minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- [Hugging Face — minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

Only the required files need to be downloaded and placed in:

```text
dataset/
```

Main datasets:

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

For quick validation, use the `*_mini.jsonl` datasets.

For full training, use `pretrain_t2t.jsonl` and `sft_t2t.jsonl`.

---

## 1. Dense Pretrain

Default dataset:

```text
../dataset/pretrain_t2t_mini.jsonl
```

### Mini validation

```powershell
python train_pretrain.py --save_weight pretrain_mini
```

### Full training

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl
```

Final output:

```text
out/pretrain_mini_768.pth
out/pretrain_768.pth
```

---

## 2. Dense Full SFT

Requires:

```text
out/pretrain_768.pth
```

Default dataset:

```text
../dataset/sft_t2t_mini.jsonl
```

### Mini validation

```powershell
python train_full_sft.py --save_weight full_sft_mini
```

### Full training

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl
```

Final output:

```text
out/full_sft_mini_768.pth
out/full_sft_768.pth
```

---

## 3. MoE Pretrain

Uses the same Pretrain defaults, with MoE enabled.

```powershell
python train_pretrain.py --use_moe 1 --data_path ../dataset/pretrain_t2t.jsonl
```

Final output:

```text
out/pretrain_768_moe.pth
```

If VRAM is insufficient, reduce `batch_size` and increase
`accumulation_steps` accordingly.

---

## 4. MoE Full SFT

Requires:

```text
out/pretrain_768_moe.pth
```

```powershell
python train_full_sft.py --use_moe 1 --data_path ../dataset/sft_t2t.jsonl
```

Final output:

```text
out/full_sft_768_moe.pth
```

---

## 5. LoRA

Requires Dense Full SFT.

Default configuration already uses:

```text
base weight: full_sft
dataset: lora_medical.jsonl
Dense model
```

### Training

```powershell
python train_lora.py
```

Final output:

```text
out/lora_medical_768.pth
```

### Evaluation

LoRA is evaluated together with its base model.

The `--weight` value must match the base model used during LoRA training.

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

Example with another LoRA adapter:

```powershell
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

This keeps the general capabilities of the base model while applying the
domain-specific behavior learned by the LoRA adapter.

### Merge

A LoRA adapter can optionally be merged back into its base model to produce
a standalone full-model checkpoint.

Use:

```text
scripts/convert_model.py
```

with:

```text
convert_merge_base_lora
```

### Full fine-tuning alternative

With sufficient domain data, full-parameter SFT can be used instead of LoRA.

Domain data should be mixed carefully with general-purpose data to reduce the
risk of overfitting the specialized domain and degrading the model's general
capabilities.

---

## 6. DPO

Requires Dense Full SFT.

Default configuration:

```text
base weight: full_sft
dataset: dpo.jsonl
beta: 0.15
```

```powershell
python train_dpo.py
```

Final output:

```text
out/dpo_768.pth
```

---

## 7. Distillation

Dense student ← MoE teacher.

Requires:

```text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

Default configuration:

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

Final output:

```text
out/full_dist_768.pth
```

---

## 8. GRPO

Requires Dense Full SFT.

Default configuration:

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

Final output:

```text
out/grpo_768.pth
```

---

## 9. PPO

Requires Dense Full SFT.

Default configuration:

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

Final output:

```text
out/ppo_actor_768.pth
```

---

## 10. Agent RL

Requires Dense Full SFT.

Default configuration:

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

Final output:

```text
out/agent_768.pth
```

---

## 11. Resume

Resume only an interrupted run of the **same stage and dataset**.

Use the same command and add:

```text
--from_resume 1
```

Examples:

```powershell
python train_pretrain.py --from_resume 1
```

```powershell
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl --from_resume 1
```

```powershell
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl --from_resume 1
```

Keep unchanged:

```text
dataset
hidden size
layer count
Dense / MoE
training stage
checkpoint name
```

Resume checkpoints:

| Trainer | Checkpoint |
| --- | --- |
| Dense Pretrain | `checkpoints/pretrain_768_resume.pth` |
| Dense Full SFT | `checkpoints/full_sft_768_resume.pth` |
| MoE Pretrain | `checkpoints/pretrain_768_moe_resume.pth` |
| MoE Full SFT | `checkpoints/full_sft_768_moe_resume.pth` |
| LoRA | `checkpoints/lora_medical_768_resume.pth` |
| Distillation | `checkpoints/full_dist_768_resume.pth` |
| GRPO | `checkpoints/grpo_768_resume.pth` |
| PPO | `checkpoints/ppo_actor_768_resume.pth` |
| Agent RL | `checkpoints/agent_768_resume.pth` |

Check available resume checkpoints with:

```powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

---