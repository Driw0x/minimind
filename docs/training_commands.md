# MiniMind ROCm — Training Commands

Environment setup and ROCm/PyTorch device semantics are documented in the root
[`README.md`](../README.md).

Training order, mini-vs-full rules and validation milestones are documented in
[`roadmap.md`](roadmap.md).

Run trainer commands from:

```powershell
cd trainer
```

The commands below assume the validated ROCm environment and use `cuda:0`,
`bfloat16` and `--use_compile 0`.

---

## 1. Dense Pretrain

```powershell
python train_pretrain.py --device cuda:0 --dtype bfloat16 --epochs 2 --batch_size 32 --learning_rate 5e-4 --num_workers 8 --accumulation_steps 8 --grad_clip 1.0 --log_interval 100 --save_interval 1000 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 --use_moe 0 --from_weight none --use_compile 0
```

Final output:

```text
out/pretrain_768.pth
```

---

## 2. Dense Full SFT

Requires:

```text
out/pretrain_768.pth
```

```powershell
python train_full_sft.py --device cuda:0 --dtype bfloat16 --epochs 2 --batch_size 16 --learning_rate 1e-5 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 100 --save_interval 1000 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --use_moe 0 --from_weight pretrain --use_compile 0
```

Final output:

```text
out/full_sft_768.pth
```

---

## 3. MoE Pretrain

```powershell
python train_pretrain.py --device cuda:0 --dtype bfloat16 --epochs 2 --batch_size 32 --learning_rate 5e-4 --num_workers 8 --accumulation_steps 8 --grad_clip 1.0 --log_interval 100 --save_interval 1000 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 --use_moe 1 --from_weight none --use_compile 0
```

Final output:

```text
out/pretrain_768_moe.pth
```

If VRAM is insufficient, reduce `batch_size` and increase `accumulation_steps`.

---

## 4. MoE Full SFT

Requires:

```text
out/pretrain_768_moe.pth
```

```powershell
python train_full_sft.py --device cuda:0 --dtype bfloat16 --epochs 2 --batch_size 16 --learning_rate 1e-5 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 100 --save_interval 1000 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --use_moe 1 --from_weight pretrain --use_compile 0
```

Final output:

```text
out/full_sft_768_moe.pth
```

---

## 5. LoRA

Requires Dense Full SFT.

```powershell
python train_lora.py --device cuda:0 --dtype bfloat16 --epochs 10 --batch_size 32 --learning_rate 1e-4 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 10 --save_interval 1000 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 --use_moe 0 --from_weight full_sft --use_compile 0
```

---

## 6. DPO

Requires Dense Full SFT.

```powershell
python train_dpo.py --device cuda:0 --dtype bfloat16 --epochs 1 --batch_size 4 --learning_rate 4e-8 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 100 --save_interval 100 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 1024 --use_moe 0 --beta 0.15 --from_weight full_sft --use_compile 0
```

---

## 7. Distillation

Dense student ← MoE teacher.

Requires:

```text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

```powershell
python train_distillation.py --device cuda:0 --dtype bfloat16 --epochs 6 --batch_size 32 --learning_rate 5e-6 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 100 --save_interval 100 --max_seq_len 340 --student_hidden_size 768 --student_num_layers 8 --student_use_moe 0 --from_student_weight full_sft --teacher_hidden_size 768 --teacher_num_layers 8 --teacher_use_moe 1 --from_teacher_weight full_sft --use_compile 0
```

---

## 8. GRPO

Requires Dense Full SFT.

```powershell
python train_grpo.py --device cuda:0 --dtype bfloat16 --epochs 1 --batch_size 2 --learning_rate 3e-7 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 1 --save_interval 10 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 1024 --use_moe 0 --num_generations 6 --beta 0.1 --loss_type cispo --from_weight full_sft --use_compile 0
```

---

## 9. PPO

Requires Dense Full SFT.

```powershell
python train_ppo.py --device cuda:0 --dtype bfloat16 --epochs 1 --batch_size 2 --learning_rate 3e-7 --critic_learning_rate 5e-7 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 1 --save_interval 10 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 --max_gen_len 1024 --use_moe 0 --mini_batch_size 2 --from_weight full_sft --use_compile 0
```

---

## 10. Agent RL

Requires Dense Full SFT.

```powershell
python train_agent.py --device cuda:0 --dtype bfloat16 --epochs 1 --batch_size 2 --learning_rate 3e-7 --num_workers 8 --accumulation_steps 1 --grad_clip 1.0 --log_interval 1 --save_interval 10 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 1024 --max_gen_len 768 --max_total_len 2500 --use_moe 0 --num_generations 4 --beta 0.1 --loss_type cispo --from_weight full_sft --use_compile 0
```

---

## 11. Resume

Resume only an interrupted run of the **same stage and dataset**.

Use the exact same command and add:

```text
--from_resume 1
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

| Trainer        | Checkpoint                                |
| -------------- | ----------------------------------------- |
| Dense Pretrain | `checkpoints/pretrain_768_resume.pth`     |
| Dense Full SFT | `checkpoints/full_sft_768_resume.pth`     |
| MoE Pretrain   | `checkpoints/pretrain_768_moe_resume.pth` |
| MoE Full SFT   | `checkpoints/full_sft_768_moe_resume.pth` |
| LoRA           | `checkpoints/lora_medical_768_resume.pth` |
| Distillation   | `checkpoints/full_dist_768_resume.pth`    |
| GRPO           | `checkpoints/grpo_768_resume.pth`         |
| PPO            | `checkpoints/ppo_actor_768_resume.pth`    |
| Agent RL       | `checkpoints/agent_768_resume.pth`        |

Check:

```powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

---
