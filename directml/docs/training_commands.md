# MiniMind DirectML --- Historical Training Commands and Checkpoint Resume

> **Project direction update — 2026-09-21**
>
> The DirectML work is now retained as a compatibility and feasibility study,
> not as the active MiniMind training backend. The upstream official
> `pretrain_768.pth` checkpoint generates coherent text on both CPU and
> DirectML, while the locally trained DirectML checkpoints remained incoherent
> after epoch 1 and epoch 2 despite apparently healthy loss and checkpoint
> diagnostics. This isolates the unresolved problem to the custom DirectML
> training path rather than the tokenizer, dataset, checkpoint loader, or
> DirectML inference path.
>
> Because acceptable pretraining quality could not be obtained reliably with
> DirectML, the DirectML training track is **abandoned for this project**.
> Development is moving to **ROCm**. DirectML benchmarks, issues, workarounds,
> commands, and validation results below are preserved as historical technical
> evidence unless explicitly stated otherwise.

This document centralizes the DirectML training commands used during the
feasibility study and explains how automatic checkpoints and training resume
worked on that branch. These commands are preserved for reproduction only and
are **not the recommended commands for new full training**.

## Current recommendation

Do not start a new full MiniMind training run with the DirectML commands below.
The final locally trained DirectML checkpoints remained incoherent after epoch 1
and epoch 2, while the official checkpoint generated coherently through the same
CPU/DirectML evaluation path.

New training work should use the ROCm environment. ROCm-specific end-to-end
commands should be documented separately after the baseline is validated rather
than inferred by mechanically replacing `--device directml:1` in these historical
commands.

## Trainer configurations

The values below are the **defaults proposed by the upstream MiniMind
trainers**. They were used as target configurations in the historical
sequential DirectML workflow.

| Trainer | Batch size | Max seq len | Accumulation |
| --- | ---: | ---: | ---: |
| Dense Pretrain | 32 | 340 | 8 |
| Dense Full SFT | 16 | 768 | 1 |
| LoRA | 32 | 340 | 1 |
| DPO | 4 | 1024 | 1 |
| Distillation | 32 | 340 | 1 |
| GRPO | 2 | 768 | 1 |
| PPO | 2 | 768 | 1 |
| Agent RL | 2 | 1024 | 1 |

The DirectML-specific numerical settings remain:

``` text
Device: directml:1
DType: float16
DirectML static loss scale: 1024
torch.compile: disabled
```

For DirectML FP16 training, the project additionally uses:

- FP32 master weights for optimizer updates;
- per-token cross-entropy (`reduction="none"`) followed by FP32
  valid-token averaging.

These adaptations avoid the FP16 optimizer instability and incorrect
DirectML cross-entropy reductions identified during long-run validation.

Dense Pretrain additionally uses FP32 master weights with AdamW
`eps = 1e-8`. Other trainer-specific precision paths remain unchanged
unless explicitly documented.

### Historical M4 DirectML baseline

The sustained M4 validation baseline was:

``` text
Trainer: Dense Pretrain
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
loss_scale = 1024
AdamW eps = 1e-4
```

This configuration remains useful as historical M4 validation evidence.

### Current M5 Dense reference

The retained Dense pretraining path uses:

``` text
batch_size = 32
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
optimizer weights = float32 master weights
AdamW eps = 1e-8
loss = per-token cross-entropy → FP32 valid-token mean
```

This corrected path completed both full pretraining epochs successfully.

The larger trainer values in the table remain training targets to validate
individually on DirectML.

### MoE DirectML override

Upstream Pretrain and Full SFT expose the same batch defaults regardless
of Dense/MoE mode. For this DirectML fork, MoE deliberately keeps a
conservative physical batch because the scatter-free compatibility path
evaluates all experts:

| MoE trainer | DirectML batch | Max seq len | Accumulation |
| --- | ---: | ---: | ---: |
| MoE Pretrain | 1 | 340 | 64 |
| MoE Full SFT | 1 | 768 | 64 |

These MoE values are a DirectML safety override, not upstream defaults.

Current upstream Pretrain and Full SFT expose `--seed 42`. The
current DirectML fork used in this project may not expose that CLI option
yet, so the commands below do not force `--seed`; keep the fork's current
seed behavior until the local argparse is synchronized.

Run individual trainer commands from:

``` powershell
cd trainer
```

The complete sequential pipeline is started from the repository root
with:

``` powershell
.\scripts\train_all.ps1
```

## Checkpoint behavior

Each command below uses the current upstream `save_interval` default
for that trainer unless a short DirectML validation run intentionally
overrides it.

The trainer writes a resume checkpoint at the configured interval and
at the normal end of an epoch.

Two kinds of files are produced:

``` text
out/
    # Model weights intended for later stages / inference.

checkpoints/
    # Complete resume state used to continue interrupted training.
```

For example, Dense Pretrain produces:

``` text
out/pretrain_768.pth
checkpoints/pretrain_768.pth
checkpoints/pretrain_768_resume.pth
```

The `_resume.pth` file is the important file for resuming training. It
contains the model state, optimizer state, epoch, step, world size, and
scaler state when applicable. Trainers with schedulers also store their
scheduler state. PPO additionally stores the critic and critic
optimizer/scheduler states.

The current checkpoint implementation keeps the **latest checkpoint**
for each training stage. A new save replaces the previous checkpoint for
the same weight name. Therefore, the recovery distance is bounded approximately by the
configured `save_interval`; the current implementation does not create
a permanent archive of every checkpoint.

## Resume an interrupted training

To resume, rerun the **same training command** and add:

``` text
--from_resume 1
```

Do not change the architecture parameters when resuming. In particular,
keep the same hidden size, layer count, Dense/MoE setting, and training
stage.

The trainer automatically looks for the matching file in `checkpoints/`,
restores the saved state, skips the already completed batches of the
current epoch, and continues from the next step.

Example:

``` text
Checkpoint saved at step 1200
Training interrupted at step 1267

Resume:
    --from_resume 1

The trainer reloads step 1200 and continues from step 1201.
The work from steps 1201–1267 must be recomputed.
```

------------------------------------------------------------------------

# 1. Dense Pretrain

Full training:

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

Resume:

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

Resume checkpoint:

``` text
checkpoints/pretrain_768_resume.pth
```

------------------------------------------------------------------------

# 2. Dense Full SFT

Requires `out/pretrain_768.pth`.

Full training:

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

Resume: use the same command and add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/full_sft_768_resume.pth
```

------------------------------------------------------------------------

# 3. MoE Pretrain

The MoE branch uses the conservative micro-batch configuration used by
the sequential training script.

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

Resume: add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/pretrain_768_moe_resume.pth
```

------------------------------------------------------------------------

# 4. MoE Full SFT

Requires `out/pretrain_768_moe.pth`.

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

Resume: add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/full_sft_768_moe_resume.pth
```

------------------------------------------------------------------------

# 5. LoRA

Uses the Dense Full SFT model by default.

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

Resume: add:

``` text
--from_resume 1
```

With the default `--lora_name lora_medical`, the resume checkpoint is:

``` text
checkpoints/lora_medical_768_resume.pth
```

If `--lora_name` is changed, the checkpoint name changes accordingly.

------------------------------------------------------------------------

# 6. DPO

The sequential pipeline keeps DPO on the Dense Full SFT branch.

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

Resume: add:

``` text
--from_resume 1
```

**Important:** DPO is outside the final 9/9 DirectML trainer smoke
runner. Its current trainer should be rechecked before relying on it for
a long DirectML FP16 run.

------------------------------------------------------------------------

# 7. Distillation --- Dense student ← MoE teacher

Requires:

``` text
out/full_sft_768.pth
out/full_sft_768_moe.pth
```

Command:

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

Resume: add:

``` text
--from_resume 1
```

With the default `--save_weight full_dist`, the resume checkpoint is:

``` text
checkpoints/full_dist_768_resume.pth
```

------------------------------------------------------------------------

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

Resume: add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/grpo_768_resume.pth
```

The resume state also contains the scheduler state.

------------------------------------------------------------------------

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

Resume: add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/ppo_actor_768_resume.pth
```

PPO restores both the actor and critic training state.

------------------------------------------------------------------------

# 10. Agent RL

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

Resume: add:

``` text
--from_resume 1
```

Resume checkpoint:

``` text
checkpoints/agent_768_resume.pth
```

The scheduler state is restored as part of the resume checkpoint.

------------------------------------------------------------------------

# Full sequential training

From the repository root:

``` powershell
.\scripts\train_all.ps1
```

The sequential script should preserve each trainer's intended checkpoint interval unless a DirectML validation run deliberately overrides it.

If a stage is interrupted, do **not** restart the whole pipeline
immediately. Resume the interrupted trainer directly with its command
above and `--from_resume 1`. Once that stage has completed, the
remaining stages can be launched individually or the sequential script
can be adapted to start from the next stage.

## Quick resume rule

``` text
Normal start:
    --from_resume 0   (default)

After interruption:
    same command
    + --from_resume 1
```

Before resuming, verify that the expected `_resume.pth` file exists in
`checkpoints/`.

Example:

``` powershell
Get-ChildItem ..\checkpoints\*_resume.pth
```

when run from `trainer/`, or:

``` powershell
Get-ChildItem .\checkpoints\*_resume.pth
```

when run from the repository root.

## Important notes

-   `out/*.pth` contains model weights; it is not sufficient to recover
    optimizer progress.
-   `checkpoints/*_resume.pth` is the file used by `--from_resume 1`.
-   Keep `out/` and `checkpoints/` when you want to resume.
-   Do not clean the checkpoint directory after an interrupted training.
-   Resume with the same architecture and stage configuration.
-   The checkpoint is replaced at each save; the current implementation
    does not retain a numbered history of every 100-step save.
