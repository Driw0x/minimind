# MiniMind DirectML --- Training Commands and Checkpoint Resume

This document centralizes the full DirectML training commands used by
the project and explains how automatic checkpoints and training resume
work.

## Trainer configurations

The values below are the **defaults proposed by the upstream MiniMind
trainers**. They are the values used as the target configuration in the
updated sequential training workflow.

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
Hidden size: 768
Layers: 8
DirectML static loss scale: 1024
Checkpoint interval: 100 iterations
torch.compile: disabled
```

Dense Pretrain additionally uses FP32 master weights with AdamW
`eps = 1e-8`. Other trainer-specific precision paths remain unchanged
unless explicitly documented.

### DirectML validated baseline

The table above must **not** be confused with the sustained DirectML
pretraining baseline validated during M4.

The sustained validation result remains:

``` text
Trainer: Dense Pretrain
batch_size = 8
max_seq_len = 340
accumulation_steps = 8
compute dtype = float16
loss_scale = 1024
optimizer weights = float32 master weights
AdamW eps = 1e-8
```

This `8 × 340` configuration was validated through global step `1100`
with teacher-forced checkpoint diagnostics. It is a **validated DirectML
baseline**, not the default configuration of every trainer.

The larger upstream trainer values in the table are therefore training
targets to validate on DirectML. A short compatibility pass does not
guarantee that they will remain stable or fit in VRAM during a full run.

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

Every training command below includes:

``` text
--save_interval 100
```

The trainer therefore writes a resume checkpoint every 100 training
iterations and at the normal end of an epoch.

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
the same weight name. Therefore, `--save_interval 100` gives a recovery
point at most approximately 100 iterations behind an unexpected
interruption; it does not create a permanent archive of every 100-step
checkpoint.

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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 8 `
  --max_seq_len 340 `
  --accumulation_steps 8 `
  --directml_loss_scale 1024 `
  --save_interval 100 `
  --from_weight none `
  --use_compile 0
```

Resume:

``` powershell
python train_pretrain.py `
  --device directml:1 `
  --dtype float16 `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 8 `
  --max_seq_len 340 `
  --accumulation_steps 8 `
  --directml_loss_scale 1024 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 16 `
  --max_seq_len 768 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 1 `
  --batch_size 1 `
  --max_seq_len 340 `
  --accumulation_steps 64 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 1 `
  --batch_size 1 `
  --max_seq_len 768 `
  --accumulation_steps 64 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 32 `
  --max_seq_len 340 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 4 `
  --max_seq_len 1024 `
  --accumulation_steps 1 `
  --save_interval 100 `
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
  --batch_size 32 `
  --max_seq_len 340 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 2 `
  --max_seq_len 768 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 2 `
  --max_seq_len 768 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --use_moe 0 `
  --batch_size 2 `
  --max_seq_len 1024 `
  --accumulation_steps 1 `
  --directml_loss_scale 1024 `
  --directml_adam_eps 1e-4 `
  --save_interval 100 `
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

The updated script passes `--save_interval 100` to every stage.

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
