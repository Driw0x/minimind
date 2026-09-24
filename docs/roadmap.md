# MiniMind ROCm Roadmap

This roadmap tracks the active MiniMind training validation on Windows with AMD ROCm.

The objective is to validate model quality progressively before running the complete upstream training pipeline.

For exact trainer commands and checkpoint resume syntax, see
[`training_commands.md`](training_commands.md).

---

## Training scope: smoke validation vs full training

Mini/smoke runs are used to validate ROCm compatibility and trainer behavior.
Their checkpoints are test artifacts, not the base models for final training.

Example validation dataset:

```text
dataset/pretrain_t2t_mini.jsonl
```

After a stage is validated, the intended full run uses the complete dataset:

```text
FULL pretrain dataset
        ↓
Pretrain from scratch
        ↓
retained full pretrain checkpoint
        ↓
FULL SFT dataset
        ↓
Full SFT
```

The same rule applies to the MoE branch.

Do **not** continue a full run from a mini-validation checkpoint. A resume is
only for continuing the same interrupted run; changing from a mini dataset to a
full dataset is a new training run.

---

## M1 — Dense Pretrain + Evaluation 🚧

Validate the upstream Dense pretraining stage on ROCm before continuing to later training stages.

### Training

- [x] Validate the ROCm environment and GPU execution
- [x] Run Dense pretraining on ROCm
- [x] Save a ROCm pretraining checkpoint
- [x] Confirm that training loss decreases over the run

### Evaluation

- [x] Load the ROCm checkpoint with `eval_llm.py`
- [x] Confirm coherent qualitative generation
- [x] Compare the ROCm checkpoint with the previous reference checkpoint on a fixed evaluation subset
- [ ] Add a reproducible pretraining evaluation command/script to the active ROCm workflow
- [ ] Confirm the final retained pretraining configuration
- [ ] Complete the intended Dense pretraining run used as the base checkpoint for SFT

### Success criterion

The retained pretraining checkpoint must:

```text
train successfully
    +
produce finite and decreasing loss
    +
reload correctly
    +
produce coherent generation
    +
pass the fixed checkpoint evaluation
```

Only then continue to Full SFT.

---

## M2 — Full SFT + Evaluation ⏳

Validate the upstream Full SFT stage from the retained ROCm pretraining checkpoint.

### Training

- [ ] Start Full SFT from the validated pretraining checkpoint
- [ ] Validate a short/bounded SFT run first
- [ ] Check loss and gradient stability
- [ ] Save and reload the SFT checkpoint
- [ ] Complete the intended Full SFT run

### Evaluation

- [ ] Evaluate the SFT checkpoint with `eval_llm.py`
- [ ] Use a fixed prompt set for qualitative comparison
- [ ] Verify that instruction-following quality improves over the pretrained checkpoint
- [ ] Add a reproducible SFT evaluation result to the project documentation

### Success criterion

The retained Full SFT checkpoint must:

```text
train successfully from the ROCm pretrain checkpoint
    +
reload correctly
    +
remain numerically stable
    +
produce coherent instruction-following responses
```

Only then continue to the complete upstream workflow.

---

## M3 — Complete Upstream Training Pipeline ⏳

Run and validate the full MiniMind training workflow on ROCm using the upstream training stages.

The goal of this milestone is not to redesign the pipeline. ROCm-specific changes should remain minimal and the upstream training logic should be preserved wherever possible.

### Pipeline

Validate the available upstream training stages, including:

```text
Dense Pretrain
    ↓
Dense Full SFT
    ├── LoRA
    ├── DPO
    ├── GRPO
    ├── PPO
    └── Agent RL

MoE Pretrain
    ↓
MoE Full SFT

Dense Full SFT + MoE Full SFT
    ↓
Distillation
```

Validate the Dense and MoE base-model paths separately. Distillation uses the
retained Dense Full SFT model as student and the retained MoE Full SFT model as
teacher.

### Validation

For each stage:

- [ ] run a short ROCm smoke test;
- [ ] verify finite loss and gradients;
- [ ] verify checkpoint save/load;
- [ ] verify resume behavior where supported;
- [ ] evaluate the resulting checkpoint when the stage produces an inference model;
- [ ] document ROCm-specific incompatibilities only when they are actually observed.

### Final validation

- [ ] Run the complete upstream training workflow end to end
- [ ] Confirm that stage-to-stage checkpoints remain compatible
- [ ] Validate the final model qualitatively
- [ ] Record the final ROCm environment and reproducible commands
- [ ] Compare the active ROCm fork with upstream and keep only necessary backend-specific changes

---

## Validation Rule

A training stage is not considered validated only because it completes.

Each retained stage should satisfy:

```text
execution
    +
numerical stability
    +
checkpoint correctness
    +
model-quality evaluation
```

The roadmap should remain focused on ROCm. Historical backend experiments are documented separately.
