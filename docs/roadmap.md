# MiniMind ROCm Roadmap

This roadmap tracks the validation of the upstream MiniMind training pipeline on Windows with AMD ROCm.

For reproducible training commands and checkpoint resume syntax, see
[`training_commands.md`](training_commands.md).

---

## Training rule

Short runs validate ROCm compatibility only. Final checkpoints must come from the intended full training runs.

Do not continue a full run from a short-validation checkpoint. Resume is only used to continue the same interrupted run.

---

## M1 — Dense Pretrain + Evaluation ✅

Validate Dense pretraining on ROCm and produce the base checkpoint for Full SFT.

### Training

- [x] Validate ROCm execution with a short run
- [x] Complete Dense pretraining
- [x] Confirm finite and decreasing loss
- [x] Save the final pretraining checkpoint

### Evaluation

- [x] Load the checkpoint with `python eval_llm.py --weight pretrain`
- [x] Run the upstream automatic test mode (`[0] 自动测试`)
- [x] Confirm coherent generation across the built-in prompts

**Output:** retained Dense pretraining checkpoint for Full SFT.

---

## M2 — Dense Full SFT + Evaluation ⏳

Validate Full SFT on ROCm from the retained M1 checkpoint.

### Training

- [x] Validate ROCm execution with a short Full SFT run
- [x] Confirm finite and decreasing loss
- [x] Save and reload the validation checkpoint
- [x] Complete Full SFT
- [x] Save the final Full SFT checkpoint

### Evaluation

- [x] Load the final checkpoint with `eval_llm.py`
- [x] Run the upstream automatic test mode (`[0] 自动测试`)
- [x] Confirm coherent and stable responses across the built-in prompts
- [ ] Confirm reliable factual accuracy and reasoning
- [ ] Confirm consistent instruction following
- [x] Compare with the M1 pretraining checkpoint on the same prompts

**Output:** retained Dense Full SFT checkpoint for downstream stages.

---

## M3 — Remaining Upstream Pipeline ⏳

Validate the remaining upstream MiniMind training stages on ROCm without redesigning the upstream pipeline.

### Pipeline

```text
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

### Dense stages

- [ ] Validate LoRA
- [ ] Validate DPO
- [ ] Validate GRPO
- [ ] Validate PPO
- [ ] Validate Agent RL

### MoE

- [ ] Validate and complete MoE pretraining
- [ ] Validate and complete MoE Full SFT
- [ ] Evaluate the final MoE checkpoint

### Distillation

- [ ] Validate Distillation with Dense student and MoE teacher
- [ ] Complete Distillation
- [ ] Evaluate the distilled checkpoint

### Final validation

- [ ] Confirm checkpoint compatibility across stages
- [ ] Record the final ROCm environment and training commands
- [ ] Keep only necessary ROCm-specific changes from upstream

---

## Validation rule

A stage is validated when applicable checks pass:

```text
ROCm execution
    +
finite and stable training
    +
checkpoint save/load
    +
upstream evaluation
```

ROCm-specific changes should only be introduced for verified compatibility issues.