[English](roadmap.md) | [Français](roadmap.fr.md) | [简体中文](roadmap.zh-CN.md)

[项目 README](../README.zh-CN.md)

<a id="minimind-rocm-roadmap"></a>
# MiniMind ROCm 路径规划

本路径规划跟踪在 Windows 上使用 AMD ROCm 验证上游 MiniMind 训练流水线的过程。

有关可复现的训练命令和检查点恢复语法，请参见 [`training_commands.md`](training_commands.zh-CN.md)。

---

<a id="training-rule"></a>
## 训练规则

短时运行仅用于验证 ROCm 兼容性。最终检查点必须来自计划中的完整训练运行。

不要从短验证检查点继续完整运行。恢复仅用于继续同一中断运行。

---

<a id="m1--dense-pretrain--evaluation-"></a>
## M1 — 密集预训练 + 评估 ✅

在 ROCm 上验证密集预训练，并生成用于完整 SFT 的基础检查点。

<a id="training"></a>
### 训练

- [x] 使用短运行验证 ROCm 的执行
- [x] 完成密集预训练
- [x] 确认损失为有限且持续下降
- [x] 保存最终预训练检查点

<a id="evaluation"></a>
### 评估

- [x] 使用 `python eval_llm.py --weight pretrain` 加载检查点
- [x] 运行上游自动测试模式 (`[0] 自动测试`)
- [x] 确认内置提示下的生成一致性

**输出：** 保留的密集预训练检查点，基于完整预训练数据集训练，用于完整 SFT。

---

<a id="m2--dense-full-sft--evaluation-"></a>
## M2 — 密集完整 SFT + 评估 ✅

从保留的 M1 检查点出发，在 ROCm 上验证完整 SFT。

<a id="training-1"></a>
### 训练

- [x] 使用短运行验证 ROCm 的完整 SFT 执行
- [x] 确认损失为有限且持续下降
- [x] 保存并重新加载验证检查点
- [x] 完成完整 SFT
- [x] 保存最终完整 SFT 检查点

<a id="evaluation-1"></a>
### 评估

- [x] 使用 `eval_llm.py` 加载最终检查点
- [x] 运行上游自动测试模式 (`[0] 自动测试`)
- [x] 确认内置提示下的生成一致性与稳定性
- [x] 评估事实准确性与推理能力
- [x] 评估指令遵循的一致性
- [x] 与 M1 预训练检查点在相同提示下进行对比

**已知限制：** 事实准确性、算术推理、代码修正以及严格指令遵循仍不可靠。

**输出：** 保留的密集完整 SFT 检查点，基于完整 SFT 数据集训练，用于下游阶段。

---

<a id="m3--remaining-upstream-pipeline-"></a>
## M3 — 剩余上游流程 ⏳

在不重新设计上游流程的前提下，在 ROCm 上验证剩余的上游 MiniMind 训练阶段。

<a id="pipeline"></a>
### 流程

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

<a id="dense-stages"></a>
### 密集阶段

- [ ] 验证 LoRA
- [ ] 验证 DPO
- [ ] 验证 GRPO
- [ ] 验证 PPO
- [ ] 验证 Agent RL

<a id="moe"></a>
### MoE

- [x] 验证并完成 MoE 预训练
- [x] 验证并完成 MoE 完整 SFT
- [x] 评估最终 MoE 检查点

保留的检查点为 `pretrain_768_moe.pth` 和 `full_sft_768_moe.pth`，依次使用 `pretrain_t2t.jsonl` 和 `sft_t2t.jsonl` 训练，Full SFT 从 MoE Pretrain 初始化。ROCm 初步评估已完成：Mini 参考损失和八个外部提示的确定性生成。结果与方法限制见 [`generation_quality.md`](generation_quality.zh-CN.md)。

<a id="distillation"></a>
### 蒸馏

- [ ] 使用密集学生模型和 MoE 教师模型验证蒸馏
- [ ] 完成蒸馏
- [ ] 评估蒸馏后的检查点

<a id="final-validation"></a>
### 最终验证

- [ ] 在各阶段之间确认检查点兼容性
- [ ] 记录最终的ROCm环境和训练命令
- [ ] 仅保留来自上游的必要ROCm特定变更

---

<a id="validation-rule"></a>
## 验证规则

当适用的检查通过时，一个阶段被视为已验证：

```text
ROCm execution
    +
finite and stable training
    +
checkpoint save/load
    +
upstream evaluation
```

ROCm特定变更仅应在已验证的兼容性问题中引入。
