[English](directml_issues.md) | [Français](directml_issues.fr.md) | [简体中文](directml_issues.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="minimind-----directml-technical-issues"></a>
# MiniMind --- DirectML 技术问题

> **历史 DirectML 归档 — 更新日期：2026-09-24**
>
> DirectML 已不再作为 MiniMind 的训练后端。M5 最终验证发现了一个特定于测试的 DirectML FP16 路径的数值发散问题。以下历史结果为可复现性和工程参考保留；ROCm 开发情况另作记录。

本文件记录了在将 MiniMind 适配至 DirectML 过程中遇到的技术兼容性问题。

基准测试和性能相关观察在 [`directml_benchmarks.md`](directml_benchmarks.zh-CN.md) 中单独记录。当前限制总结如下；详细原因和决策保留在问题历史中。


<a id="current-directml-limitations"></a>
## 当前 DirectML 限制

|功能 / 操作|状态|保留处理方式|
| --- | --- | --- |
|AdamW `aten::lerp.Scalar_out`|部分支持|自动切换至 CPU|
|奖励模型因果掩码|不支持|奖励模型在 CPU 上运行|
| `torch.compile` |不支持|保持禁用|
|CUDA AMP/autocast 等效功能|不可用|DirectML 特定 FP16 路径|
|稀疏 MoE 路由|部分支持|无散射兼容路径|
|代理全序列注意力掩码|部分支持|避免使用有问题的掩码路径|
|交叉熵 `ignore_index` 减少|部分支持|每 token 损失 + FP32 有效 token 均值|
|DirectML FP16 数值一致性|**阻塞**|不要使用 DirectML 进行完整训练|

最终的阻塞限制是 M5 验证期间发现的 FP16 数值发散问题。DirectML 仅能作为历史兼容性和回归测试目标保留使用。

------------------------------------------------------------------------

<a id="directml-device-displayed-as-privateuseone0"></a>
# DirectML 设备显示为 `privateuseone:0`

<a id="problem"></a>
## 问题

初始化 DirectML 后，PyTorch 报告：

``` text
privateuseone:0
```

而不是像 `directml:0` 这样的设备名称。

<a id="cause"></a>
## 原因

`torch-directml` 通过 `PrivateUse1` 后端将 DirectML 集成到 PyTorch 中。

因此，`privateuseone:0` 是 DirectML 设备在 PyTorch 内部的预期表示。

<a id="solution"></a>
## 解决方案

无需修改代码。

通过检查模型参数、输入和输出来验证设备放置。

<a id="decision"></a>
## 决策

`privateuseone:0` 被视为预期的 DirectML 行为。

面向用户的设备选项仍为：

``` text
directml
```

------------------------------------------------------------------------

<a id="directml-not-available-as-an-original-device-option"></a>
# DirectML 未作为原始设备选项提供

<a id="problem-1"></a>
## 问题

上游的 MiniMind 并未将 DirectML 作为执行设备提供。

<a id="cause-1"></a>
## 原因

MiniMind 原本并未围绕 `torch-directml` 设计，该功能需要通过显式初始化实现：

``` python
import torch_directml

device = torch_directml.device()
```

<a id="solution-1"></a>
## 解决方案

将 `directml` 作为显式设备选项添加。

最终设备仍使用标准 PyTorch 操作：

``` python
model.to(device)
tensor.to(device)
```

<a id="decision-1"></a>
## 决策

DirectML 应该复用现有的 MiniMind 执行流水线，而不是引入独立的训练实现。

------------------------------------------------------------------------

<a id="adamw-cpu-fallback"></a>
# AdamW CPU 降级

<a id="problem-2"></a>
## 问题

在 AdamW 优化器步骤中，DirectML 报告：

``` text
The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU.
```

<a id="cause-2"></a>
## 原因

DirectML 并未实现 `torch.optim.AdamW` 内部使用的所有 PyTorch 操作。

`torch-directml` 会自动在 CPU 上执行不支持的操作。

<a id="solution-2"></a>
## 解决方案

尽管有警告，完整的训练步骤仍被验证通过：

``` text
DirectML backward pass: OK
DirectML optimizer step: OK
DirectML zero_grad: OK
DirectML training step: OK
```

<a id="decision-2"></a>
## 决策

由于降级不会阻止正确训练，因此该降级被接受。

它仍被记录下来，因为它可能影响性能。

------------------------------------------------------------------------

<a id="torchcompile-unsupported-on-directml"></a>
# `torch.compile` 在 DirectML 上不支持

<a id="problem-3"></a>
## 问题

已归档的 DirectML 训练路径不支持 `torch.compile`。

<a id="solution-3"></a>
## 解决方案

运行 DirectML 工作流时使用：

```text
--use_compile 0
```

<a id="decision-3"></a>
## 决策

DirectML 归档中的编译功能仍被禁用。由于 DirectML 已不再作为活跃的训练后端，因此不再寻求进一步的补救措施。

------------------------------------------------------------------------

<a id="checkpoint-and-model-architecture-mismatch"></a>
# 检查点与模型架构不匹配

<a id="problem-4"></a>
## 问题

在加载具有严格状态字典匹配的验证检查点时，`eval_llm.py` 失败。

<a id="cause-3"></a>
## 原因

评估模型架构与训练时使用的架构不匹配。

检查点兼容性取决于如下参数：

``` text
hidden_size
num_hidden_layers
vocab_size
```

所使用的验证检查点为：

``` text
hidden_size = 128
num_hidden_layers = 2
```

<a id="solution-4"></a>
## 解决方案

评估必须实例化一个兼容的模型。

对于验证配置：

``` powershell
--hidden_size 128 `
--num_hidden_layers 2
```

<a id="decision-4"></a>
## 决策

检查点加载失败不应自动归因于 DirectML。

必须首先验证架构兼容性。

------------------------------------------------------------------------

<a id="reward-model-fails-on-directml"></a>
# 奖励模型在 DirectML 上失败

<a id="problem-5"></a>
## 问题

在 GRPO 训练过程中，主 MiniMind 模型在 DirectML 上运行正常，而当奖励模型被强制部署到同一设备时则失败。

<a id="cause-4"></a>
## 原因

奖励模型推理路径使用了 DirectML 无法可靠支持的操作。

原始工作流也假设参与的模型可以共享同一个执行设备。

<a id="solution-5"></a>
## 解决方案

可训练模型仍保留在 DirectML 上，而奖励模型在 CPU 上执行：

``` text
Trainable model → DirectML
Reward model    → CPU
```

奖励模型输入在推理前被移动到 CPU。

当需要时，奖励值会被传回训练设备。

<a id="decision-5"></a>
## 决策

在 DirectML 下，奖励模型有意保留在 CPU 上。

正确性与稳定性优先于将所有组件强制部署到 GPU 上。

------------------------------------------------------------------------

<a id="duplicated-device-handling-across-trainers"></a>
# 多个训练器中的设备处理重复问题

<a id="problem-6"></a>
## 问题

随着 DirectML 支持的扩展，各个训练器中开始出现设备特定的处理逻辑。

这可能导致重复且不一致的后端逻辑。

<a id="cause-5"></a>
## 原因

DirectML 引入了组件特定的设备要求，这与最初假设的单一共享执行设备不符。

<a id="solution-6"></a>
## 解决方案

设备和后端兼容性处理被集中到共享训练工具中。

这些工具提供了一个统一的位置用于：

-   设备分辨率；
-   DirectML 初始化；
-   模型放置；
-   组件特定的设备要求；
-   未来后端兼容性规则。

<a id="decision-6"></a>
## 决策

各个训练器应专注于其训练算法。

后端兼容性应位于共享训练基础设施中。

------------------------------------------------------------------------

<a id="empty-token-handling-during-generation"></a>
# 生成过程中的空标记处理

<a id="problem-7"></a>
## 问题

基于生成的训练可能会产生一个空的标记序列。

将空结果传递到下游可能会生成一个无效的训练样本。

<a id="cause-6"></a>
## 原因

生成路径假设生成的输出始终包含可用的标记内容。

没有共享的防护机制来防止空序列的出现。

<a id="solution-7"></a>
## 解决方案

在共享训练工具中增加了对空标记的处理。

在传递到下游处理之前，会验证生成的序列，并在必要时使用安全的降级方案。

<a id="decision-7"></a>
## 决策

共享生成工具不得返回不可用的空标记序列。

防护措施应位于通用工具中，而不是个别 GRPO 或 PPO 实现中。

------------------------------------------------------------------------

<a id="empty-distillation-masks"></a>
# 空蒸馏掩码

<a id="problem-8"></a>
## 问题

一些截断的 SFT 样本在配置的序列长度内可能不包含任何监督标记。

这导致蒸馏 KL 散度计算接收到空的 logit 并失败。

<a id="cause-7"></a>
## 原因

蒸馏损失假设每个训练样本在截断和掩码后至少包含一个监督标记。

这一假设对于短或截断的验证样本并不总是成立。

<a id="solution-8"></a>
## 解决方案

蒸馏损失现在能够检测到空标记选择，并返回与计算图相连的零损失。

<a id="decision-8"></a>
## 决策

在掩码标记选择上运行的训练损失必须安全地处理有效的空选择。

这被视为训练鲁棒性问题，而不是 DirectML 的限制。

------------------------------------------------------------------------

<a id="checkpoint-compatibility-across-training-stages"></a>
# 跨训练阶段的检查点兼容性

<a id="problem-9"></a>
## 问题

DirectML 的修改可能会潜在影响 MiniMind 训练阶段之间的检查点互操作性。

相关阶段包括：

``` text
Pretraining
SFT
DPO
GRPO
PPO
```

<a id="cause-8"></a>
## 原因

各个训练阶段依赖于兼容的架构和状态字典。

设备处理必须与检查点表示独立。

<a id="solution-9"></a>
## 解决方案

检查点兼容性独立于 DirectML 执行进行测试。

<a id="decision-9"></a>
## 决策

DirectML 不得引入特定后端的检查点格式。

保留现有的 MiniMind 检查点语义。

------------------------------------------------------------------------

<a id="training-measurement-introduced-directml-synchronization-overhead"></a>
# 训练测量引入了 DirectML 同步开销

<a id="problem-10"></a>
## 问题

初始真实数据基准测试报告的数值约为：

``` text
5.280 s / iteration
```

这意味着预训练时长的估算约为 `9.70 days / epoch`。

这使得 DirectML 训练看起来比预期慢得多。

<a id="cause-9"></a>
## 原因

训练测量路径在异步 DirectML 执行与 CPU 之间引入了不必要的同步。

频繁将设备生成的值材料化到 CPU 上，导致基准测试不仅测量了实际训练工作量，还测量了同步和仪器开销。

<a id="solution-10"></a>
## 解决方案

已调整关键训练与测量路径，以避免每一步不必要的 DirectML 到 CPU 同步。

随后，使用相同的参考工作负载重新运行了 100 步：

``` text
batch_size = 8
max_seq_len = 340
gradient_accumulation_steps = 8
```

修正后的基准测试测量结果为：

``` text
Average iteration time:    0.482 s
Samples / second:          16.61
Effective tokens / second: 3341.37
Estimated epoch duration:  21.24 h
```

<a id="decision-10"></a>
## 决策

性能测量不得在关键训练路径中引入不必要的同步。

在将缓慢的 DirectML 性能归因于后端本身之前，必须先调查测量、日志记录和同步开销。

------------------------------------------------------------------------

<a id="multiple-gpus-and-directml-device-selection"></a>
# 多块 GPU 与 DirectML 设备选择

<a id="problem-11"></a>
## 问题

开发机器暴露了两个图形适配器：

``` text
Integrated GPU (iGPU)
Dedicated GPU (dGPU)
```

这使得在训练和基准测试期间 DirectML 使用的是哪一块物理 GPU 变得不明确。

<a id="cause-10"></a>
## 原因

集成显卡和专用显卡是两个独立的 DirectML 兼容图形适配器。

因此：

``` text
CPU
 ≠
Integrated GPU
 ≠
Dedicated GPU
```

隐式设备选择无法充分确保执行工作负载的物理 GPU。

<a id="solution-11"></a>
## 解决方案

DirectML 适配器选择必须明确指向目标 GPU。

对于 MiniMind 训练，应选择专用显卡。

概念上：

``` text
Available DirectML adapters
        ↓
Identify dedicated GPU
        ↓
Explicitly select adapter
        ↓
Create DirectML device
        ↓
MiniMind training
```

此时可通过 Windows 任务管理器中的相应适配器验证 GPU 利用率。

<a id="decision-11"></a>
## 决策

专用显卡是 MiniMind 训练的预期设备。

性能基准测试仅应在使用相同明确选择的物理 GPU 时进行比较。

<a id="directml-fp16-adamw-instability"></a>
# DirectML FP16 AdamW 不稳定性

<a id="problem-12"></a>
## 问题

当 AdamW 使用默认 epsilon 时，DirectML FP16 预训练在第一次优化器更新后立即变为非有限值：

``` text
eps = 1e-8
```

静态损失缩放（缩放因子为 `1024`）单独无法解决此失败。

<a id="cause-11"></a>
## 原因

在优化器步骤之前梯度是有限的，这将故障隔离到优化器更新路径，而非前向或反向传播过程。

默认的 AdamW epsilon 对于已验证的纯 FP16 DirectML 优化器路径来说过小。

<a id="solution-12"></a>
## 解决方案

DirectML FP16 路径使用：

``` text
Static loss scale = 1024
AdamW epsilon     = 1e-4
```

该行为集中于共享的训练工具中，并被 DirectML FP16 训练器复用。

<a id="decision-12"></a>
## 决策

DirectML FP16 训练不得在未经验证的情况下使用默认的 AdamW epsilon。

历史直接FP16优化器路径的验证epsilon值为`1e-4`。保留的Dense FP32-master优化器路径使用AdamW `eps = 1e-8`。

------------------------------------------------------------------------

<a id="inconsistent-bounded-run-support-across-trainers"></a>
# 训练器之间存在的有界运行支持不一致

<a id="problem-13"></a>
## 问题

初始M4有界运行实现首先将`--max_steps`添加到预训练阶段，而其他训练器仍期望完成基于完整周期的执行。

这导致一个统一的烟雾测试运行器无法一致地控制所有训练工作负载。

<a id="solution-13"></a>
## 解决方案

有界全局`--max_steps`处理被传播到DirectML FP16烟雾套件所使用的所有训练器中。

新增了一个Python运行器，位于：

``` text
tests/test_all_trainers.py
```

它按顺序启动训练器，并在首次失败时停止。

<a id="decision-13"></a>
## 决策

有界执行是可训练DirectML工作流共享验证策略的一部分。

重型训练器烟雾测试被明确执行，而不是被当作普通的轻量级Pytest测试处理。


------------------------------------------------------------------------

<a id="moe-routing-uses-unsupported-scatter-behavior-on-directml"></a>
# MoE路由在DirectML上使用了不支持的散列行为

<a id="problem-14"></a>
## 问题

上游MoE路由路径在DirectML验证过程中失败。

首次观察到的失败发生在使用`F.one_hot`的辅助专家路由逻辑部分。仅替换该操作使前向传递得以推进，但反向执行仍遇到稀疏路由路径中的不支持散列行为。

<a id="cause-12"></a>
## 原因

上游稀疏MoE实现依赖于索引路由和类似散列的操作，而这些操作在测试的DirectML前向/反向路径中并未完全得到支持。

<a id="solution-14"></a>
## 解决方案

增加了一个专为DirectML设计的无需散列的路由路径。

概念上：

``` text
Gate scores
    ↓
Top-k expert selection
    ↓
Broadcast routing mask
    ↓
Differentiable routing weights
    ↓
Evaluate experts
    ↓
Weighted expert combination
```

CPU和CUDA保留原始的稀疏路由实现。

<a id="decision-14"></a>
## 决策

DirectML使用无需散列的MoE兼容路径。

降级方案优先保证正确执行而非稀疏MoE效率，因此不应被用来推断原生稀疏MoE的性能。

------------------------------------------------------------------------

<a id="agent-rl-compatibility-on-windows-and-directml-fp16"></a>
# 在Windows和DirectML FP16上Agent RL的兼容性

<a id="problem-15"></a>
## 问题

在最终M4烟雾验证期间，Agent RL遇到了两个独立的问题。

首先，Windows DataLoader工作进程无法导入一个本地定义的`collate_fn`。

其次，在修复多进程问题后，第一个Agent训练步骤产生了非有限的损失。

<a id="cause-13"></a>
## 原因

Windows多进程使用进程启动方式，因此DataLoader `collate_fn`必须从模块作用域可导入。

对于数值失败，定向有限值检查显示：

``` text
rewards     = finite
advantages  = finite
behavior    = finite
policy      = non-finite
reference   = non-finite
```

进一步检查确认，在使用`attention_mask=full_mask`的完整序列策略/参考重新计算路径中，策略日志包含NaN值。

轨迹生成本身保持有限。

<a id="solution-15"></a>
## 解决方案

为实现Windows多进程兼容性，将Agent `collate_fn`移至模块作用域。

对于右填充的Agent批次，策略和参考的完整序列重新计算避免了问题性的DirectML FP16注意力掩码路径。

数值敏感的对数概率、KL散度和比率计算在FP32中进行。

<a id="decision-15"></a>
## 决策

保留了针对代理的兼容性处理，因为它能够修复DirectML FP16路径，而无需对MiniMind注意力语义进行更广泛的更改。

在这些变更之后，代理强化学习通过了测试，整合的烟雾套件也成功完成，具体如下：

``` text
Passed: 9/9
```

------------------------------------------------------------------------

<a id="long-run-directml-fp16-pretraining-collapse"></a>
# 长周期DirectML FP16预训练崩溃

<a id="problem-16"></a>
## 问题

一次完整的密集预训练运行保持有限，但后续产生了退化的自我复制模型。

教师强制诊断显示大约为：

``` text
Mean loss:           13.57
Top-1 accuracy:      0.18%
Top-1 repeat rate:  93.48%
```

未经训练的模型并未表现出这种行为。

<a id="cause-14"></a>
## 原因

DirectML FP16路径直接使用AdamW对FP16模型参数进行更新。静态损失缩放保护了梯度，但并未提供传统混合精度优化所使用的FP32主权重。

<a id="solution-16"></a>
## 解决方案

密集DirectML预训练现在使用：

``` text
FP16 model compute
        ↓
FP32 gradient unscale
        ↓
AdamW on FP32 master weights
        ↓
FP32 master weights copied back to FP16 model
```

FP32优化器路径使用AdamW epsilon `1e-8`。

经过100步验证，未训练模型的平均诊断损失从`8.89`提升至`7.48`。在恢复到全局步数`1100`后，平均损失达到`6.75`，Top-1准确率达到`6.30%`，Top-1重复率保持在较低水平`0.89%`。

<a id="decision-16"></a>
## 决策

仅有限的损失值不足以验证DirectML FP16预训练。

密集预训练在优化器更新中使用FP32主权重，且保留教师强制检查点诊断以检测无声训练崩溃。

------------------------------------------------------------------------

<a id="directml-cross-entropy-reduction-and-valid-token-normalization"></a>
# DirectML交叉熵减少与有效令牌归一化

<a id="problem-17"></a>
## 问题

在测试的DirectML FP16路径上，因果语言模型交叉熵使用`ignore_index=-100`未能重现预期的有效令牌归一化。

默认的`reduction="mean"`将有效损失除以所有移位位置，而非仅有效非忽略令牌。

最初使用以下方法作为临时解决方案：

``` python
loss = F.cross_entropy(x, y, ignore_index=-100, reduction="sum")
loss = loss / (y != -100).sum().clamp_min(1)
```

该方法后来也被发现，在测试的DirectML FP16路径上是错误的。

<a id="cause-15"></a>
## 原因

首次通过CPU/DirectML测试确认了当存在被忽略的填充令牌时，DirectML平均减少操作的错误分母。

随后的第二次减少比较评估了相同模型日志与：

``` text
A = DirectML reduction="sum" / valid tokens
B = DirectML reduction="none" → valid tokens → FP32 mean
C = CPU FP32 reference
```

结果为：

``` text
Batch 8:
A = 14.1078
B =  6.6688
C =  6.6715

Batch 32:
A = 10.7772
B =  6.3027
C =  6.3053
```

`B`与CPU FP32参考值在大约`0.003`范围内匹配，而`A`则不匹配。

<a id="solution-17"></a>
## 解决方案

最终的DirectML因果语言模型损失采用逐令牌交叉熵，随后在有效令牌上进行FP32平均：

``` python
token_loss = F.cross_entropy(
    x,
    y,
    ignore_index=-100,
    reduction="none",
)

valid = y != -100
loss = token_loss[valid].float().mean()
```

应用该变更后，模型在批大小为32的参考测试上的训练损失变为：

``` text
Model outputs.loss:             6.30274916
DirectML none → FP32 mean:      6.30274916
CPU FP32 reference:             6.30530691
|model - CPU|:                  0.0025577545
```

<a id="decision-17"></a>
## 决策

DirectML因果语言模型训练不得依赖后端默认的平均减少操作，或测试的FP16 `sum / valid_tokens`临时解决方案。

保留的实现方式采用`reduction="none"`计算逐令牌损失，并在有效令牌上进行FP32平均。

------------------------------------------------------------------------

<a id="pretraining-diagnostic-loss-aggregation-mismatch"></a>
# 预训练诊断损失聚合不匹配

<a id="problem-18"></a>
## 问题

在第一个完整修正的密集预训练周期结束后，`diagnose_pretrain.py`报告了聚合训练路径损失值约为`12–13`，尽管Top-1准确率有所提升且重复率较低。

<a id="cause-16"></a>
## 原因

初步调查指出存在诊断聚合问题，但专门的减少测试后来表明，当前`reduction="sum" / valid_tokens`训练路径本身在测试的DirectML FP16后端上是错误的。

由于手动标记损失值接近独立的CPU FP32参考值，因此该诊断的标记损失仍然具有参考价值。

<a id="solution-18"></a>
## 解决方案

将诊断更改为使用批处理评估，模型损失实现被替换为逐标记DirectML交叉熵，随后进行FP32有效标记平均。

关键的批处理32验证结果为：

``` text
Model outputs.loss:        6.30274916
Manual DirectML loss:      6.30274916
CPU FP32 reference:        6.30530691
Difference vs CPU:         0.0025577545
```

<a id="decision-18"></a>
## 决策

检查点诊断必须将模型训练路径与同一logits上的独立参考进行比较。

`reduction="none"` → FP32有效令牌均值实现是保留的DirectML损失路径。
------------------------------------------------------------------------

<a id="directml-bf16-tensor-execution-is-unsupported-in-the-current-stack"></a>
# DirectML BF16张量执行在当前堆栈中不受支持

<a id="problem-19"></a>
## 问题

一个类似上游的DirectML预训练实验尝试直接使用`bfloat16`模型张量。

运行在第一个训练步骤之前失败，错误信息为：

``` text
Invalid or unsupported data type BFloat16
```

来自DirectML后端路径。

<a id="cause-17"></a>
## 原因

测试的PyTorch / `torch-directml` / `PrivateUse1`执行栈不接受该工作负载所需的BF16张量类型。

该结果确立了当前软件栈的一个限制。它本身并不能证明物理GPU硬件无法执行BF16算术运算。

该实验也与上游CUDA混合精度路径不同：上游将模型参数保留在FP32中，并使用BF16自动转换来支持兼容的计算操作，而DirectML实验则必须直接请求BF16张量，因为目前尚无经过验证的DirectML自动转换路径可用。

<a id="solution-19"></a>
## 解决方案

当前MiniMind训练路径不使用DirectML BF16。

保留的混合精度策略仍为：

``` text
FP16 model compute
        ↓
FP32 master weights
        ↓
FP32 AdamW update
        ↓
FP32 master → FP16 model synchronization
```

<a id="decision-19"></a>
## 决策

不要将DirectML BF16视为当前环境中CUDA BF16自动转换的可用替代方案。

------------------------------------------------------------------------

<a id="fast-direct-fp16-optimizer-re-test"></a>
# 快速Direct FP16优化器重测

<a id="problem-20"></a>
## 问题

为减少FP32主权重的开销，已重新测试使用AdamW直接在FP16模型参数上进行的密集预训练实验。

实时训练损失保持有限并持续下降，但检查点诊断显示学习效果不佳，随后出现快速自我重复现象。

在样本`0–255`上的结果为：

``` text
Step 1000:
loss   = 9.2573
Top-1  = 0.31%
repeat = 0.29%

Step 2000:
loss   = 8.3102
Top-1  = 0.92%
repeat = 1.20%

Step 5000:
loss   = 7.3196
Top-1  = 1.80%
repeat = 39.24%
```

真实数据重复率保持在大约`0.29%`的水平。

<a id="cause-18"></a>
## 原因

该实验路径去除了FP32主参数，并直接将AdamW更新应用于FP16参数。

结果进一步证实了早期发现：静态损失缩放可以保护反向信号，但无法提供足够精度以支持可靠的长期FP16参数更新。

<a id="solution-20"></a>
## 解决方案

快速直接-FP16优化路径未被保留。

Dense DirectML预训练仍使用FP16模型计算，主权重为FP32，AdamW更新为FP32。

<a id="decision-20"></a>
## 决策

训练损失的下降不足以验证直接FP16优化的有效性。

保留的Dense DirectML FP16预训练架构仍需FP32主权重。

------------------------------------------------------------------------

<a id="diagnostic-cpu-synchronization-limited-to-logging-intervals"></a>
# 诊断CPU同步限制在日志间隔内

<a id="problem-21"></a>
## 问题

稳定的FP16 + FP32主权重训练器仍比实际优化更新所需更频繁地执行诊断DirectML到CPU的同步。

示例包括：

``` text
finite-loss read on every batch
gradient-norm scalar read on optimizer updates
```

这些标量读取迫使CPU等待异步DirectML操作。

<a id="solution-21"></a>
## 解决方案

一个优化后的训练器变体保持完整的FP32主权重优化路径不变，但禁用每步标量材料化以用于诊断检查。

损失和最新的梯度范数仅在以下情况被传输到CPU并检查：

``` text
step % log_interval == 0
```

或在最终日志点时。

梯度裁剪本身仍每一步优化更新都执行，且FP32主权重仍每一步优化后同步回FP16模型。

<a id="decision-21"></a>
## 决策

当可能时，诊断CPU同步应遵循`log_interval`而非优化器的更新节奏。

一个`1000`步长的基准测试确认了从`2.0038 s/step`到`1.9307 s/step`的吞吐量提升，对应于`1.038×`的加速和`3.65%`更低的总运行时间。

该吞吐量影响已在测试配置中得到验证。在优化训练器被视为最终训练参考之前，仍需进行检查点质量验证。

------------------------------------------------------------------------

<a id="m5-final-blocking-issue--directml-fp16-numerical-divergence"></a>
# M5最终阻塞问题——DirectML FP16数值发散

<a id="problem-22"></a>
## 问题

即使在ROCm环境下评估，最终DirectML训练的检查点仍存在不一致性，而ROCm训练的检查点生成的文本是连贯的。

在相同的`1,000`示例（`204,327`个有效标记）上，使用ROCm进行评估：

```text
DirectML-trained checkpoint: loss 6.880233, perplexity 972.8529
ROCm-trained checkpoint:     loss 1.846587, perplexity   6.3382
```

这证实了DirectML检查点本身已退化；该问题不仅限于DirectML生成。

<a id="final-diagnosis"></a>
## 最终诊断

使用相同的DirectML训练检查点和相同的固定批次（`873`个有效标记）：

```text
DirectML FP16: 5.733902
ROCm FP16:     6.230469

DirectML FP32: 6.231627
ROCm FP32:     6.231628
```

DirectML FP16未使用SDPA仅产生非有限的logits，而ROCm保持稳定。相比之下，DirectML和ROCm的FP32结果在大约`1e-6`范围内一致。

DirectML FP16损失本身与从相同DirectML logits通过CPU FP32交叉熵重新计算的结果（`5.733902` vs `5.736305`）内部一致。因此，保留的交叉熵减少并非剩余的根本问题：FP16前向路径已经产生了实质不同的logits。

<a id="decision-22"></a>
## 决策

在测试堆栈中，最终阻塞被分类为**DirectML FP16数值发散**。不再寻求额外的DirectML训练工作绕行方案。

DirectML实现仍被归档以保持兼容性、回归测试和可重现性。活跃的MiniMind训练将继续使用**ROCm**。
