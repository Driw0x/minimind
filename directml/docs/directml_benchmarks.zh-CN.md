[English](directml_benchmarks.md) | [Français](directml_benchmarks.fr.md) | [简体中文](directml_benchmarks.zh-CN.md)

[项目 README](../../README.zh-CN.md)

<a id="minimind-----directml-benchmarks"></a>
# MiniMind --- DirectML 性能基准

> **历史 DirectML 归档 — 更新于 2026-09-24**
>
> DirectML 已不再作为 MiniMind 训练的主后端。M5 最终验证发现了一种特定于测试的 DirectML FP16 路径的数值偏差。以下历史结果为可复现性和工程参考保留；ROCm 开发情况另作记录。

本文件记录了 MiniMind 的实验性 DirectML 兼容性与性能测试。

目标是确定适用于目标 DirectML 硬件的实用训练配置。

这些结果描述了测试环境，不应被解释为 DirectML 的通用性能限制。

------------------------------------------------------------------------

<a id="batch-size-and-sequence-length"></a>
# 批大小与序列长度

<a id="motivation"></a>
## 动机

`batch_size` 和 `max_seq_len` 直接影响训练内存需求和执行时间。

增大 `batch_size` 会同时处理更多的序列。

增大 `max_seq_len` 会增加每个序列处理的标记数量，并显著增加 Transformer 注意力成本。

概念上：

``` text
batch_size ↑
     +
max_seq_len ↑
     ↓
Higher memory usage
Higher computation cost
Longer training steps
```

因此，上游 MiniMind 的默认设置不能自动假设在 DirectML 硬件上是可行的。

------------------------------------------------------------------------

<a id="methodology"></a>
# 方法论

引入了一个兼容性基准测试，用于测试若干组合的：

``` text
batch_size
max_seq_len
```

每种配置均执行一次简短的实际训练运行，而非完整训练会话。

结果被分类为：

``` text
PASS
FAIL
TIMEOUT
```

这使得在运行完整训练管道之前就能识别出不稳定或实际运行缓慢的配置。

------------------------------------------------------------------------

<a id="initial-pretraining-benchmark"></a>
# 初始预训练基准

初始`train_pretrain.py`基准测试结果为：

    Batch size   Sequence length Result
  ------------ ----------------- ---------------------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 FAIL (`0xC0000409`)
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 TIMEOUT
             8               128 TIMEOUT

`2 × 256`最初被选为一个保守的配置，用于验证剩余训练管道。

这些结果仅适用于所测试的预训练配置，不应自动推广到每一个MiniMind训练器。

------------------------------------------------------------------------

<a id="unexpected-result"></a>
# 意外结果

初始基准测试产生了一个看似不一致的观察结果：

``` text
1 × 340 → FAIL
2 × 340 → PASS
```

起初，这似乎表明DirectML的稳定性并不随批大小和序列长度的增加而单调提升。

然而，后来发现机器存在以下两个问题：

``` text
Integrated GPU
Dedicated GPU
```

以及显式物理GPU选择尚未完全受控。

因此，这一观察结果不能安全地解释为DirectML在批大小或序列长度方面的固有行为。

它仍被记录下来，因为这帮助识别了多GPU设备选择问题。

------------------------------------------------------------------------

<a id="multi-gpu-impact-on-benchmarking"></a>
# 多GPU对基准测试的影响

只有在执行环境保持一致的情况下，基准测试结果才能直接进行比较。

特别是：

``` text
Same model
    +
Same trainer
    +
Same DirectML adapter
    +
Same benchmark conditions
    ↓
Comparable results
```

因此，在建立参考性能限制之前，必须明确选择物理DirectML适配器。

在集成显卡和专用显卡上获得的结果不应合并到一个兼容性表格中。

------------------------------------------------------------------------

<a id="reference-benchmark"></a>
# 参考基准

在明确选择专用显卡后，兼容性基准重新运行于：

``` text
Device: directml:1
```

得到的兼容性矩阵为：

    Batch size   Sequence length Result
  ------------ ----------------- --------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 PASS
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 PASS
             4               340 PASS
             8               128 PASS
             8               256 PASS
             8               340 PASS
            16               128 PASS
            16               256 PASS
            16               340 PASS
            32               128 PASS
            32               256 FAIL

当批大小为`32`时，由于`32 × 256`失败，基准测试停止测试更大的序列长度。

这确立了在明确选择专用显卡上的显著更高的验证兼容范围，相较于初始基准所建议的范围。具体而言，所有测试配置直到`16 × 340`均通过，而`32 × 128`通过，`32 × 256`未通过。

初始基准作为历史发展数据仍然具有参考价值，因为它记录了在物理显卡选择被控制之前观察到的行为。

------------------------------------------------------------------------

<a id="benchmark-vs-real-training"></a>
# 基准测试与实际训练

通过兼容性基准的配置并不能保证在整个训练运行期间保持稳定。

该基准仅执行一个短暂的训练工作负载。其目的是识别那些立即失败或明显与测试DirectML环境不兼容的配置。

完整的训练运行可能表现不同，因为它运行时间显著更长，并可能遇到不同的内存和执行条件。

因此：

``` text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

基准测试结果应被视为兼容性指示而非保证安全的训练配置。

最终的`batch_size`和`max_seq_len`必须始终通过实际训练运行进行验证。

------------------------------------------------------------------------

<a id="configuration-selection"></a>
# 配置选择

最终的DirectML训练配置应平衡：

``` text
Stability
    +
Training speed
    +
Memory usage
```

而不是简单地选择首次成功启动的最大配置。

当相关的执行条件发生变化时，应重复进行基准测试，包括：

-   物理显卡；
-   模型大小；
-   训练器；
-   DirectML/PyTorch环境；
-   其他显著影响内存或计算需求的变化。

------------------------------------------------------------------------

<a id="real-training-performance"></a>
# 实际训练性能

通过兼容性基准并不意味着配置在实际训练中是稳定或高效的。

因此，实际预训练基准使用了真实的MiniMind预训练数据集和完整规模的预训练模型。

测试环境使用了：

-   设备：`directml:1`
-   PyTorch设备表示：`privateuseone:1`
-   数据集：`pretrain_t2t_mini.jsonl`
-   数据样本：`1,270,238`
-   隐藏层大小：`768`
-   隐藏层数量：`8`
-   模型参数：约 `63.91M`
-   最大序列长度：`340`
-   梯度累积步数：`8`

基准测试执行了真实的前向传播、反向传播、梯度裁剪以及AdamW优化器步骤。

------------------------------------------------------------------------

<a id="real-training-stability"></a>
## 真实训练稳定性

真实数据训练揭示了在短兼容性基准中不可见的稳定性限制。

<a id="batch-size-32"></a>
### 批大小 32

配置：

    batch_size = 32
    max_seq_len = 340

训练工作负载成功启动，但在第二次反向传播期间因GPU内存不足而失败。

结果：

    32 × 340 → OOM

因此，该配置在测试硬件上不被视为真实训练的稳定配置。

<a id="batch-size-16"></a>
### 批大小 16

配置：

    batch_size = 16
    max_seq_len = 340
    accumulation_steps = 8

基准测试成功完成了第一个梯度累积周期和优化器步骤。

随后在一次反向传播期间因GPU内存不足而失败。

结果：

    16 × 340 → OOM after first accumulation cycle

这表明成功完成若干训练步骤并不能保证持续的内存稳定性。

<a id="batch-size-8"></a>
### 批大小 8

配置：

    batch_size = 8
    max_seq_len = 340
    accumulation_steps = 8

一个有界100步的真实训练基准测试成功完成，未出现GPU内存溢出错误。

前8步用作预热，剩余92步为实际测量步数。

结果：

    8 × 340 → PASS (100-step bounded run)

这是目前成功完成有界真实训练验证的最大的测试配置。

------------------------------------------------------------------------

<a id="initial-real-training-performance-measurement"></a>
# 初始真实训练性能测量

初始100步真实训练测量使用了：

    Device:                   directml:1
    PyTorch device:           privateuseone:1
    Dataset samples:          1,270,238
    Batch size:               8
    Max sequence length:      340
    Gradient accumulation:    8
    Warmup steps:             8
    Total benchmark steps:    100
    Measured steps:           92

测量性能：

    Average iteration time:   5.280 s
    Samples / second:         1.52
    Effective tokens / sec:   304.81
    Padded tokens / sec:      515.17
    Steps / epoch:            158,780
    Estimated epoch duration: 232.87 h
    Estimated epoch duration: 9.70 days
    Peak dedicated GPU memory: 11,623.12 MB

专用GPU内存通过参考基准期间Windows GPU进程内存计数器外部测量。

最大专用GPU内存达到约11.62 GB。

训练工作负载在运行的大部分时间里保持在约11.6 GB的专用GPU内存水平，表明该参考配置运行时接近可用VRAM限制。

最初的测量表明，真实MiniMind预训练在测试的DirectML硬件上从技术上是可执行的，但速度极慢。

在测得的吞吐量方面：

    1 epoch  ≈ 9.70 days
    2 epochs ≈ 19.4 days

这些数值是基于首次有界基准测试的初步估算值。后来发现，它们受到测量路径中同步开销的强烈影响，因此这里保留作为历史M4结果。

------------------------------------------------------------------------

<a id="corrected-real-training-benchmark"></a>
# 修正后的实际训练基准

进一步的M4调查表明，最初的计时路径在DirectML与CPU之间引入了显著的同步开销。

在移除关键训练和测量路径中不必要的同步后，同一参考工作负载被重新运行了100步。

参考配置：

    Device:                   directml:1
    PyTorch device:           privateuseone:1
    Model dtype:              float16
    Loss scale:               1024.0
    AdamW epsilon:            0.0001
    Dataset samples:          1,270,238
    Batch size:               8
    Max sequence length:      340
    Gradient accumulation:    8
    Warmup steps:             8
    Total benchmark steps:    100
    Measured steps:           92

修正后的测量性能：

    Average iteration time:   0.482 s
    Samples / second:         16.61
    Effective tokens / sec:   3341.37
    Padded tokens / sec:      5647.46
    Steps / epoch:            158,780
    Estimated epoch duration: 21.24 h
    Estimated epoch duration: 0.89 days

修正后的基准测试成功完成了全部100步。

与初始测量相比：

    Average iteration time: 5.280 s → 0.482 s
    Samples / second:       1.52    → 16.61
    Effective tokens / sec: 304.81  → 3341.37
    Estimated epoch:        9.70 d  → 0.89 d

迭代时间的提升约为10.95倍。

为获得这一提升，模型架构、数据集、物理批次大小、序列长度和梯度累积配置均未被减少。

该结果表明，初始性能估算主要由执行和测量过程中的同步开销主导，而非真实稳态DirectML训练成本本身。

AdamW `aten::lerp.Scalar_out` CPU回退机制仍然存在，但它并未阻止修正后的基准测试在测试配置下维持亚秒级迭代。

------------------------------------------------------------------------

<a id="compatibility-benchmark-vs-real-training"></a>
# 兼容性基准测试与实际训练基准测试

兼容性基准测试和实际训练基准测试具有不同的用途。

兼容性基准测试显示：

    8 × 340  → PASS
    16 × 340 → PASS
    32 × 128 → PASS
    32 × 256 → FAIL

然而，真实数据训练显示：

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

`32 × 340`配置未包含在已验证的兼容性矩阵中，因为对批次大小`32`的测试在`32 × 256`失败后终止。

因此重要的区别在于`16 × 340`的行为。

它通过了短时兼容性基准测试，但在长时间真实数据执行过程中失败。

这证实了：

    Compatibility PASS
            ≠
    Sustained training stability

短时兼容性基准对于检测即时后端和内存故障仍然有用，但最终训练配置必须使用真实数据和多个梯度累积周期进行验证。

------------------------------------------------------------------------

<a id="throughput-and-batch-size"></a>
# 吞吐量和批次大小

实际训练实验表明，短时兼容性基准接受的最大配置未必是最佳实用配置。

真实数据稳定性结果仍为：

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

在M4调查的当前阶段，`batch_size = 8`与`max_seq_len = 340`以及梯度累积`8`是首选的实际训练参考配置。

在修正测量路径中的同步开销后，该配置实现了大约：

    0.482 s / iteration
    16.61 samples / second
    3341.37 effective tokens / second

该选择基于真实数据的稳定性以及修正后的测量吞吐量，而非兼容性基准所接受的最大配置。

------------------------------------------------------------------------

<a id="directml-optimizer-limitation"></a>
# DirectML优化器限制

在实际训练过程中，PyTorch报告了以下操作的DirectML降级：

    aten::lerp.Scalar_out

该操作目前不被DirectML后端支持，在AdamW优化器步骤中会降级到CPU执行。

该降级不会阻止训练运行，但可能会引入同步和性能开销。

参考基准包含此降级，因此反映了测试训练操作的实际端到端行为，而非纯粹的DirectML GPU执行。

修正后的基准仍包含该降级，同时平均维持了大约`0.482 s/iteration`的性能。

因此，该降级仍然是DirectML的一个已知限制，但初始的较大`5.280 s/iteration`结果不应仅归因于此降级。

------------------------------------------------------------------------

<a id="intermediate-m4-performance-conclusion"></a>
# 中间M4性能结论

M4实际训练调查确立了几个重要观察结果。

首先，兼容性基准不足以确定长期训练稳定性。

其次，真实数据验证确认了`8 × 340`配合梯度累积`8`在100步有界运行中是稳定的，而更大的物理批次则遭遇了GPU内存故障。

第三，初始的`5.280 s/iteration`性能测量并不能代表实际稳态训练成本。测量路径引入了显著的DirectML到CPU的同步开销。

在修正此问题后，相同的参考配置完成了一次新的100步基准测试，结果为：

    16.61 samples/s
    3341.37 effective tokens/s
    0.482 s/iteration

并估计整个周期耗时约为：

    21.24 hours
    0.89 days

修正后的结果比初始测量快约10.95倍。

全规模训练因此比M4最初的测量结果要实际得多，尽管100步基准尚未足以证明持续全周期的稳定性。

在该阶段，进一步的M4验证聚焦于：

-   超出有限100步运行的持续训练稳定性；
-   在更长运行中验证修正后的吞吐量；
-   持续观察GPU内存行为；
-   最终评估DirectML实际可行性的结论；
-   对剩余性能限制的整合。

------------------------------------------------------------------------

<a id="final-m4-fp16-validation"></a>
# 最终M4 FP16验证

M4以一种专为DirectML设计的FP16训练路径结束，该路径旨在避免使用默认AdamW epsilon时观察到的数值不稳定性。

已验证的DirectML FP16设置为：

``` text
Model dtype:              float16
Static loss scale:        1024
AdamW epsilon:            1e-4
Reference device:         directml:1
PyTorch representation:   privateuseone:1
```

使用默认AdamW epsilon（`1e-8`）在第一次优化器更新后立即产生非有限值。仅使用静态损失缩放无法解决该问题。使用`1e-4`作为DirectML FP16 AdamW优化器稳定了优化器路径。

随后，使用完整的`768 / 8-layer`模型、梯度累积`8`以及有限的9步运行重新运行了DirectML FP16兼容性矩阵，以确保每种配置都跨越了首次优化器更新边界。

    Batch size   Sequence length Result
  ------------ ----------------- --------
             1                64 PASS
             1               128 PASS
             1               256 PASS
             1               340 PASS
             2               128 PASS
             2               256 PASS
             2               340 PASS
             4               128 PASS
             4               256 PASS
             4               340 PASS
             8               128 PASS
             8               256 PASS
             8               340 PASS
            16               128 PASS
            16               256 PASS
            16               340 PASS
            32               128 PASS
            32               256 PASS
            32               340 PASS

该矩阵验证了在首次优化器更新边界处的短周期有限运行。但它**并不**意味着所有大配置都适合持续训练。早期的实际数据测试仍然具有相关性：在更长的运行中，大物理批次仍可能耗尽VRAM。

------------------------------------------------------------------------

<a id="sustained-pretraining-validation"></a>
# 持续预训练验证

最终的持续M4验证使用了：

``` text
Device:                 directml:1
Model dtype:            float16
Batch size:             8
Max sequence length:    340
Gradient accumulation:  8
Static loss scale:      1024
AdamW epsilon:          1e-4
Maximum steps:          1000
```

该运行成功完成了所有`1000`训练步骤。

由于`1000`可被累积因子`8`整除，该运行执行了`125`次优化器更新。

在持续运行期间未观察到NaN或Inf损失。记录的损失值在第`1000`步时仍保持有限。

最终训练的预计耗时约为：

``` text
1360 minutes
22.67 hours / epoch
```

这与修正后的100步基准估计值（约为`21.24 hours / epoch`）一致。

因此，该持续运行证实了修正后的吞吐量在超出短100步基准后是具有代表性的，并且所选DirectML FP16参考配置在多次优化器更新中保持数值稳定性。

------------------------------------------------------------------------

<a id="final-fp16-gpu-memory-measurement"></a>
# 最终FP16 GPU内存测量

在最终FP16参考运行期间外部监控到的GPU内存数据为：

``` text
Peak dedicated VRAM:     7,769.62 MB
Median active VRAM:      7,751.02 MB
Average active VRAM:     7,218.87 MB
Typical active plateau:  ~7,751 MB
```

与早期FP32峰值（`11,623.12 MB`）相比，FP16将专用GPU峰值内存降低了约：

``` text
3,853.50 MB
33.15%
```

因此，最终的FP16参考工作负载大约使用了`7.59 GiB`峰值专用显存，而不是早期FP32测量中使用的大约`11.35 GiB`显存。

------------------------------------------------------------------------

<a id="cross-trainer-smoke-validation"></a>
# 跨训练器烟雾验证

DirectML FP16路径和受限的`--max_steps`执行被传播到M4烟雾运行器所使用的主可训练工作流中：

``` text
Pretrain
Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

整合后的`tests/test_all_trainers.py`运行器成功完成了所有包含的训练器烟雾测试。

DPO仍属于之前验证过的MiniMind训练管道，但不包含在本M4 FP16烟雾运行器结果中。

------------------------------------------------------------------------

<a id="final-m4-conclusion--historical-superseded"></a>
# 最终M4结论 — 历史性、已过时

在M4阶段结束时，可用的执行能力、吞吐量、内存以及烟雾测试证据表明，DirectML训练对于在参考硬件上测试的MiniMind配置来说是实际可行的。**这一结论后来被完整的训练模型质量验证所取代。**

推荐的实际训练基线是：

``` text
Device:                 directml:1
Model dtype:            float16
Batch size:             8
Max sequence length:    340
Gradient accumulation:  8
Static loss scale:      1024
AdamW epsilon:          1e-4
```

最终证据结合了：

``` text
Short compatibility tests
        +
100-step performance benchmark
        +
1000-step sustained pretraining validation
        +
GPU memory monitoring
        +
Cross-trainer smoke validation
```

在M4阶段，后端限制和CPU降级并未阻止训练循环的执行。后来的检查点质量和定性生成检查表明，训练执行成功和合理的损失指标不足以确立正确的完整训练行为。


------------------------------------------------------------------------

<a id="moe-directml-validation"></a>
# MoE DirectML验证

最终的M4烟雾工作流被扩展以覆盖MiniMind MoE配置，以及密集模型。

测试的`768 / 8-layer` MoE模型报告大约为：

``` text
Model Params: 198.42M-A63.94M
```

DirectML在上游稀疏专家路由路径中暴露出一个兼容性问题。受影响的路由操作需要散列行为，而DirectML前向/反向路径无法可靠支持该行为。

因此引入了一条专用于DirectML的无散列路由路径。CPU和CUDA仍保留原始的稀疏路由行为。

DirectML兼容性路径会计算所有专家，并使用选定的路由权重组合其输出。该路径旨在作为功能兼容性的后备方案，不应被解释为原生稀疏MoE性能。

以下工作流成功完成：

``` text
MoE Pretrain
MoE Full SFT
Dense student ← MoE teacher Distillation
```

另外还保留了一个独立的、有限的20步MoE预训练运行。

在相同的小规模验证设置下进行的密集模型与MoE的短期对比，产生了初步的运行时间比率约为：

``` text
MoE / Dense ≈ 1.158×
```

或大约`15.8%`额外的运行时间。

该数值仅是短期相对观察，不被视为完整的MoE性能基准。

------------------------------------------------------------------------

<a id="final-cross-trainer-smoke-result"></a>
# 最终跨训练器烟雾结果

在增加密集模型和MoE覆盖并解决剩余代理兼容性问题后，最终的DirectML训练器烟雾运行器完成了：

``` text
All trainer smoke tests passed
Passed: 9/9
```

最终运行器验证了：

``` text
Dense Pretrain
Dense Full SFT
MoE Pretrain
MoE Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

DPO仍属于之前验证过的训练管道，但不包含在本特定的9项测试运行器中。

------------------------------------------------------------------------

<a id="fp32-master-weight-and-cross-entropy-validation"></a>
# FP32主权重和交叉熵验证

后来的检查点质量调查表明，早期纯FP16 DirectML优化器路径虽然可以保持有限，但仍会收敛到一个退化的自我复制模型。

因此，密集预训练被修改为使用FP16计算，主权重为FP32，AdamW更新为FP32。

一项独立的损失调查显示，在测试的FP16路径中存在两个DirectML减少问题：

1. `reduction="mean"`与`ignore_index=-100`未仅对有效标记进行归一化；
2. 初始的`reduction="sum" / valid_tokens`补救措施也未能重现CPU FP32参考值。

决定性的相同对数减少比较结果为：

``` text
Batch 8:
sum / valid:             14.1078
none → FP32 valid mean:   6.6688
CPU FP32 reference:       6.6715

Batch 32:
sum / valid:             10.7772
none → FP32 valid mean:   6.3027
CPU FP32 reference:       6.3053
```

因此，最终的损失实现采用：

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

变更后，批大小为32的模型损失与DirectML每标记FP32平均值完全一致，并且与CPU FP32参考值保持在大约`0.003`范围内：

``` text
Model outputs.loss:        6.30274916
DirectML none/FP32 mean:   6.30274916
CPU FP32 reference:        6.30530691
```

早期在第100步和第1000步的运行验证了FP32主权重优化器的方向，但这些训练是在最终交叉熵减少修正之前完成的，因此被保留为中间调查结果，而非最终训练质量参考。

------------------------------------------------------------------------

<a id="intermediate-full-epoch-dense-pretraining-validation"></a>
# 中间全周期密集预训练验证

在FP32主权重变更以及初始`reduction="sum" / valid_tokens`交叉熵补救措施之后，密集预训练从头开始重新启动，仅调整了DirectML精度/后端路径：

``` text
Device:                 directml:1
Model dtype:            float16
Epochs:                 2
Batch size:             32
Learning rate:          5e-4
Max sequence length:    340
Gradient accumulation:  8
Hidden size:            768
Hidden layers:          8
Dense / use_moe:        0
```

第一个完整的周期成功完成。

这一点很重要，因为早期的`32 × 340 → OOM`观察是在最终DirectML训练路径建立之前获得的。它被保留为历史基准证据，但不再描述当前修正后的实现情况。

在样本`0–255`上进行的检查点验证结果为：

``` text
Valid tokens:                 51,596
Mean batch loss:              6.0922
Token-weighted global loss:   6.1017
Minimum batch loss:           5.6996
Maximum batch loss:           6.4880
Top-1 accuracy:               15.07%
Top-1 repeat rate:             0.83%
Mean entropy:                  5.3160
```

模型训练损失与同一批次上独立手动计算的标记归一化交叉熵之间的直接比较结果为：

``` text
Model loss:   6.48799419
Manual loss:  6.48799419
Difference:   0.0000000000
MATCH
```

检查点显示了有用的下一个标记学习效果和较低的重复率，但后续的减少比较证明，该运行在反向传播期间仍使用了错误的DirectML `reduction="sum" / valid_tokens`损失。

因此，它被保留为证据，证明物理`batch_size = 32`能够在当前内存路径上完成一个完整周期，但它不是最终训练质量的参考，不应作为最终SFT管道的基础检查点。

最终的密集预训练必须从头开始重新启动，采用保留的`reduction="none"` → FP32有效标记平均实现。

------------------------------------------------------------------------

<a id="final-loss-path--1200-step-validation"></a>
# 最终损失路径 — 1200步验证

通过第`1200`步使用FP16计算、FP32主权重以及每标记交叉熵后接FP32有效标记平均，验证了最终的DirectML损失路径。

两个独立的256样本数据集区域产生：

|样本|训练损失|手动损失|差异|Top-1|重复|
|---|---:|---:|---:|---:|---:|
|0–255| 6.3752 | 6.3779 | 0.00264 | 8.66% | 1.96% |
|100000–100255| 6.3568 | 6.3595 | 0.00263 | 8.91% | 2.08% |

训练路径和独立计算的标记损失在大约`0.003`范围内保持一致。

与第`200`步相比，损失从大约`7.18`降低到`6.36`，而Top-1准确率从大约`5.1%`提升到`8.8%`。

Top-1 重复率与历史自复制失败相比仍保持较低水平，在两个评估区域中，Top-1 正确且重复率均约为 `0.02%`。

未观察到历史自复制崩溃的复发。

该结果验证了最终 FP16 计算 + FP32 主权重 + 每个标记 FP32 有效均值损失路径在初始短检查点测试之后的可行性。随后成功完成了全周期验证。

------------------------------------------------------------------------

<a id="final-loss-path--full-epoch-1-validation"></a>
# 最终损失路径 — 全周期 1 验证

最终 DirectML 密集预训练路径使用 FP16 计算、FP32 主权重以及每个标记的交叉熵损失后接 FP32 有效标记平均，完成了第一个完整周期的训练。

|样本|训练损失|手动损失|差异|Top-1|重复|熵|
|---|---:|---:|---:|---:|---:|---:|
|0–255| 6.0612 | 6.0637 | 0.00250 | 13.71% | 0.81% | 5.3487 |
|100000–100255| 6.0177 | 6.0201 | 0.00246 | 13.80% | 0.71% | 5.3436 |

在完成一个完整周期后，训练路径与独立计算的标记损失保持在大约 `0.0025` 的范围内对齐。

与步骤 `1200` 相比，训练损失从大约 `6.36` 降低至 `6.02–6.06`，而 Top-1 准确率从大约 `8.8%` 提升至 `13.7–13.8%`。

Top-1 正确且重复率仍然非常低（`0.06%` 和 `0.03%`），且历史自复制崩溃未再次发生。

这是首次成功完成保留的 DirectML 密集预训练路径的全周期验证。

<a id="final-epoch-2-validation"></a>
## 最终周期 2 验证

相同的修正运行成功完成了第二个周期。更广泛的评估覆盖了四个数据区域的 4,096 个样本：

|样本区域|有效标记|训练损失|手动损失|Top-1|重复|
|---|---:|---:|---:|---:|---:|
|0–1023| 205,439 | 6.3317 | 6.3343 | 12.23% | 0.69% |
|300000–301023| 116,790 | 6.3619 | 6.3645 | 10.57% | 0.56% |
|600000–601023| 92,088 | 5.9203 | 5.9288 | 13.22% | 0.96% |
|1000000–1001023| 261,529 | 6.0706 | 6.0731 | 12.94% | 0.73% |

在这些区域中，训练损失与独立手动损失保持紧密对齐，重复率始终低于 `1%`。

最终的定性生成检查产生了多样化的输出，而非历史性的重复标记崩溃。文本仍保持薄弱和碎片化，这在监督微调之前是预期的。

修正后的两周期密集预训练运行在数值上已完成，但由此得到的本地训练模型未能产生可接受的连贯生成。因此，该路径**未被保留作为全监督微调的最终预训练基础**。DirectML 训练路径已关闭，项目正转向 ROCm。
------------------------------------------------------------------------

<a id="fast-directml-fp16-re-test"></a>
# 快速 DirectML FP16 重测

测试了一条独立的实验性预训练路径，以确定是否可以移除 FP32 主权重机制以提升 DirectML 的吞吐量。

该实验路径保留了：

``` text
FP16 model parameters
FP16 forward / backward
Static loss scale = 1024
AdamW directly on FP16 parameters
```

并有意移除了 FP32 主参数和主权重到模型的复制操作。

在相同 `0–255` 评估区域的检查点诊断显示：

|步骤|训练路径损失|手动损失|Top-1|重复|熵|
|---:|---:|---:|---:|---:|---:|
| 1000 | 9.2573 | 9.2609 | 0.31% | 0.29% | 7.0694 |
| 2000 | 8.3102 | 8.3136 | 0.92% | 1.20% | 7.2550 |
| 5000 | 7.3196 | 7.3225 | 1.80% | 39.24% | 7.3674 |

真实数据重复率保持在大约 `0.29%` 的水平。

尽管在 `1000` 到 `5000` 步骤之间诊断损失有所改善，但 Top-1 重复率急剧上升至 `39.24%`。在步骤 `5000`，大约 `39.10%` 的预测结果仅为重复错误。

这重现了在早期纯 FP16 优化器路径中已观察到的重要长期失败模式：有限且递减的损失并不能证明直接在 FP16 参数上进行更新的模型实际上在学习一个可用的模型。

确切的失败轨迹与早期 `93.48%` 自复制崩溃并不完全相同，但两个实验均表明，直接在 FP16 模型参数上进行 AdamW 更新，并不能作为可靠的预训练路径保留下来。

因此，FP32 主权重路径仍为参考架构：

``` text
FP16 forward / backward
        ↓
FP32 gradient conversion and unscale
        ↓
FP32 gradient clipping
        ↓
AdamW on FP32 master weights
        ↓
FP32 master → FP16 model synchronization
```

------------------------------------------------------------------------

<a id="stable-master-weight-synchronization-optimization"></a>
# 主权重同步优化稳定

性能优化现在聚焦于在保留已验证的 FP32 主权重架构的同时，降低诊断同步开销。

之前的稳定训练器比训练所需更频繁地在 CPU 上生成 DirectML 产生的标量值，包括每个批次的有限损失检查以及优化器更新时的梯度范数检查。

优化后的训练器变体现在保持了优化路径不变，但仅在 `log_interval` 边界处执行诊断性的 DirectML 到 CPU 读取操作。

概念上：

``` text
Between log points:
    FP16 forward / backward
    FP32 master update
    gradient clipping
    master → model synchronization
    no diagnostic .cpu().item()

At log_interval:
    read loss
    read latest gradient norm
    finite-value checks
    logging
```

该变更旨在减少同步开销，同时不移除 FP32 主权重。

随后对两个训练器在相同训练配置下进行了直接 `1000` 步骤的对比：

``` text
Steps:                    1000

Stable total:             2003.79 s
Log-Sync total:           1930.75 s

Stable time / step:       2.0038 s
Log-Sync time / step:     1.9307 s

Stable steps / second:    0.4991
Log-Sync steps / second:  0.5179

Speedup:                  1.038×
Time reduction:           3.65%
```

优化后的训练器在 `1000` 步骤内将总运行时间减少了大约 `73.04 s`，同时保留了 FP32 主权重的优化路径。

因此，该性能提升在测试配置下已得到验证。在日志同步训练器取代稳定训练器作为最终训练参考之前，检查点质量的验证仍待完成。

------------------------------------------------------------------------

<a id="m5-final-directml-cross-backend-validation--2026-09-24"></a>
# M5 最终 DirectML 跨后端验证 — 2026-09-24

最终对比超越了训练日志损失，评估了在相同数据和后端下的生成检查点。

在固定 `1,000` 样本子集（`204,327` 有效标记）上，两个检查点均在 ROCm 下进行评估：

|检查点|评估损失|困惑度|生成|
|---|---:|---:|---|
|DirectML 训练| 6.880233 | 972.8529 |不连贯|
|ROCm 训练| 1.846587 | 6.3382 |连贯|

随后进行的相同检查点后端测试使用了 DirectML 训练的检查点在一个固定批次（`873` 有效标记）上执行：

|执行路径|损失|结果|
|---|---:|---|
|DirectML FP16| 5.733902 |有限但发散|
|ROCm FP16| 6.230469 |参考 FP16 行为|
|DirectML FP32| 6.231627 |与 ROCm 一致|
|ROCm FP32| 6.231628 |与 DirectML 一致|
|DirectML FP16 未使用 SDPA|NaN|数值不稳定|
|ROCm FP16 未使用 SDPA| 6.230469 |稳定|

DirectML FP16模型的损失值也与CPU FP32交叉熵重新计算从其自身logits的结果相匹配（`5.733902` 对比 `5.736305`）。相应的ROCm值为`6.230469` 对比 `6.231570`。因此，保留的有效token交叉熵减少并非差异的最终来源：DirectML FP16前向路径生成的logits本身已存在显著差异。

在FP32下，DirectML和ROCm在大约`1e-6`处达成一致，包括损失和logit统计信息。这将最终阻塞行为定位在测试的DirectML FP16数值路径，而非检查点格式、分词器、数据集或通用DirectML推理。

**决策**：本次M5验证关闭了DirectML可行性研究。未来ROCm工作将另行记录。
