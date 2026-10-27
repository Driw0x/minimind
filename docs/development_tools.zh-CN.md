[English](development_tools.md) | [Français](development_tools.fr.md) | [简体中文](development_tools.zh-CN.md)

[项目 README](../README.zh-CN.md)

<a id="development-tools"></a>
# 开发工具

<a id="purpose"></a>
## 目的

MiniMind目前使用两个诊断程序进行模型和数据集的有效性验证：

- `scripts/diagnose_validation.py` 比较模型检查点，并整合了之前的训练诊断和验证工作流程。
- `scripts/diagnose_sft.py` 整合了SFT数据集质量分析和按块的源分布分析。

这两个合并后的工具已在当前ROCm环境中完成端到端验证。之前的 `diagnose_training.py`、`diagnose_sft_data.py` 和 `diagnose_sft_blocks.py` 已被取代，可以移除。

---

<a id="1-diagnose_validationpy"></a>
## 1. `diagnose_validation.py`

<a id="goal"></a>
### 目标

`diagnose_validation.py` 通过三种互补视角比较一个或多个MiniMind权重：

1. 训练元数据和数据集统计信息；
2. 在预训练和SFT数据集上可复现的随机样本上的局部损失比较；
3. 在外部提示基准上的确定性生成。

首个通过`--weights`的权重被用作比较基准。

<a id="modes"></a>
### 模式

如果省略`--mode`，脚本将交互式询问：

```text
[0] Local data validation + training audit
[1] External prompt validation
[2] Full validation
```

相同的模式也可以直接选择：

```text
--mode local
--mode external
--mode full
```

<a id="training-audit"></a>
### 训练审计

本地/完整模式报告：

- 数据集和权重文件大小；
- 推断出的恢复检查点元数据；
- 保存的周期、步数、世界大小和优化器学习率（如有）；
- 分词器词汇表和特殊标记ID；
- 随机样本数据集统计信息；
- 平均输入和监督标记数量。

存储的优化器学习率是恢复检查点中保存的值，不一定等于初始训练学习率。

<a id="local-validation"></a>
### 本地验证

脚本为每个比较的检查点选择相同的随机索引。这使得损失比较可复现，并避免了仅评估数据集前几条记录的偏差。

对于每个权重，它会报告：

```text
weight on pretrain | loss | perplexity | supervised tokens
weight on SFT      | loss | perplexity | supervised tokens
```

然后打印相对于第一个权重的损失差异。

SFT之后预训练损失升高被解释为可能的遗忘。SFT损失降低意味着该检查点更符合SFT分布。

<a id="external-validation"></a>
### 外部验证

外部基准使用一组固定的中文提示，涵盖：

- 事实解释；
- Python代码；
- 算术运算；
- 光合作用；
- 指令遵循；
- RAM与SSD知识；
- 实用建议；
- 代码修正。

生成是确定性的（`do_sample=False`）。对于每个响应，工具报告生成标记数量和一个字符3-gram重复率。

重复率仅作为退化指标。事实正确性、推理过程、代码正确性和指令遵循仍需人工审查。

<a id="reference-results"></a>
### 参考结果

模型配置：

```text
Parameters: 63.91M
hidden_size: 768
layers: 8
MoE: false
```

数据集规模：

```text
Pretrain samples: 1,270,238
SFT samples     :   895,718
```

经验证的合并工具运行（`seed=42`，256个采样记录）测得：

```text
Pretrain average input/supervised tokens: 205.57 / 205.57
SFT average input/supervised tokens     : 478.06 / 402.80
```

这些值是基于样本的诊断指标，而非整个语料库的标记数量估算。

历史迷你检查点验证（`seed=42`，每个数据集256个样本）产生：

|权重|预训练损失|SFT损失|
| --- | ---: | ---: |
| `pretrain` | 1.8275 | 2.5207 |
| `full_sft` | 1.9992 | 1.5663 |

相对于`pretrain`：

```text
Pretrain loss delta: +0.1717
SFT loss delta     : -0.9543
```

解释：完整的SFT显著提升了对SFT分布的拟合度，但在预训练分布上造成了中等程度的遗忘。

迷你检查点外部基准测试产生：

```text
pretrain average generated tokens: 246.8
pretrain average repetition      : 0.795

full_sft average generated tokens: 212.5
full_sft average repetition      : 0.361
```

因此，完整的SFT提升了生成的稳定性，减少了严重重复现象，但在多个提示上事实准确性和推理能力仍有限。

<a id="current-dense-full-data-checkpoints"></a>
### 当前密集型全数据检查点

当前保留的密集型检查点是在完整的T2T数据集上训练的：

```text
pretrain_768.pth  -> pretrain_t2t.jsonl
full_sft_768.pth -> sft_t2t.jsonl
```

`diagnose_validation.py`默认使用迷你数据集作为固定参考集，除非显式传递了不同的路径。

当前验证（`seed=42`，每个参考数据集256个样本）：

|权重|预训练参考损失|SFT参考损失|
| --- | ---: | ---: |
| `pretrain` | 1.6372 | 2.8685 |
| `full_sft` | 1.8730 | 1.3771 |

相对于`pretrain`：

```text
Pretrain loss delta: +0.2358
SFT loss delta     : -1.4913
```

当前完整的SFT确定性外部基准测试产生：

```text
average generated tokens: 213.5
average repetition      : 0.244
```

全预训练的外部重复平均值未被保留作为可靠指标，因为多个生成达到了令牌限制但解码为一个空可见响应。


<a id="current-moe-full-data-checkpoints"></a>
### 当前 MoE 全数据检查点

MoE Pretrain 和 Full SFT 训练及其在 AMD ROCm 上的初步评估均已完成：`out/pretrain_768_moe.pth` 使用 `pretrain_t2t.jsonl` 训练，随后 `out/full_sft_768_moe.pth` 使用 `sft_t2t.jsonl` 训练，从 MoE Pretrain 初始化。配置为隐藏维度768、8层、每层4个专家、top-1 路由，总参数量198.42M，报告的活跃参数量63.94M。

`seed=42`，每个 Mini 参考数据集256个样本：

|权重|预训练参考损失|SFT 参考损失|预训练 / SFT 困惑度|外部重复率|
| --- | ---: | ---: | --- | --- |
| `pretrain_768_moe.pth` | 1.5468 | 2.1544 | 4.70 / 8.62 |0.842（方法限制）|
| `full_sft_768_moe.pth` | 1.7544 | 1.2994 | 5.78 / 3.67 | 0.262 |

Mini 参考样本来自训练数据。这些损失衡量分布拟合与遗忘，而非独立测试集上的泛化。八个外部提示使用确定性解码和聊天模板，Pretrain 也不例外。相比之下，`eval_llm.py` 对 Pretrain 使用 BOS + 原始提示，并进行采样。因此，`0.842` 需结合这一方法限制解读。

MoE Full SFT 的参考损失低于 Dense Full SFT，但重复率略高（`0.262` 对比 `0.244`）。算术错误（`12 - 5 + 3` 回答为 `16`）、Fibonacci 代码错误、格式要求遵循不完善、重复和截断仍然存在。这些结果不能证明 MoE 全面优于 Dense；详见 [`generation_quality.md`](generation_quality.zh-CN.md)。

<a id="baseline-status"></a>
### 基线状态

当前`full_sft_768.pth`检查点是基于完整SFT数据集训练的保留密集型完整SFT基线。诊断结果显示：

- 该检查点能在ROCm上正确加载和评估；
- SFT提升了对SFT分布的拟合度；
- 中等程度的预训练分布遗忘仍然可测量；
- 外部生成的重复性显著低于预训练检查点；
- 剩余的事实准确性、推理和指令遵循限制属于模型/数据限制，而非诊断工具故障。

<a id="recommended-commands"></a>
### 推荐命令

交互式完整对比：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0
```

非交互式本地审计：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4
```

非交互式外部基准测试：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode external --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --max_new_tokens 256
```

完整审计+外部基准测试：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode full --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4 --max_new_tokens 256
```

MoE 审计 + 外部基准，从仓库根目录执行：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode full --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 1 --eval_samples 256 --stat_samples 256 --batch_size 4 --max_new_tokens 256 --seed 42
```

`--use_moe 1` 选择 `out/` 下的 `_moe.pth` 检查点；默认参考路径仍为 `dataset/pretrain_t2t_mini.jsonl` 和 `dataset/sft_t2t_mini.jsonl`。Dense 和 MoE 必须分别运行，因为 `--use_moe` 对一次运行中的所有权重生效。

---

<a id="2-diagnose_sftpy"></a>
## 2. `diagnose_sft.py`

<a id="goal-1"></a>
### 目标

`diagnose_sft.py`对SFT JSONL本身进行审计。它在一个数据集扫描中结合了详细的样本质量分析与块级分布分析。

它检查：

- JSON有效性；
- 缺失或空的对话；
- 角色分布；
- 工具调用比例；
- 首样本与随机样本分布；
- 用户、助手及总令牌长度；
- 与`max_seq_len`的截断对比；
- 角色顺序异常；
- 精确提示/响应重复；
- 字符n-gram重复；
- 可疑示例；
- 文件块间源分布偏移。

<a id="modes-1"></a>
### 模式

```text
--mode dataset   # detailed FIRST vs RANDOM analysis
--mode blocks    # block-level analysis only
--mode all       # both analyses (default)
```

<a id="current-dataset-findings"></a>
### 当前数据集发现

SFT数据集包含：

```text
Total lines             : 895,718
Valid lines             : 895,718
Invalid JSON            : 0
Missing conversations   : 0
Empty conversations     : 0
Conversations with tools: 74,832
Tool share              : 8.35%
```

文件开头与随机样本差异显著。

前5000个样本：

```text
messages / conversation mean: 7.40
conversation tokens mean     : 663.02
truncated at 768              : 25.34%
assistant repetition mean    : 0.13
```

随机5000个样本：

```text
messages / conversation mean: 2.83
conversation tokens mean     : 491.90
truncated at 768              : 5.60%
assistant repetition mean    : 0.17
```

这表明JSONL是异构的，并按源类型物理排序。

<a id="block-level-findings"></a>
### 块级发现

使用10万行块和每个块2000个分词样本：

|块|行数|单轮|工具|平均分词数|截断|重复|
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 |1–100,000| 28.88% | 0.00% | 612.9 | 21.65% | 0.153 |
| 2 |100,001–200,000| 100.00% | 0.00% | 421.5 | 1.05% | 0.256 |
| 3 |200,001–300,000| 100.00% | 0.00% | 492.2 | 1.70% | 0.204 |
| 4 |300,001–400,000| 100.00% | 0.00% | 490.1 | 1.65% | 0.209 |
| 5 |400,001–500,000| 100.00% | 0.00% | 488.5 | 1.90% | 0.210 |
| 6 |500,001–600,000| 48.30% | 0.00% | 558.6 | 10.30% | 0.203 |
| 7 |600,001–700,000| 100.00% | 0.00% | 527.2 | 5.05% | 0.099 |
| 8 |700,001–800,000| 73.58% | 28.93% | 486.7 | 2.25% | 0.082 |
| 9 |800,001–895,718| 56.04% | 47.96% | 361.5 | 3.30% | 0.118 |

因此该文件包含清晰的源边界：长多轮数据、若干单轮区域，以及末尾的工具密集区域。

估算的整体截断率约为5%–6%，但第1和第6个块受影响更严重。

在工具密集的块中，重复提示/响应尤为频繁。这些重复不应盲目移除，因为重复的工具模板可能是合法的。

<a id="full-sft-training-dataset"></a>
### 完整SFT训练数据集

完整的`sft_t2t.jsonl`包含5,099,432个样本。块扫描显示其长上下文分布远重于Mini SFT数据集。

来自10万行块分析的加权估算：

```text
Single-turn share        : ~81.6%
Tool-conversation share  : ~7.7%
Mean conversation length : ~826.8 tokens
Truncated at 768         : ~34.3%
```

若干后期块在`max_seq_len=768`上超过50%的截断，部分块甚至接近100%。这些数值是块样本估算，而非整个语料库的精确分词统计。


<a id="training-order-note"></a>
### 训练顺序说明

在正常训练中，训练器不会按JSONL顺序逐行消费数据。在非分布式训练中，它使用随机索引（`torch.randperm`）；在分布式训练中，它使用`DistributedSampler`与`set_epoch`。

因此，块顺序对诊断源异构性是有用的，但不能证明模型是先训练了第1个块，最后才训练了工具数据。

<a id="recommended-commands-1"></a>
### 推荐命令

完整SFT审计：

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 5000 --first_samples 5000 --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

完整训练数据集审计：

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t.jsonl --tokenizer model --mode all --samples 5000 --first_samples 5000 --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```


仅详细样本分析：

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode dataset --samples 5000 --first_samples 5000 --max_seq_len 768
```

仅块分析：

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode blocks --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

---

<a id="verification-workflow"></a>
## 验证流程

已合并的工具已在当前环境中成功完成完整验证流程。对于未来变更，需重新运行以下检查。

首先验证语法：

```powershell
python -m py_compile scripts\diagnose_validation.py scripts\diagnose_sft.py
```

验证两个CLI：

```powershell
python -m scripts.diagnose_validation --help
python -m scripts.diagnose_sft --help
```

然后在完整诊断前运行一个小的烟雾测试：

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 16 --stat_samples 16 --batch_size 1
```

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 100 --first_samples 100 --block_size 100000 --sample_per_block 50 --max_seq_len 768
```

如果这些测试均成功，运行上述文档中记录的完整命令，并将结果与本文件中的参考值进行对比。
