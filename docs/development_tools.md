# Development Tools

## Purpose

MiniMind now uses two diagnostic programs for model and dataset validation:

- `scripts/diagnose_validation.py` compares model checkpoints and consolidates the former training diagnostic and validation workflows.
- `scripts/diagnose_sft.py` consolidates SFT dataset quality analysis and block-level source-distribution analysis.

Both merged tools have been verified end-to-end on the current ROCm environment. The former `diagnose_training.py`, `diagnose_sft_data.py`, and `diagnose_sft_blocks.py` are superseded and can be removed.

---

## 1. `diagnose_validation.py`

### Goal

`diagnose_validation.py` compares one or more MiniMind weights using three complementary views:

1. training metadata and dataset statistics;
2. local loss comparison on reproducible random samples from the pretraining and SFT datasets;
3. deterministic generation on an external prompt benchmark.

The first weight passed through `--weights` is used as the comparison baseline.

### Modes

If `--mode` is omitted, the script asks interactively:

```text
[0] Local data validation + training audit
[1] External prompt validation
[2] Full validation
```

The same modes can be selected directly:

```text
--mode local
--mode external
--mode full
```

### Training audit

The local/full modes report:

- dataset and weight file sizes;
- inferred resume checkpoint metadata;
- saved epoch, step, world size, and optimizer learning rate when available;
- tokenizer vocabulary and special-token IDs;
- random-sample dataset statistics;
- average input and supervised-token counts.

The stored optimizer learning rate is the value saved in the resume checkpoint. It is not necessarily the initial training learning rate.

### Local validation

The script selects the same random indices for every compared checkpoint. This makes loss comparisons reproducible and avoids the bias of evaluating only the first dataset records.

For each weight it reports:

```text
weight on pretrain | loss | perplexity | supervised tokens
weight on SFT      | loss | perplexity | supervised tokens
```

It then prints loss deltas relative to the first weight.

A higher pretraining loss after SFT is interpreted as possible forgetting. A lower SFT loss means the checkpoint fits the SFT distribution better.

### External validation

The external benchmark uses a fixed set of Chinese prompts covering:

- factual explanation;
- Python code;
- arithmetic;
- photosynthesis;
- instruction following;
- RAM vs SSD knowledge;
- practical advice;
- code correction.

Generation is deterministic (`do_sample=False`). For each response, the tool reports generated-token count and a character 3-gram repetition ratio.

The repetition ratio is only a degeneration indicator. Factual correctness, reasoning, code correctness, and instruction following still require manual review.

### Current reference results

Model configuration:

```text
Parameters: 63.91M
hidden_size: 768
layers: 8
MoE: false
```

Dataset scale:

```text
Pretrain samples: 1,270,238
SFT samples     :   895,718
```

The verified merged-tool run (`seed=42`, 256 sampled records) measured:

```text
Pretrain average input/supervised tokens: 205.57 / 205.57
SFT average input/supervised tokens     : 478.06 / 402.80
```

These values are sample-dependent diagnostics, not corpus-wide token-count estimates.

Random local validation (`seed=42`, 256 samples per dataset) produced:

| Weight | Pretrain loss | SFT loss |
| --- | ---: | ---: |
| `pretrain` | 1.8275 | 2.5207 |
| `full_sft` | 1.9992 | 1.5663 |

Relative to `pretrain`:

```text
Pretrain loss delta: +0.1717
SFT loss delta     : -0.9543
```

Interpretation: Full SFT substantially improves fit to the SFT distribution while causing moderate forgetting on the pretraining distribution.

The external benchmark previously produced:

```text
pretrain average generated tokens: 246.8
pretrain average repetition      : 0.795

full_sft average generated tokens: 212.5
full_sft average repetition      : 0.361
```

Full SFT therefore improved generation stability and reduced severe repetition, while factual accuracy and reasoning remained limited on several prompts.

### Baseline status

The current `full_sft_768.pth` checkpoint is the validated experimental baseline for the next runtime-memory milestone. The diagnostics show that:

- the checkpoint loads and evaluates correctly on ROCm;
- SFT improves fit to the SFT distribution;
- moderate pretraining-distribution forgetting remains measurable;
- external generation is substantially less repetitive than the pretraining checkpoint;
- remaining factual, reasoning, and instruction-following limitations are model/data limitations rather than diagnostic-tool failures.

### Recommended commands

Interactive full comparison:

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0
```

Non-interactive local audit:

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4
```

Non-interactive external benchmark:

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode external --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --max_new_tokens 256
```

Complete audit + external benchmark:

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode full --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 256 --stat_samples 256 --batch_size 4 --max_new_tokens 256
```

---

## 2. `diagnose_sft.py`

### Goal

`diagnose_sft.py` audits the SFT JSONL itself. It combines detailed sample-quality analysis with block-level distribution analysis in one dataset scan.

It checks:

- JSON validity;
- missing or empty conversations;
- role distribution;
- tool-calling share;
- first-sample vs random-sample distributions;
- user, assistant, and total token lengths;
- truncation against `max_seq_len`;
- role-order anomalies;
- exact prompt/response duplication;
- character n-gram repetition;
- suspicious examples;
- source-distribution shifts across file blocks.

### Modes

```text
--mode dataset   # detailed FIRST vs RANDOM analysis
--mode blocks    # block-level analysis only
--mode all       # both analyses (default)
```

### Current dataset findings

The SFT dataset contains:

```text
Total lines             : 895,718
Valid lines             : 895,718
Invalid JSON            : 0
Missing conversations   : 0
Empty conversations     : 0
Conversations with tools: 74,832
Tool share              : 8.35%
```

The beginning of the file differs strongly from a random sample.

First 5,000 samples:

```text
messages / conversation mean: 7.40
conversation tokens mean     : 663.02
truncated at 768              : 25.34%
assistant repetition mean    : 0.13
```

Random 5,000 samples:

```text
messages / conversation mean: 2.83
conversation tokens mean     : 491.90
truncated at 768              : 5.60%
assistant repetition mean    : 0.17
```

This shows that the JSONL is heterogeneous and physically ordered by source/type.

### Block-level findings

Using 100,000-line blocks and 2,000 tokenized samples per block:

| Block | Lines | Single-turn | Tools | Mean tokens | Truncated | Repetition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1–100,000 | 28.88% | 0.00% | 612.9 | 21.65% | 0.153 |
| 2 | 100,001–200,000 | 100.00% | 0.00% | 421.5 | 1.05% | 0.256 |
| 3 | 200,001–300,000 | 100.00% | 0.00% | 492.2 | 1.70% | 0.204 |
| 4 | 300,001–400,000 | 100.00% | 0.00% | 490.1 | 1.65% | 0.209 |
| 5 | 400,001–500,000 | 100.00% | 0.00% | 488.5 | 1.90% | 0.210 |
| 6 | 500,001–600,000 | 48.30% | 0.00% | 558.6 | 10.30% | 0.203 |
| 7 | 600,001–700,000 | 100.00% | 0.00% | 527.2 | 5.05% | 0.099 |
| 8 | 700,001–800,000 | 73.58% | 28.93% | 486.7 | 2.25% | 0.082 |
| 9 | 800,001–895,718 | 56.04% | 47.96% | 361.5 | 3.30% | 0.118 |

The file therefore contains clear source boundaries: long multi-turn data, several single-turn regions, and tool-heavy regions near the end.

The estimated global truncation rate is roughly 5–6%, but blocks 1 and 6 are much more affected.

Duplicate prompts/responses are particularly frequent in the tool-heavy blocks. These duplicates should not be removed blindly because repeated tool templates can be legitimate.

### Training-order note

The trainer does not consume the JSONL sequentially in normal training. It uses shuffled indices (`torch.randperm`) in non-distributed training and `DistributedSampler` with `set_epoch` in distributed training.

Therefore, block ordering is useful for diagnosing source heterogeneity but is not evidence that the model trained on block 1 first and tool data last.

### Recommended commands

Full SFT audit:

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 5000 --first_samples 5000 --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

Detailed sample analysis only:

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode dataset --samples 5000 --first_samples 5000 --max_seq_len 768
```

Block analysis only:

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode blocks --block_size 100000 --sample_per_block 2000 --max_seq_len 768
```

---

## Verification workflow

The merged tools have already completed the full validation workflow successfully on the current environment. For future changes, rerun the following checks.

First verify syntax:

```powershell
python -m py_compile scripts\diagnose_validation.py scripts\diagnose_sft.py
```

Verify both CLIs:

```powershell
python -m scripts.diagnose_validation --help
python -m scripts.diagnose_sft --help
```

Then run a small smoke test before the full diagnostics:

```powershell
python -m scripts.diagnose_validation --weights pretrain full_sft --mode local --device cuda:0 --hidden_size 768 --num_hidden_layers 8 --use_moe 0 --eval_samples 16 --stat_samples 16 --batch_size 1
```

```powershell
python -m scripts.diagnose_sft --data dataset/sft_t2t_mini.jsonl --tokenizer model --mode all --samples 100 --first_samples 100 --block_size 100000 --sample_per_block 50 --max_seq_len 768
```

If these complete successfully, run the full commands documented above and compare the results with the reference values in this document.
