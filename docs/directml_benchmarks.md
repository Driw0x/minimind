# MiniMind — DirectML Benchmarks

This document records experimental DirectML compatibility and performance tests for MiniMind.

The objective is to identify practical training configurations for the target DirectML hardware.

These results describe the tested environment and should not be interpreted as universal DirectML limits.

---

# Batch Size and Sequence Length

## Motivation

`batch_size` and `max_seq_len` directly affect training memory requirements and execution time.

Increasing `batch_size` processes more sequences simultaneously.

Increasing `max_seq_len` increases the number of tokens processed for each sequence and significantly increases transformer attention cost.

Conceptually:

```text
batch_size ↑
     +
max_seq_len ↑
     ↓
Higher memory usage
Higher computation cost
Longer training steps
```

The upstream MiniMind defaults therefore cannot automatically be assumed to be practical on DirectML hardware.

---

# Methodology

A compatibility benchmark was introduced to test several combinations of:

```text
batch_size
max_seq_len
```

Each configuration performs a short real training run instead of a complete training session.

Results are classified as:

```text
PASS
FAIL
TIMEOUT
```

This allows unstable or impractically slow configurations to be identified before running the complete training pipeline.

---

# Initial Pretraining Benchmark

The initial `train_pretrain.py` benchmark produced:

| Batch size | Sequence length | Result              |
| ---------: | --------------: | ------------------- |
|          1 |              64 | PASS                |
|          1 |             128 | PASS                |
|          1 |             256 | PASS                |
|          1 |             340 | FAIL (`0xC0000409`) |
|          2 |             128 | PASS                |
|          2 |             256 | PASS                |
|          2 |             340 | PASS                |
|          4 |             128 | PASS                |
|          4 |             256 | TIMEOUT             |
|          8 |             128 | TIMEOUT             |

`2 × 256` was initially selected as a conservative configuration for validating the remaining training pipeline.

These results apply specifically to the tested pretraining configuration and should not automatically be generalized to every MiniMind trainer.

---

# Unexpected Result

The initial benchmark produced an apparently inconsistent observation:

```text
1 × 340 → FAIL
2 × 340 → PASS
```

At first, this appeared to indicate that DirectML stability was not monotonic with respect to batch size and sequence length.

However, the machine was later found to expose both:

```text
Integrated GPU
Dedicated GPU
```

and explicit physical GPU selection had not yet been fully controlled.

The observation therefore cannot safely be interpreted as an inherent DirectML batch-size or sequence-length behavior.

It remains recorded because it helped identify the multi-GPU device-selection issue.

---

# Multi-GPU Impact on Benchmarking

Benchmark results are only directly comparable when the execution environment remains consistent.

In particular:

```text
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

The physical DirectML adapter must therefore be explicitly selected before establishing reference performance limits.

Results obtained on the integrated GPU and dedicated GPU should not be combined into a single compatibility table.

---

# Reference Benchmark

After explicitly selecting the dedicated GPU, the compatibility benchmark was rerun on:

```text
Device: directml:1
```

The resulting compatibility matrix was:

| Batch size | Sequence length | Result |
| ---------: | --------------: | ------ |
|          1 |              64 | PASS   |
|          1 |             128 | PASS   |
|          1 |             256 | PASS   |
|          1 |             340 | PASS   |
|          2 |             128 | PASS   |
|          2 |             256 | PASS   |
|          2 |             340 | PASS   |
|          4 |             128 | PASS   |
|          4 |             256 | PASS   |
|          4 |             340 | PASS   |
|          8 |             128 | PASS   |
|          8 |             256 | PASS   |
|          8 |             340 | PASS   |
|         16 |             128 | PASS   |
|         16 |             256 | PASS   |
|         16 |             340 | PASS   |
|         32 |             128 | PASS   |
|         32 |             256 | FAIL   |

The benchmark stopped testing larger sequence lengths for batch size `32` after `32 × 256` failed.

This establishes a substantially higher validated compatibility range on the explicitly selected dedicated GPU than the initial benchmark suggested. In particular, all tested configurations up to `16 × 340` passed, while `32 × 128` passed and `32 × 256` did not.

The initial benchmark remains useful as historical development data because it documents the behavior observed before physical GPU selection was controlled.

---

# Benchmark vs Real Training

A configuration passing the compatibility benchmark is not guaranteed to remain stable during a complete training run.

The benchmark only executes a short training workload. Its purpose is to identify configurations that fail immediately or are clearly incompatible with the tested DirectML environment.

A complete training run may behave differently because it runs for significantly longer and may encounter different memory and execution conditions.

Therefore:

```text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

Benchmark results should be treated as compatibility indicators rather than guaranteed safe training configurations.

The final `batch_size` and `max_seq_len` should always be validated with an actual training run.

---

# Configuration Selection

The final DirectML training configuration should balance:

```text
Stability
    +
Training speed
    +
Memory usage
```

rather than simply selecting the largest configuration that starts successfully.

Benchmarking should be repeated when relevant execution conditions change, including:

* physical GPU;
* model size;
* trainer;
* DirectML/PyTorch environment;
* other changes that significantly affect memory or computation requirements.

---

# Real Training Observation

The compatibility benchmark only determines whether a configuration can execute successfully. It does not determine whether that configuration is practical for full-scale training.

A real pretraining run using:

- `batch_size = 32`
- `max_seq_len = 340`

executed successfully on DirectML.

However, the observed throughput was approximately:

- 5.9 seconds per step;
- 158780 steps per epoch;
- 10.9 days per epoch.

This confirms that a configuration can be technically compatible while still being impractical for full-scale training.

Compatibility benchmark results should therefore not be interpreted as performance recommendations.

---