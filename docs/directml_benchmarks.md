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

# Real Training Performance

Passing the compatibility benchmark does not imply that a configuration is stable or efficient for real training.

Real pretraining benchmarks were therefore performed using the actual MiniMind pretraining dataset and the full-size pretraining model.

The tested environment used:

* Device: `directml:1`
* PyTorch device representation: `privateuseone:1`
* Dataset: `pretrain_t2t_mini.jsonl`
* Dataset samples: `1,270,238`
* Hidden size: `768`
* Hidden layers: `8`
* Model parameters: approximately `63.91M`
* Max sequence length: `340`
* Gradient accumulation steps: `8`

The benchmark performs real forward passes, backward passes, gradient clipping, and AdamW optimizer steps.

---

## Real Training Stability

Real-data training revealed stability limits that were not visible in the short compatibility benchmark.

### Batch Size 32

Configuration:

    batch_size = 32
    max_seq_len = 340

The training workload started successfully but failed during the second backward pass because of insufficient GPU memory.

Result:

    32 × 340 → OOM

This configuration is therefore not considered stable for real training on the tested hardware.

### Batch Size 16

Configuration:

    batch_size = 16
    max_seq_len = 340
    accumulation_steps = 8

The benchmark completed the first gradient accumulation cycle and optimizer step.

Training then failed during a subsequent backward pass because of insufficient GPU memory.

Result:

    16 × 340 → OOM after first accumulation cycle

This demonstrates that successfully completing several training steps does not guarantee sustained memory stability.

### Batch Size 8

Configuration:

    batch_size = 8
    max_seq_len = 340
    accumulation_steps = 8

A bounded 100-step real-training benchmark completed successfully without GPU out-of-memory errors.

The first 8 steps were used as warmup, leaving 92 measured steps.

Result:

    8 × 340 → PASS (100-step bounded run)

This is currently the largest tested configuration that completed the bounded real-training validation successfully.

---

# Reference Real Training Benchmark

The current reference real-training configuration is:

    Device:                   directml:1
    PyTorch device:           privateuseone:1
    Dataset samples:          1,270,238
    Batch size:               8
    Max sequence length:      340
    Gradient accumulation:    8
    Warmup steps:             8
    Total benchmark steps:    100
    Measured steps:           92

Measured performance:

    Average iteration time:   5.280 s
    Samples / second:         1.52
    Effective tokens / sec:   304.81
    Padded tokens / sec:      515.17
    Steps / epoch:            158,780
    Estimated epoch duration: 232.87 h
    Estimated epoch duration: 9.70 days
    Peak dedicated GPU memory: 11,623.12 MB

Dedicated GPU memory was measured externally using Windows GPU process
memory counters during the reference benchmark.

Peak dedicated GPU memory reached approximately 11.62 GB.

The training workload remained close to 11.6 GB of dedicated GPU memory
for a significant portion of the run, indicating that the reference
configuration operates relatively close to the available VRAM limit.

The benchmark therefore indicates that real MiniMind pretraining is technically executable on the tested DirectML hardware, but full-dataset training remains very slow.

At the measured throughput:

    1 epoch  ≈ 9.70 days
    2 epochs ≈ 19.4 days

These values are estimates based on the bounded benchmark and should not be interpreted as guaranteed full-training runtimes.

---

# Compatibility Benchmark vs Real Training

The compatibility benchmark and the real-training benchmark serve different purposes.

The compatibility benchmark showed:

    8 × 340  → PASS
    16 × 340 → PASS
    32 × 128 → PASS
    32 × 256 → FAIL

However, real-data training showed:

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

The `32 × 340` configuration was not part of the validated compatibility matrix because testing for batch size `32` stopped after `32 × 256` failed.

The important difference is therefore the behavior of `16 × 340`.

It passed the short compatibility benchmark but failed during longer real-data execution.

This confirms:

    Compatibility PASS
            ≠
    Sustained training stability

Short compatibility benchmarks remain useful for detecting immediate backend and memory failures, but final training configurations must be validated using real data and multiple gradient accumulation cycles.

---

# Throughput and Batch Size

The real-training experiments also show that increasing batch size does not necessarily improve throughput proportionally on the tested DirectML environment.

With `batch_size = 8`, measured iteration time is approximately:

    5.28 seconds

With `batch_size = 16`, iterations observed before the OOM were approximately:

    11 seconds

Doubling the batch size therefore approximately doubled iteration time while also increasing memory pressure enough to make sustained execution unstable.

The larger configuration consequently provides no clear practical advantage on the tested hardware.

For the current environment, `batch_size = 8` with `max_seq_len = 340` is the preferred real-training validation configuration.

This selection is based on observed real-data stability and throughput rather than on the maximum configuration accepted by the compatibility benchmark.

---

# DirectML Optimizer Limitation

During real training, PyTorch reports a DirectML fallback for:

    aten::lerp.Scalar_out

This operation is currently unsupported by the DirectML backend and falls back to CPU execution during the AdamW optimizer step.

The fallback does not prevent training from running, but it may introduce synchronization and performance overhead.

The reference benchmark includes this fallback and therefore represents the observed end-to-end behavior of the tested training operations rather than pure DirectML GPU execution.

The measured iteration pattern also shows a small periodic increase in execution time around optimizer steps.

Further investigation is required to determine how much of this overhead is specifically caused by the CPU fallback.

---

# Current Performance Conclusion

The M4 real-training benchmarks establish three important observations.

First, compatibility benchmarks are not sufficient to determine sustained training stability.

Second, real MiniMind pretraining with the full-size tested model is computationally expensive on the selected DirectML hardware.

Third, reducing batch size improves memory stability without significantly reducing overall sample throughput.

The current reference configuration is therefore:

    batch_size = 8
    max_seq_len = 340
    accumulation_steps = 8

This configuration completed the 100-step bounded real-training benchmark successfully and achieved approximately:

    1.52 samples/s
    304.81 effective tokens/s
    5.280 s/iteration

with an estimated full-epoch duration of approximately:

    9.70 days

Full-scale training is therefore technically possible in principle but is not currently considered practical based on the measured runtime.

Further M4 investigation should focus on:

* GPU memory usage;
* CPU fallback overhead;
* DirectML synchronization and execution overhead;
* identification of operations responsible for the observed training cost;
* possible mitigations for the identified bottlenecks.

---