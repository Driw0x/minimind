# MiniMind --- DirectML Benchmarks

This document records experimental DirectML compatibility and
performance tests for MiniMind.

The objective is to identify practical training configurations for the
target DirectML hardware.

These results describe the tested environment and should not be
interpreted as universal DirectML limits.

------------------------------------------------------------------------

# Batch Size and Sequence Length

## Motivation

`batch_size` and `max_seq_len` directly affect training memory
requirements and execution time.

Increasing `batch_size` processes more sequences simultaneously.

Increasing `max_seq_len` increases the number of tokens processed for
each sequence and significantly increases transformer attention cost.

Conceptually:

``` text
batch_size ↑
     +
max_seq_len ↑
     ↓
Higher memory usage
Higher computation cost
Longer training steps
```

The upstream MiniMind defaults therefore cannot automatically be assumed
to be practical on DirectML hardware.

------------------------------------------------------------------------

# Methodology

A compatibility benchmark was introduced to test several combinations
of:

``` text
batch_size
max_seq_len
```

Each configuration performs a short real training run instead of a
complete training session.

Results are classified as:

``` text
PASS
FAIL
TIMEOUT
```

This allows unstable or impractically slow configurations to be
identified before running the complete training pipeline.

------------------------------------------------------------------------

# Initial Pretraining Benchmark

The initial `train_pretrain.py` benchmark produced:

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

`2 × 256` was initially selected as a conservative configuration for
validating the remaining training pipeline.

These results apply specifically to the tested pretraining configuration
and should not automatically be generalized to every MiniMind trainer.

------------------------------------------------------------------------

# Unexpected Result

The initial benchmark produced an apparently inconsistent observation:

``` text
1 × 340 → FAIL
2 × 340 → PASS
```

At first, this appeared to indicate that DirectML stability was not
monotonic with respect to batch size and sequence length.

However, the machine was later found to expose both:

``` text
Integrated GPU
Dedicated GPU
```

and explicit physical GPU selection had not yet been fully controlled.

The observation therefore cannot safely be interpreted as an inherent
DirectML batch-size or sequence-length behavior.

It remains recorded because it helped identify the multi-GPU
device-selection issue.

------------------------------------------------------------------------

# Multi-GPU Impact on Benchmarking

Benchmark results are only directly comparable when the execution
environment remains consistent.

In particular:

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

The physical DirectML adapter must therefore be explicitly selected
before establishing reference performance limits.

Results obtained on the integrated GPU and dedicated GPU should not be
combined into a single compatibility table.

------------------------------------------------------------------------

# Reference Benchmark

After explicitly selecting the dedicated GPU, the compatibility
benchmark was rerun on:

``` text
Device: directml:1
```

The resulting compatibility matrix was:

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

The benchmark stopped testing larger sequence lengths for batch size
`32` after `32 × 256` failed.

This establishes a substantially higher validated compatibility range on
the explicitly selected dedicated GPU than the initial benchmark
suggested. In particular, all tested configurations up to `16 × 340`
passed, while `32 × 128` passed and `32 × 256` did not.

The initial benchmark remains useful as historical development data
because it documents the behavior observed before physical GPU selection
was controlled.

------------------------------------------------------------------------

# Benchmark vs Real Training

A configuration passing the compatibility benchmark is not guaranteed to
remain stable during a complete training run.

The benchmark only executes a short training workload. Its purpose is to
identify configurations that fail immediately or are clearly
incompatible with the tested DirectML environment.

A complete training run may behave differently because it runs for
significantly longer and may encounter different memory and execution
conditions.

Therefore:

``` text
Benchmark PASS
    ≠
Guaranteed full-training stability
```

Benchmark results should be treated as compatibility indicators rather
than guaranteed safe training configurations.

The final `batch_size` and `max_seq_len` should always be validated with
an actual training run.

------------------------------------------------------------------------

# Configuration Selection

The final DirectML training configuration should balance:

``` text
Stability
    +
Training speed
    +
Memory usage
```

rather than simply selecting the largest configuration that starts
successfully.

Benchmarking should be repeated when relevant execution conditions
change, including:

-   physical GPU;
-   model size;
-   trainer;
-   DirectML/PyTorch environment;
-   other changes that significantly affect memory or computation
    requirements.

------------------------------------------------------------------------

# Real Training Performance

Passing the compatibility benchmark does not imply that a configuration
is stable or efficient for real training.

Real pretraining benchmarks were therefore performed using the actual
MiniMind pretraining dataset and the full-size pretraining model.

The tested environment used:

-   Device: `directml:1`
-   PyTorch device representation: `privateuseone:1`
-   Dataset: `pretrain_t2t_mini.jsonl`
-   Dataset samples: `1,270,238`
-   Hidden size: `768`
-   Hidden layers: `8`
-   Model parameters: approximately `63.91M`
-   Max sequence length: `340`
-   Gradient accumulation steps: `8`

The benchmark performs real forward passes, backward passes, gradient
clipping, and AdamW optimizer steps.

------------------------------------------------------------------------

## Real Training Stability

Real-data training revealed stability limits that were not visible in
the short compatibility benchmark.

### Batch Size 32

Configuration:

    batch_size = 32
    max_seq_len = 340

The training workload started successfully but failed during the second
backward pass because of insufficient GPU memory.

Result:

    32 × 340 → OOM

This configuration is therefore not considered stable for real training
on the tested hardware.

### Batch Size 16

Configuration:

    batch_size = 16
    max_seq_len = 340
    accumulation_steps = 8

The benchmark completed the first gradient accumulation cycle and
optimizer step.

Training then failed during a subsequent backward pass because of
insufficient GPU memory.

Result:

    16 × 340 → OOM after first accumulation cycle

This demonstrates that successfully completing several training steps
does not guarantee sustained memory stability.

### Batch Size 8

Configuration:

    batch_size = 8
    max_seq_len = 340
    accumulation_steps = 8

A bounded 100-step real-training benchmark completed successfully
without GPU out-of-memory errors.

The first 8 steps were used as warmup, leaving 92 measured steps.

Result:

    8 × 340 → PASS (100-step bounded run)

This is currently the largest tested configuration that completed the
bounded real-training validation successfully.

------------------------------------------------------------------------

# Initial Real Training Performance Measurement

The initial 100-step real-training measurement used:

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

The initial measurement suggested that real MiniMind pretraining was
technically executable on the tested DirectML hardware but extremely
slow.

At the measured throughput:

    1 epoch  ≈ 9.70 days
    2 epochs ≈ 19.4 days

These values were initial estimates based on the first bounded
benchmark. They were later found to be strongly affected by
synchronization overhead in the measurement path and are retained here
as historical M4 results.

------------------------------------------------------------------------

# Corrected Real Training Benchmark

Further M4 investigation showed that the initial timing path introduced
significant synchronization overhead between DirectML and the CPU.

After removing unnecessary synchronization from the critical training
and measurement path, the same reference workload was rerun for 100
steps.

Reference configuration:

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

Corrected measured performance:

    Average iteration time:   0.482 s
    Samples / second:         16.61
    Effective tokens / sec:   3341.37
    Padded tokens / sec:      5647.46
    Steps / epoch:            158,780
    Estimated epoch duration: 21.24 h
    Estimated epoch duration: 0.89 days

The corrected benchmark completed all 100 steps successfully.

Compared with the initial measurement:

    Average iteration time: 5.280 s → 0.482 s
    Samples / second:       1.52    → 16.61
    Effective tokens / sec: 304.81  → 3341.37
    Estimated epoch:        9.70 d  → 0.89 d

The iteration-time improvement is approximately 10.95×.

The model architecture, dataset, physical batch size, sequence length,
and gradient accumulation configuration were not reduced to obtain this
improvement.

The result demonstrates that the initial performance estimate was
dominated by execution and measurement synchronization overhead rather
than by the real steady-state DirectML training cost alone.

The AdamW `aten::lerp.Scalar_out` CPU fallback remains present, but it
does not prevent the corrected benchmark from sustaining sub-second
iterations for the tested configuration.

------------------------------------------------------------------------

# Compatibility Benchmark vs Real Training

The compatibility benchmark and the real-training benchmark serve
different purposes.

The compatibility benchmark showed:

    8 × 340  → PASS
    16 × 340 → PASS
    32 × 128 → PASS
    32 × 256 → FAIL

However, real-data training showed:

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

The `32 × 340` configuration was not part of the validated compatibility
matrix because testing for batch size `32` stopped after `32 × 256`
failed.

The important difference is therefore the behavior of `16 × 340`.

It passed the short compatibility benchmark but failed during longer
real-data execution.

This confirms:

    Compatibility PASS
            ≠
    Sustained training stability

Short compatibility benchmarks remain useful for detecting immediate
backend and memory failures, but final training configurations must be
validated using real data and multiple gradient accumulation cycles.

------------------------------------------------------------------------

# Throughput and Batch Size

Real-training experiments show that the largest configuration accepted
by a short compatibility benchmark is not necessarily the best practical
configuration.

Real-data stability results remain:

    32 × 340 → OOM
    16 × 340 → OOM after first accumulation cycle
     8 × 340 → PASS for 100 bounded steps

For the current environment, `batch_size = 8` with `max_seq_len = 340`
and gradient accumulation `8` remains the preferred real-training
reference configuration.

After correcting synchronization overhead in the measurement path, this
configuration achieves approximately:

    0.482 s / iteration
    16.61 samples / second
    3341.37 effective tokens / second

This selection is based on sustained real-data stability and corrected
measured throughput rather than on the maximum configuration accepted by
the compatibility benchmark.

------------------------------------------------------------------------

# DirectML Optimizer Limitation

During real training, PyTorch reports a DirectML fallback for:

    aten::lerp.Scalar_out

This operation is currently unsupported by the DirectML backend and
falls back to CPU execution during the AdamW optimizer step.

The fallback does not prevent training from running, but it may
introduce synchronization and performance overhead.

The reference benchmark includes this fallback and therefore represents
the observed end-to-end behavior of the tested training operations
rather than pure DirectML GPU execution.

The corrected benchmark still includes the fallback while sustaining
approximately `0.482 s/iteration` on average.

The fallback therefore remains a known DirectML limitation, but the
large initial `5.280 s/iteration` result should not be attributed to
this fallback alone.

------------------------------------------------------------------------

# Current Performance Conclusion

The M4 real-training investigation establishes several important
observations.

First, compatibility benchmarks are not sufficient to determine
sustained training stability.

Second, real-data validation confirmed that `8 × 340` with gradient
accumulation `8` is stable for the bounded 100-step reference run, while
larger physical batches encountered GPU memory failures.

Third, the initial performance measurement of `5.280 s/iteration` was
not representative of the actual steady-state training cost. The
measurement path introduced significant DirectML-to-CPU synchronization
overhead.

After correcting this issue, the same reference configuration completed
a new 100-step benchmark with:

    16.61 samples/s
    3341.37 effective tokens/s
    0.482 s/iteration

and an estimated full-epoch duration of approximately:

    21.24 hours
    0.89 days

The corrected result is approximately 10.95× faster in iteration time
than the initial measurement.

Full-scale training is therefore substantially more practical than the
first M4 measurements suggested, although the 100-step benchmark is not
yet sufficient to prove sustained full-epoch stability.

Further M4 validation should focus on:

-   sustained training stability beyond the bounded 100-step run;
-   validation of the corrected throughput over longer runs;
-   continued observation of GPU memory behavior;
-   final assessment of DirectML practical viability;
-   consolidation of the remaining performance limitations.

------------------------------------------------------------------------

# Final M4 FP16 Validation

M4 concluded with a DirectML-specific FP16 training path designed to
avoid the numerical instability observed with the default AdamW epsilon.

The validated DirectML FP16 settings are:

``` text
Model dtype:              float16
Static loss scale:        1024
AdamW epsilon:            1e-4
Reference device:         directml:1
PyTorch representation:   privateuseone:1
```

Using the default AdamW epsilon of `1e-8` produced non-finite values
immediately after the first optimizer update. Static loss scaling alone
did not solve the problem. Using `1e-4` for DirectML FP16 AdamW
stabilized the optimizer path.

The DirectML FP16 compatibility matrix was then rerun with the full
`768 / 8-layer` model, gradient accumulation `8`, and a bounded 9-step
run so that each configuration crossed the first optimizer-update
boundary.

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

This matrix validates short bounded execution across the first
optimizer-update boundary. It does **not** mean that every large
configuration is suitable for sustained training. Earlier real-data
tests remain relevant: large physical batches can still exhaust VRAM
during longer runs.

------------------------------------------------------------------------

# Sustained Pretraining Validation

The final sustained M4 validation used:

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

The run completed all `1000` training steps successfully.

Because `1000` is divisible by the accumulation factor `8`, the run
performed `125` optimizer updates.

No NaN or Inf loss was observed during the sustained run. Logged losses
remained finite through step `1000`.

The final training ETA was approximately:

``` text
1360 minutes
22.67 hours / epoch
```

This is consistent with the corrected 100-step benchmark estimate of
approximately `21.24 hours / epoch`.

The sustained run therefore confirms that the corrected throughput is
representative beyond the short 100-step benchmark and that the selected
DirectML FP16 reference configuration remains numerically stable across
many optimizer updates.

------------------------------------------------------------------------

# Final FP16 GPU Memory Measurement

External GPU memory monitoring during the final FP16 reference run
recorded:

``` text
Peak dedicated VRAM:     7,769.62 MB
Median active VRAM:      7,751.02 MB
Average active VRAM:     7,218.87 MB
Typical active plateau:  ~7,751 MB
```

Compared with the earlier FP32 peak of `11,623.12 MB`, FP16 reduced peak
dedicated GPU memory by approximately:

``` text
3,853.50 MB
33.15%
```

The final FP16 reference workload therefore uses approximately
`7.59 GiB` peak dedicated VRAM instead of approximately `11.35 GiB` in
the earlier FP32 measurement.

------------------------------------------------------------------------

# Cross-Trainer Smoke Validation

The DirectML FP16 path and bounded `--max_steps` execution were
propagated across the main trainable workflows used by the M4 smoke
runner:

``` text
Pretrain
Full SFT
LoRA
Distillation
GRPO
Agent RL
PPO
```

The consolidated `tests/test_all_trainers.py` runner completed all
included trainer smoke tests successfully.

DPO remains part of the previously validated MiniMind training pipeline,
but it is not part of this M4 FP16 smoke-runner result.

------------------------------------------------------------------------

# Final M4 Conclusion

M4 establishes DirectML training as practically viable for the tested
MiniMind configuration on the reference hardware.

The recommended real-training baseline is:

``` text
Device:                 directml:1
Model dtype:            float16
Batch size:             8
Max sequence length:    340
Gradient accumulation:  8
Static loss scale:      1024
AdamW epsilon:          1e-4
```

The final evidence combines:

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

DirectML still has backend limitations and CPU fallbacks, but they do
not prevent practical training with the validated configuration.


------------------------------------------------------------------------

# MoE DirectML Validation

The final M4 smoke workflow was extended to cover the MiniMind MoE
configuration in addition to the Dense model.

The tested `768 / 8-layer` MoE model reports approximately:

``` text
Model Params: 198.42M-A63.94M
```

DirectML exposed a compatibility issue in the upstream sparse expert
routing path. The affected routing operations required scatter behavior
that was not supported reliably by the DirectML forward/backward path.

A DirectML-specific scatter-free routing path was therefore introduced.
CPU and CUDA keep the original sparse routing behavior.

The DirectML compatibility path computes all experts and combines their
outputs using the selected routing weights. It is intended as a
functional compatibility fallback and should not be interpreted as
native sparse-MoE performance.

The following workflows completed successfully:

``` text
MoE Pretrain
MoE Full SFT
Dense student ← MoE teacher Distillation
```

A separate bounded 20-step MoE pretraining run also remained finite.

A short Dense/MoE comparison under identical small validation settings
produced a preliminary runtime ratio of approximately:

``` text
MoE / Dense ≈ 1.158×
```

or about `15.8%` additional runtime in that experiment.

This value is only a short relative observation and is not considered a
full MoE performance benchmark.

------------------------------------------------------------------------

# Final Cross-Trainer Smoke Result

After adding Dense and MoE coverage and resolving the remaining Agent
compatibility issues, the final DirectML trainer smoke runner completed:

``` text
All trainer smoke tests passed
Passed: 9/9
```

The final runner validates:

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

DPO remains part of the previously validated training pipeline but is
not included in this specific 9-test runner.
