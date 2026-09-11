# MiniMind --- DirectML Limitations

This document tracks the current unsupported, partially supported, and
fallback operations identified while running MiniMind on DirectML.

Detailed explanations of encountered problems, their causes, and
implemented solutions are documented in
[`directml_issues.md`](directml_issues.md).

Performance measurements are documented in
[`directml_benchmarks.md`](directml_benchmarks.md).

------------------------------------------------------------------------

# Current Limitations

  ---------------------------------------------------------------------------
  Feature / Operation             Status        Workaround
  ------------------------------- ------------- -----------------------------
  AdamW `aten::lerp.Scalar_out`   Partial       Automatic CPU fallback

  InternLM2 Reward Model causal   Unsupported   Reward Model runs on CPU
  mask                                          

  `torch.compile`                 Unsupported   Use `--use_compile 0`

  CUDA AMP / autocast             Not used on   DirectML FP16 uses static
                                  DirectML      loss scaling

  Sparse MoE routing              Partial       Scatter-free DirectML
                                                compatibility path

  Agent full-sequence attention   Partial       Avoid problematic mask for
  mask in FP16                                  right-padded recomputation

  Cross-entropy `ignore_index`     Partial       Explicit valid-token
  mean reduction                                 normalization
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

# AdamW CPU Fallback

The AdamW optimizer uses:

``` text
aten::lerp.Scalar_out
```

which is not currently supported natively by the DirectML backend.

`torch-directml` automatically falls back to CPU execution for this
operation.

Training remains functional, but the fallback may affect performance.

The fallback remains present during M4 real-training benchmarks. It does
not prevent training, but its isolated performance cost has not been
fully quantified.

------------------------------------------------------------------------

# Reward Model

The InternLM2 Reward Model used by alignment workflows cannot currently
execute entirely on DirectML because of incompatible causal-mask
operations.

The current execution strategy is:

``` text
MiniMind policy          → DirectML
MiniMind reference model → DirectML
Reward Model             → CPU
```

Required inputs are moved to CPU before reward-model inference, and
reward values are transferred back to the training device when
necessary.

This fallback is handled by the shared training utilities.

------------------------------------------------------------------------

# torch.compile

`torch.compile` is currently unsupported for the DirectML training path.

When DirectML is selected, compilation should remain disabled:

``` text
--use_compile 0
```

------------------------------------------------------------------------

# Mixed Precision

The upstream mixed-precision path relies on CUDA AMP/autocast, which is
not used for DirectML.

Dense DirectML pretraining now uses:

``` text
Compute dtype:       float16
Static loss scale:   1024
Optimizer weights:   float32 master weights
AdamW epsilon:       1e-8
```

The earlier pure-FP16 optimizer path could remain finite with
`AdamW epsilon = 1e-4` but was later found to converge toward a
self-copying model during long pretraining.

The FP32 master-weight path preserves FP16 compute while keeping
optimizer updates in FP32. This is a backend-specific precision
workaround, not CUDA AMP.

------------------------------------------------------------------------

# Performance Considerations

DirectML training is functional, and M4 showed that performance
measurements can be strongly distorted by synchronization introduced by
instrumentation.

Real-data stability testing produced:

``` text
32 × 340 → OOM
16 × 340 → OOM after first accumulation cycle
 8 × 340 → PASS for 100 bounded steps
```

The current reference configuration is:

``` text
batch_size = 8
max_seq_len = 340
gradient_accumulation_steps = 8
```

An initial 100-step benchmark measured approximately:

``` text
Average iteration time:      5.280 s
Samples / second:            1.52
Effective tokens / second:   304.81
Estimated epoch duration:    9.70 days
```

Further investigation showed that this timing path introduced
significant DirectML-to-CPU synchronization overhead.

After removing unnecessary synchronization from the critical training
and measurement path, the same 100-step reference workload measured:

``` text
Average iteration time:      0.482 s
Samples / second:            16.61
Effective tokens / second:   3341.37
Padded tokens / second:      5647.46
Estimated epoch duration:    21.24 h
Estimated epoch duration:    0.89 days
```

The corrected iteration time is approximately 10.95× faster than the
initial measurement.

The AdamW `aten::lerp.Scalar_out` CPU fallback remains present, but the
large initial slowdown should not be attributed to this fallback alone.

The corrected benchmark substantially improves the practical outlook for
DirectML training.

A subsequent `1000`-step pretraining run using DirectML FP16, static
loss scale `1024`, AdamW epsilon `1e-4`, `batch_size = 8`,
`max_seq_len = 340`, and gradient accumulation `8` completed
successfully with finite losses throughout the run.

Final FP16 monitoring measured approximately `7,769.62 MB` peak
dedicated VRAM, about `33.15%` below the earlier FP32 peak.

The validated reference configuration is therefore considered
practically viable on the tested hardware. This conclusion remains
hardware- and workload-specific and does not imply that every
configuration passing a short compatibility benchmark is suitable for
sustained training.

Detailed measurements and historical results are maintained in
[`directml_benchmarks.md`](directml_benchmarks.md).

------------------------------------------------------------------------


# MoE Sparse Routing

The upstream sparse MoE routing path relies on scatter-like operations
that are not fully supported by the tested DirectML forward/backward
path.

DirectML therefore uses a scatter-free compatibility path, while CPU and
CUDA retain the original sparse routing implementation.

The DirectML fallback computes all experts and should not be interpreted
as native sparse-MoE performance.

------------------------------------------------------------------------

# Agent RL Full-Sequence Attention Mask

During Agent RL validation, rollout generation remained finite while
policy/reference full-sequence recomputation with
`attention_mask=full_mask` produced NaN logits in DirectML FP16.

The validated Agent batches use right padding. The DirectML Agent path
therefore avoids this problematic mask during policy/reference
recomputation while retaining the response mask used by the training
objective.

# Cross-Entropy Mean Reduction

On the tested DirectML backend,
`F.cross_entropy(..., ignore_index=-100, reduction="mean")` does not
normalize padded causal-LM batches over valid tokens as expected.

MiniMind therefore computes the summed cross-entropy and divides
explicitly by the number of non-ignored target tokens.

This workaround preserves correct token-level loss normalization during
DirectML training.

------------------------------------------------------------------------

# Scope

This document only tracks limitations that remain relevant to the
current DirectML implementation.

Resolved implementation bugs and robustness issues should be documented
in [`directml_issues.md`](directml_issues.md) and
[`update_log.md`](update_log.md) rather than retained as active DirectML
limitations.
