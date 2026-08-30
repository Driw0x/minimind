# DirectML Limitations

This document tracks unsupported, partially supported, and fallback
operations identified while running and testing MiniMind on DirectML.

The list reflects the current state of the DirectML adaptation and may
evolve as additional training workflows and hardware configurations are tested.

## Training

| Feature / Operation | Status | Workaround |
| --- | --- | --- |
| AdamW `aten::lerp.Scalar_out` | Partial | Automatic CPU fallback |
| InternLM2 Reward Model causal mask | Unsupported | Reward Model runs on CPU |
| `torch.compile` | Unsupported | Use `--use_compile 0` |
| CUDA AMP / autocast | Not enabled | Standard DirectML execution path |

## Known fallbacks

### AdamW

`aten::lerp.Scalar_out` is currently unsupported by the DirectML backend.

PyTorch automatically falls back to CPU, allowing training to continue.
This may have a performance impact.

The performance impact will be evaluated during M4.

## Reward Model

The InternLM2 Reward Model used by GRPO, PPO and Agent RL cannot currently
run entirely on DirectML because of an incompatibility in its causal-mask
operations.

MiniMind DirectML therefore uses:

- MiniMind policy: DirectML
- MiniMind reference model: DirectML
- Reward Model: CPU

This fallback is handled automatically by the training utilities.

## torch.compile

`torch.compile` is currently disabled when DirectML is selected.

Use:

`--use_compile 0`

## Mixed precision

The current mixed-precision implementation relies on CUDA AMP/autocast and
is therefore not enabled for the DirectML training path.

## Training robustness

### Empty distillation masks

Some truncated SFT samples may contain no supervised tokens within the
configured sequence length.

This previously caused the distillation KL-divergence computation to fail
when receiving empty logits.

The distillation loss now handles empty token selections safely by returning
a zero loss connected to the computation graph.

## Performance considerations

CPU fallbacks allow unsupported operations to remain functional but may
reduce training performance.

The performance impact of these fallbacks has not yet been quantified.

M4 — Performance & Stability will evaluate:

- GPU memory usage;
- training throughput;
- CPU fallback frequency;
- performance impact of CPU fallbacks;
- longer training runs and stability.