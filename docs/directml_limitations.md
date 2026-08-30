# MiniMind — DirectML Limitations

This document tracks the current unsupported, partially supported, and fallback operations identified while running MiniMind on DirectML.

Detailed explanations of encountered problems, their causes, and implemented solutions are documented in [`directml_issues.md`](directml_issues.md).

Performance measurements are documented in [`directml_benchmarks.md`](directml_benchmarks.md).

---

# Current Limitations

| Feature / Operation                | Status      | Workaround                       |
| ---------------------------------- | ----------- | -------------------------------- |
| AdamW `aten::lerp.Scalar_out`      | Partial     | Automatic CPU fallback           |
| InternLM2 Reward Model causal mask | Unsupported | Reward Model runs on CPU         |
| `torch.compile`                    | Unsupported | Use `--use_compile 0`            |
| CUDA AMP / autocast                | Not enabled | Standard DirectML execution path |

---

# AdamW CPU Fallback

The AdamW optimizer uses:

```text
aten::lerp.Scalar_out
```

which is not currently supported natively by the DirectML backend.

`torch-directml` automatically falls back to CPU execution for this operation.

Training remains functional, but the fallback may affect performance.

The performance impact will be evaluated during M4.

---

# Reward Model

The InternLM2 Reward Model used by alignment workflows cannot currently execute entirely on DirectML because of incompatible causal-mask operations.

The current execution strategy is:

```text
MiniMind policy          → DirectML
MiniMind reference model → DirectML
Reward Model             → CPU
```

Required inputs are moved to CPU before reward-model inference, and reward values are transferred back to the training device when necessary.

This fallback is handled by the shared training utilities.

---

# torch.compile

`torch.compile` is currently unsupported for the DirectML training path.

When DirectML is selected, compilation should remain disabled:

```text
--use_compile 0
```

---

# Mixed Precision

The upstream mixed-precision path relies on CUDA AMP/autocast.

CUDA AMP is therefore not enabled for DirectML.

DirectML currently uses the standard execution path without CUDA-specific automatic mixed precision.

---

# Performance Considerations

DirectML training is functional, but full-scale MiniMind training can be impractically slow on the tested setup.

During a real pretraining run using the standard `pretrain_t2t_mini.jsonl` dataset with:

```text
batch_size = 32
max_seq_len = 340
epochs = 2
```

the training loop reported approximately:

```text
158780 steps / epoch
~5.9 seconds / step
~10.9 days / epoch
```

The run confirmed that real training works on DirectML, but also showed that compatibility does not imply practical full-training performance.

CPU fallbacks, including the AdamW `aten::lerp.Scalar_out` fallback, may further reduce training performance.

For DirectML validation, bounded real-data training runs using `--max_steps` are therefore preferred over requiring completion of the full dataset.

Full training remains possible, but is considered optional and hardware-dependent.

---

# Scope

This document only tracks limitations that remain relevant to the current DirectML implementation.

Resolved implementation bugs and robustness issues should be documented in [`directml_issues.md`](directml_issues.md) and [`update_log.md`](update_log.md) rather than retained as active DirectML limitations.
