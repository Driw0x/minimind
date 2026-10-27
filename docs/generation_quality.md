# Model Output Quality

[English](generation_quality.md) | [Français](generation_quality.fr.md) | [简体中文](generation_quality.zh-CN.md)

[Project README](../README.md)


## Objective

Document and compare the generation quality of MiniMind checkpoints using the automatic test prompts provided by the upstream project.

The comparison currently covers:

- Mini Pretrain checkpoint
- Mini Full SFT checkpoint
- Full Pretrain checkpoint
- Full SFT checkpoint
- MoE Full Pretrain checkpoint
- MoE Full SFT checkpoint

This document evaluates model outputs only. It does not introduce new inference features or modify the training pipeline.

---

## Current evaluated models

| Model | Training data | Status |
| --- | --- | --- |
| Mini Pretrain | Mini pretraining dataset | Evaluated |
| Mini Full SFT | Mini pretraining + Mini SFT datasets | Evaluated |
| Full Pretrain | Complete pretraining dataset | Evaluated |
| Full SFT | Complete pretraining + SFT datasets | Evaluated |
| MoE Full Pretrain | Complete pretraining dataset | Evaluated |
| MoE Full SFT | Complete pretraining + SFT datasets | Evaluated |

The current Mini models contain approximately:

```text
Model parameters: 63.91M
```

---

## Quantitative reference

Current validation measurements:

| Metric | Mini Pretrain | Mini Full SFT |
| --- | ---: | ---: |
| Loss on pretrain data | 1.8275 | 1.9992 |
| Loss on SFT data | 2.5207 | 1.5663 |
| External repetition | 0.795 | 0.361 |

The Full SFT checkpoint therefore shows a clear improvement on the SFT objective and substantially less repetitive generation.

The slightly higher loss on pretraining data after SFT is expected to be interpreted separately from generation quality.

<a id="full-data-dense-reference"></a>
### Full-data Dense/MoE reference

The current Dense and MoE checkpoints were trained on the complete T2T datasets and evaluated
on the same Mini reference datasets (`seed=42`, 256 samples each):

| Model | Pretrain reference loss | SFT reference loss | External repetition |
| --- | ---: | ---: | --- |
| Dense Full Pretrain | 1.6372 | 2.8685 | Not retained as reliable |
| Dense Full SFT | 1.8730 | 1.3771 | 0.244 |
| MoE Full Pretrain | 1.5468 | 2.1544 | 0.842 (methodological caveat) |
| MoE Full SFT | 1.7544 | 1.2994 | 0.262 |

MoE perplexity on the pretrain / SFT references is `4.70 / 8.62` before SFT
and `5.78 / 3.67` after Full SFT.

The references are `dataset/pretrain_t2t_mini.jsonl` and `dataset/sft_t2t_mini.jsonl`.
These samples come from training data: losses and perplexities measure fit to those
distributions, not generalization on an independent test set.

The Dense Full Pretrain repetition aggregate is not retained because several generations
reached the token limit while decoding to an empty visible response.

External repetition comes from the eight Chinese prompts in `scripts/diagnose_validation.py`,
with deterministic decoding (`do_sample=False`, up to 256 new tokens). For Pretrain,
this benchmark uses a chat template, whereas `eval_llm.py` uses BOS + raw prompt
and sampling (`do_sample=True`). The MoE Pretrain score `0.842` must therefore be
interpreted with this protocol difference; it does not describe raw completion behavior.

MoE has lower reference losses than Dense at the corresponding stages, but MoE Full SFT
has slightly higher external repetition (`0.262` vs `0.244`). These results do not
establish general MoE superiority over Dense.

---

# Upstream generation tests

## Evaluation method

Both checkpoints are evaluated with the automatic test mode already provided by the upstream MiniMind project.

The same:

- inference script;
- generation configuration;
- tokenizer;
- built-in prompts;
- model architecture;

should be used for both checkpoints so that the comparison isolates the effect of training.

No custom benchmark is introduced at this stage.

---

## Mini Pretrain

### General behavior

The Mini Pretrain checkpoint can generate text, but its outputs show important limitations for conversational use.

Observed characteristics:

- weak instruction following;
- frequent repetition;
- unstable continuation;
- limited ability to produce a direct answer;
- poor conversational structure;
- unreliable factual content;
- limited reasoning quality.

The high external repetition measurement (`0.795`) is consistent with the repetitive behavior visible in generation.

### Interpretation

The checkpoint demonstrates that pretraining produced a functioning language model, but pretraining alone is insufficient for reliable instruction-following behavior.

It serves primarily as the baseline against which Full SFT is evaluated.

---

## Mini Full SFT

### General behavior

The Mini Full SFT checkpoint produces noticeably more structured and conversational outputs than the Mini Pretrain checkpoint.

Observed improvements include:

- better instruction following;
- more direct answers;
- better conversational structure;
- substantially less repetition;
- more stable generations.

The external repetition metric decreases from:

```text
0.795 -> 0.361
```

and the loss on SFT data decreases from:

```text
2.5207 -> 1.5663
```

These results are consistent with the qualitative improvement observed in the upstream automatic tests.

### Remaining limitations

The generated answers can be coherent without being factually correct.

Current limitations therefore include:

- unreliable factual knowledge;
- incorrect answers expressed in plausible language;
- limited reasoning ability;
- possible hallucinations;
- sensitivity to the limited Mini training datasets.

A coherent answer must therefore not be interpreted as a correct answer.

---

# Mini Pretrain vs Mini Full SFT

| Criterion | Mini Pretrain | Mini Full SFT |
| --- | --- | --- |
| Text generation | Functional | Functional |
| Conversational structure | Weak | Improved |
| Instruction following | Weak | Improved |
| Repetition | High | Much lower |
| Answer coherence | Limited | Improved |
| Factual reliability | Limited | Still limited |
| Reasoning quality | Limited | Still limited |

## Current conclusion

Full SFT provides a clear improvement in **how the model answers**.

It improves conversational stability, instruction following and repetition compared with the Mini Pretrain checkpoint.

However, the current Mini Full SFT model should not yet be considered factually reliable.

The current results therefore demonstrate:

```text
Mini Pretrain
    -> basic language modeling capability

Mini Full SFT
    -> improved instruction-following and conversational behavior
    -> factual and reasoning limitations remain
```

---

# Full-dataset evaluation

The same evaluation protocol has now been run on the Dense checkpoints trained on the
complete MiniMind T2T datasets.

## Full Pretrain

The Full Pretrain checkpoint is more coherent in the upstream automatic test than the
Mini Pretrain checkpoint and reaches a lower pretrain reference loss (`1.6372` vs
`1.8275`).

It still shows repetition, continuation drift, factual inaccuracies, and limited
instruction-following behavior, which is expected before SFT.

## Full SFT

The Full SFT checkpoint produces coherent and stable assistant-style responses and
improves SFT reference loss from `2.8685` to `1.3771`.

Its deterministic external repetition metric is `0.244`.

Manual review still shows known limitations:

- factual accuracy remains unreliable;
- arithmetic reasoning can fail;
- code correction can remain incorrect;
- strict instruction following is imperfect.

The complete SFT dataset therefore improves assistant behavior and stability, but does
not make the 63.91M model reliably factual or reasoning-correct.

## Current comparison

| Criterion | Mini Pretrain | Full Pretrain | Mini Full SFT | Full SFT |
| --- | --- | --- | --- | --- |
| Instruction following | Weak | Still limited | Improved | Improved but imperfect |
| Conversational structure | Weak | Improved | Improved | Improved |
| Repetition | High | Still present | Lower | Low |
| Factual reliability | Limited | Limited | Limited | Still limited |
| Reasoning quality | Limited | Limited | Limited | Still limited |
| Overall output stability | Weak | Improved | Improved | Improved |

---

# MoE full-dataset evaluation

Both training stages and their initial AMD ROCm evaluation are complete.
`pretrain_768_moe.pth` was trained on `pretrain_t2t.jsonl`;
`full_sft_768_moe.pth` was trained on `sft_t2t.jsonl`, initialized from MoE Pretrain.
The architecture uses hidden size 768, 8 layers, 4 experts per layer and top-1 routing:
198.42M total parameters and a reported 63.94M active parameters.

## MoE Full Pretrain

The checkpoint loads and evaluates on ROCm. Its lower reference losses indicate
better fit to the sampled training distributions. External chat-formatted generation
remains highly repetitive (`0.842`), subject to the protocol caveat above.

## MoE Full SFT

SFT lowers SFT reference loss from `2.1544` to `1.2994` and external repetition from
`0.842` to `0.262`, while pretrain reference loss rises from `1.5468` to `1.7544`.
This indicates stronger SFT fit with measurable forgetting of the pretraining distribution.

Manual review still finds:

- arithmetic errors: `12 - 5 + 3` answered as `16` instead of `10`;
- incorrect Fibonacci code;
- imperfect adherence to format instructions;
- repetition and truncated responses.

Lower reference loss does not make these generations reliably correct.

---

# Final status

## Mini models

- [x] Mini Pretrain trained
- [x] Mini Full SFT trained
- [x] Upstream automatic generation tests executed
- [x] Mini Pretrain and Mini Full SFT compared
- [x] Full SFT conversational improvement observed
- [x] Remaining factual and reasoning limitations identified

## Full-data models

- [x] Complete Full Pretrain training
- [x] Evaluate Full Pretrain with upstream automatic tests
- [x] Complete Full SFT training
- [x] Evaluate Full SFT with upstream automatic tests
- [x] Compare Mini and full-data checkpoints
- [x] Document whether factual accuracy, reasoning and generation quality improve with the complete datasets

## MoE models

- [x] Complete MoE Pretrain and MoE Full SFT training
- [x] Complete initial ROCm evaluation of both checkpoints
- [x] Compare Dense/MoE reference results and document generation limits
