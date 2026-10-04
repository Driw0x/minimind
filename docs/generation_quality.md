# Model Output Quality

## Objective

Document and compare the generation quality of MiniMind checkpoints using the automatic test prompts provided by the upstream project.

The comparison currently covers:

- Mini Pretrain checkpoint
- Mini Full SFT checkpoint
- Full Pretrain checkpoint
- Full SFT checkpoint

This document evaluates model outputs only. It does not introduce new inference features or modify the training pipeline.

---

## Current evaluated models

| Model | Training data | Status |
| --- | --- | --- |
| Mini Pretrain | Mini pretraining dataset | Evaluated |
| Mini Full SFT | Mini pretraining + Mini SFT datasets | Evaluated |
| Full Pretrain | Complete pretraining dataset | Evaluated |
| Full SFT | Complete pretraining + SFT datasets | Evaluated |

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

### Full-data Dense reference

The current Dense checkpoints were trained on the complete T2T datasets and evaluated
on the same Mini reference datasets (`seed=42`, 256 samples each):

| Metric | Full Pretrain | Full SFT |
| --- | ---: | ---: |
| Loss on pretrain reference | 1.6372 | 1.8730 |
| Loss on SFT reference | 2.8685 | 1.3771 |
| External repetition | Not retained as reliable | 0.244 |

The Full Pretrain repetition aggregate is not retained because several generations
reached the token limit while decoding to an empty visible response.


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