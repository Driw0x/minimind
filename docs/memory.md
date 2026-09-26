# Memory

## Status

Ready for implementation as the next milestone.

The current `full_sft_768.pth` checkpoint is the validated experimental baseline. The training and diagnostic pipeline has been verified on ROCm, SFT improves conversational stability, and the remaining model/data limitations are documented. Memory should therefore be added as a runtime capability without changing the baseline model weights.

---

## Objective

Add conversational memory so MiniMind can use relevant information from earlier turns when answering later turns.

The first version should answer a simple question:

> Can the model retain and reuse useful conversation context across multiple turns without retraining?

Memory is not intended to repair factual knowledge, reasoning quality, or SFT dataset issues. Those remain model/data concerns.

### Validated baseline

The memory milestone starts from the following reference measurements:

```text
Model parameters                  : 63.91M
Pretrain loss on pretrain data    : 1.8275
Full SFT loss on pretrain data    : 1.9992
Pretrain loss on SFT data         : 2.5207
Full SFT loss on SFT data         : 1.5663
Pretrain external repetition      : 0.795
Full SFT external repetition      : 0.361
```

These measurements define the no-memory baseline. Memory experiments should not be interpreted as improvements to factual knowledge or reasoning unless separate evaluation demonstrates that.

---

## Scope

### Initial scope

Implement short-term conversational memory based on prior messages in the current conversation.

The system should:

- retain recent user and assistant turns;
- inject relevant history into the next prompt;
- respect the model context-length limit;
- preserve the existing chat-template format;
- work with the current `full_sft` checkpoint;
- remain independent from training.

### Later scope

Only after short-term memory is validated, consider:

- conversation summarization;
- persistent memory across sessions;
- selective retrieval of older memories;
- semantic retrieval or vector search.

These are intentionally deferred to keep the first implementation small and measurable.

---

## Non-goals

The first memory milestone should not:

- fine-tune or modify model weights;
- introduce a vector database;
- implement RAG over external documents;
- store unlimited conversation history;
- silently persist user information across unrelated sessions;
- attempt to solve the model's factual or reasoning limitations.

---

## Minimal design

### 1. Conversation history

Maintain an ordered list of chat messages:

```text
system
user
assistant
user
assistant
...
```

Each new user message is appended to the history before generation, and the generated assistant response is appended afterward.

### 2. Context budget

Before generation, estimate the number of tokens required by the conversation history.

If the history exceeds the configured context budget, remove the oldest non-system turns first.

The initial implementation should prefer deterministic truncation over summarization. This keeps behavior easy to debug and evaluate.

### 3. Prompt construction

Use the same tokenizer chat template already used by MiniMind inference and SFT data processing.

Memory should change only the message list passed to the template, not the model architecture.

### 4. Separation from model memory

Runtime conversation memory is external state. It is not learned model memory.

```text
Model weights
    -> learned during pretraining/SFT

Conversation memory
    -> runtime messages injected into the prompt
```

This distinction should remain explicit in both code and documentation.

---

## Evaluation plan

Memory should be evaluated against the existing `full_sft` baseline.

### Test 1 — Direct recall

Conversation:

```text
User: My project is called CS2Guard.
Assistant: ...
User: What is the name of my project?
```

Expected behavior: answer `CS2Guard` using conversation history.

### Test 2 — Multi-turn attribute recall

Provide two or three independent facts across separate messages and ask for one of them several turns later.

Expected behavior: retrieve the correct earlier fact without mixing it with unrelated turns.

### Test 3 — Update

Conversation:

```text
User: My preferred editor is VS Code.
...
User: I switched to PyCharm.
...
User: Which editor do I use now?
```

Expected behavior: prefer the latest explicit information.

### Test 4 — Irrelevant-history resistance

Insert unrelated messages between the stored fact and the recall question.

Expected behavior: preserve the relevant fact and avoid copying irrelevant history into the answer.

### Test 5 — Context overflow

Create a conversation longer than the configured token budget.

Expected behavior:

- no crash;
- oldest turns are removed predictably;
- recent turns remain usable;
- generated prompt stays within the model limit.

### Test 6 — Baseline regression

Run `scripts/diagnose_validation.py` with memory disabled and with an empty memory state.

Expected behavior:

- local loss results remain unchanged because runtime memory does not modify model weights;
- empty-memory generation remains equivalent to the current no-memory baseline apart from unavoidable implementation-level differences;
- the external repetition metric should remain close to the current Full SFT reference value (`0.361`).

---

## Metrics

The first memory milestone should track simple, interpretable metrics:

- recall success rate;
- update/correction success rate;
- number of retained turns;
- prompt token count;
- number of dropped turns after budget enforcement;
- generation latency before and after memory integration.

A vector-retrieval metric is not needed until semantic long-term memory exists.

---

## Success criteria

The initial memory implementation is considered validated when:

1. direct recall works reliably on a small deterministic test set;
2. newer information overrides older conflicting information;
3. irrelevant turns do not break recall;
4. context overflow is handled without exceeding the configured limit;
5. memory can be disabled cleanly;
6. the existing no-memory validation benchmark still runs;
7. no model retraining is required.

---

## First implementation target

Keep the first implementation minimal:

```text
conversation history
    -> chat template
    -> token-budget trimming
    -> model.generate(...)
    -> append assistant response
```

No retrieval index, summarizer, persistent store, or training change is required for this milestone.

---

## Recommended implementation order

```text
1. Add conversation-history state
2. Reuse the existing chat template
3. Add token-budget enforcement
4. Add an option to disable/clear memory
5. Add deterministic memory tests
6. Compare with the full_sft baseline
7. Document results
8. Only then consider summarization or persistent memory
```

---

## Relationship to current MiniMind findings

Current validation shows that Full SFT substantially improves conversational stability compared with the pretraining checkpoint, while factual accuracy and reasoning are still limited.

Memory should therefore be evaluated as a separate capability:

```text
Pretrain
   -> Full SFT
      -> validated baseline
         -> runtime Memory
```

The memory milestone should not be used to hide or compensate for training-quality problems. Its purpose is to test whether the validated model can make effective use of prior conversational context.
