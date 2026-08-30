# MiniMind — Update Log

This document provides a chronological record of the main changes made to the MiniMind DirectML fork.

For technical problems, causes, and solutions, see `project_memory.md`.

---

## DirectML Compatibility

* Configured PyTorch 2.4.1 with `torch-directml`.
* Added DirectML device detection.
* Validated model execution on DirectML.
* Validated forward pass and loss computation.
* Validated backward pass and gradient computation.
* Validated AdamW optimizer step.
* Validated a complete minimal training step.
* Confirmed that known DirectML CPU fallbacks do not prevent training.

---

## MiniMind Integration

* Added `directml` as an explicit device option.
* Integrated DirectML into the existing MiniMind device handling.
* Added DirectML support to the relevant training workflow.
* Added DirectML support to `eval_llm.py`.
* Validated a small `hidden_size=128`, 2-layer training configuration.
* Validated model weight generation.
* Validated checkpoint loading.
* Validated evaluation on DirectML.
* Validated text generation on DirectML.
* Confirmed the complete training-to-evaluation DirectML workflow.
