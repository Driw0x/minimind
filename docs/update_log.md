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

---

## M3 — Training Pipeline Compatibility

* Extended device-aware handling across the MiniMind training utilities.
* Removed remaining assumptions that model execution always uses CUDA.
* Updated trainer utilities to handle DirectML devices consistently.
* Added device-aware reward model handling for alignment training workflows.
* Ensured reward model inputs and model execution use compatible devices.
* Added handling for empty token sequences to prevent invalid model inputs during reward computation.
* Improved compatibility of the training pipeline across pretraining, SFT, DPO and GRPO workflows.
* Added checkpoint compatibility tests for:

  * pretraining;
  * full SFT;
  * DPO;
  * GRPO.
* Validated checkpoint save/load compatibility across the supported training workflows.
* Ran the complete test suite successfully with **34 tests passing**.
* Documented known DirectML limitations and expected CPU fallbacks.
* Confirmed that the codebase is ready for full training validation on DirectML.
