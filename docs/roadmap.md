# MiniMind DirectML Roadmap

This document tracks the development milestones of the MiniMind DirectML adaptation.

---

## M1 — DirectML Foundation ✅

Establish and validate the basic DirectML environment.

- [x] Set up a Windows + DirectML environment
- [x] Replace the CUDA-oriented PyTorch dependency with `torch-directml`
- [x] Initialize a DirectML device
- [x] Move MiniMind models and tensors to DirectML
- [x] Validate the forward pass
- [x] Validate loss computation
- [x] Validate the backward pass
- [x] Validate the AdamW optimizer step
- [x] Add DirectML compatibility tests with Pytest
- [x] Validate installation from a clean virtual environment

---

## M2 — MiniMind Training on DirectML ✅

Adapt the actual MiniMind training pipeline to run on DirectML.

- [x] Audit CUDA-specific code in the training pipeline
- [x] Introduce DirectML-compatible device handling
- [x] Adapt mixed precision and CUDA-specific utilities where necessary
- [x] Run a minimal pretraining job on DirectML
- [x] Verify that the training loss evolves correctly
- [x] Save a model checkpoint
- [x] Reload the checkpoint successfully
- [x] Run inference using the trained checkpoint

---

## M3 — Full Training Pipeline ✅

Validate the main MiniMind training stages with DirectML.

- [x] Validate pretraining
- [x] Validate supervised fine-tuning (SFT)
- [x] Validate LoRA training
- [x] Validate additional training stages supported by MiniMind
- [x] Validate checkpoint compatibility between training stages
- [x] Document unsupported or partially supported DirectML operations

---

## M4 — Performance & Stability 🚧

Evaluate the practical usability of MiniMind with DirectML.

- [ ] Measure GPU memory usage
- [ ] Measure training throughput
- [ ] Identify CPU fallback operations
- [ ] Evaluate the performance impact of CPU fallbacks
- [ ] Test longer training runs
- [ ] Improve stability and error handling
- [ ] Document known limitations

---

## M5 — DirectML-ready Release

Prepare the fork for reproducible use by other Windows users.

- [ ] Finalize installation documentation
- [ ] Document the supported training workflows
- [ ] Add troubleshooting documentation
- [ ] Clean up remaining CUDA assumptions
- [ ] Validate the project from a fresh clone
- [ ] Publish the first stable DirectML-ready version