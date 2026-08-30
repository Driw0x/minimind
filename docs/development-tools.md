# Development Tools

This document describes the development utilities added to support and validate the DirectML adaptation of MiniMind.

## DirectML Compatibility Audit

The DirectML audit scans the project for code that may rely on CUDA-specific behavior.

Run from the project root:

```bash
python scripts/audit_directml.py
```

The generated report is written to:

```text
docs/directml_audit.md
```

The audit detects potential compatibility concerns including:

- CUDA-specific references;
- device placement assumptions;
- CUDA utilities;
- automatic mixed precision (AMP);
- distributed training code.

The report is intended as a static analysis aid. A reported occurrence does not necessarily represent an incompatibility and must be reviewed manually.

---

## Test Fixture Generation

Small deterministic datasets are used by the automated test suite to validate MiniMind training stages without requiring the full training datasets.

Generate the fixtures with:

```bash
python scripts/generate_test_fixtures.py
```

The generated datasets are stored in:

```text
tests/fixtures/
```

They provide lightweight inputs for testing training workflows such as:

- pretraining;
- supervised fine-tuning;
- DPO;
- Agent RL;
- other supported training stages.

These fixtures are intended for compatibility and smoke testing, not for actual model training.

---

## Sequential Training

The `train_all.ps1` PowerShell script runs the main MiniMind training stages sequentially on Windows.

Run it from the project root:

```powershell
.\scripts\train_all.ps1
```

The script executes each training stage one after another. A new stage starts only after the previous one has completed successfully.

This avoids running multiple DirectML training processes in parallel, which could increase GPU memory usage and reduce stability.

If a training stage fails, the script stops immediately instead of continuing with the remaining stages.

The script keeps the standard MiniMind output structure:

```text
out/            # Generated model weights
checkpoints/    # Training resume checkpoints
```

This utility is intended to simplify sequential validation and execution of the DirectML training pipeline on Windows.