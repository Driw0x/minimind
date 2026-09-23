import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), "
    r"loss: ([\d.eE+-]+), "
    r"logits_loss: ([\d.eE+-]+), "
    r"aux_loss: ([\d.eE+-]+), "
    r"lr: ([\d.eE+-]+), "
    r"epoch_time: ([\d.eE+-]+)min"
)


def parse_log(log_path: Path):
    records = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = LOG_PATTERN.search(line)
            if not match:
                continue

            epoch = int(match.group(1))
            epochs = int(match.group(2))
            step = int(match.group(3))
            steps_per_epoch = int(match.group(4))

            global_step = (epoch - 1) * steps_per_epoch + step
            total_steps = epochs * steps_per_epoch
            progress_pct = 100.0 * global_step / total_steps

            records[global_step] = {
                "epoch": epoch,
                "epochs": epochs,
                "step": step,
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "progress_pct": progress_pct,
                "loss": float(match.group(5)),
                "logits_loss": float(match.group(6)),
                "aux_loss": float(match.group(7)),
                "lr": float(match.group(8)),
                "eta_min": float(match.group(9)),
            }

    return [records[step] for step in sorted(records)]


def moving_average(values, window):
    if window <= 1:
        return list(values)

    result = []

    for index in range(len(values)):
        start = max(0, index - window + 1)
        current = values[start:index + 1]
        result.append(sum(current) / len(current))

    return result


def parse_run_spec(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=PATH, for example DirectML=logs/directml/pretrain_768.log"
        )

    label, path = spec.split("=", 1)

    if not label.strip():
        raise argparse.ArgumentTypeError("Run label cannot be empty.")

    return label.strip(), Path(path.strip())


def save_line_plot(
    runs,
    x_key,
    y_key,
    output,
    title,
    xlabel,
    ylabel,
    smooth=1,
    relative_to_first=False,
):
    plt.figure(figsize=(11, 6))
    plotted = False

    for label, records in runs.items():
        if not records:
            continue

        x = [record[x_key] for record in records]
        y = [record[y_key] for record in records]

        if relative_to_first:
            first = y[0]
            if first == 0:
                continue
            y = [100.0 * (first - value) / first for value in y]

        if smooth > 1:
            y = moving_average(y, smooth)

        plt.plot(x, y, linewidth=2, label=label)
        plotted = True

    if not plotted:
        plt.close()
        return False

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()

    return True


def write_summary(runs, output):
    rows = []

    for label, records in runs.items():
        if not records:
            continue

        first = records[0]
        last = records[-1]

        loss_drop = first["loss"] - last["loss"]
        loss_drop_pct = (
            100.0 * loss_drop / first["loss"]
            if first["loss"] != 0
            else 0.0
        )

        rows.append({
            "run": label,
            "points": len(records),
            "first_step": first["global_step"],
            "last_step": last["global_step"],
            "steps_per_epoch": last["steps_per_epoch"],
            "progress_pct": last["progress_pct"],
            "first_loss": first["loss"],
            "last_loss": last["loss"],
            "loss_drop": loss_drop,
            "loss_drop_pct": loss_drop_pct,
            "first_lr": first["lr"],
            "last_lr": last["lr"],
            "last_eta_min": last["eta_min"],
        })

    fieldnames = [
        "run",
        "points",
        "first_step",
        "last_step",
        "steps_per_epoch",
        "progress_pct",
        "first_loss",
        "last_loss",
        "loss_drop",
        "loss_drop_pct",
        "first_lr",
        "last_lr",
        "last_eta_min",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare one or more MiniMind pretraining logs."
    )

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Training run to compare. Can be repeated. "
            "Example: --run DirectML=logs/directml/pretrain_768.log "
            "--run ROCm=logs/rocm/pretrain_768.log"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/figures"),
        help="Directory where graphs and summary.csv are saved.",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Moving-average window over logged points. Use 1 to disable.",
    )

    args = parser.parse_args()

    if args.smooth <= 0:
        raise ValueError("--smooth must be >= 1")

    runs = {}

    for spec in args.run:
        label, path = parse_run_spec(spec)

        if not path.exists():
            raise FileNotFoundError(f"{label}: log file not found: {path}")

        records = parse_log(path)

        if not records:
            raise RuntimeError(
                f"{label}: no training entries found in {path}. "
                "Expected lines like "
                "'Epoch:[1/2](100/39695), loss: ...'"
            )

        runs[label] = records

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    plots = [
        (
            "loss_vs_step.png",
            "global_step",
            "loss",
            "Pretraining loss vs training step",
            "Training step",
            "Loss",
            args.smooth,
            False,
        ),
        (
            "loss_vs_progress.png",
            "progress_pct",
            "loss",
            "Pretraining loss vs training progress",
            "Training progress (%)",
            "Loss",
            args.smooth,
            False,
        ),
        (
            "logits_loss_vs_step.png",
            "global_step",
            "logits_loss",
            "Logits loss vs training step",
            "Training step",
            "Logits loss",
            args.smooth,
            False,
        ),
        (
            "aux_loss_vs_step.png",
            "global_step",
            "aux_loss",
            "Auxiliary loss vs training step",
            "Training step",
            "Aux loss",
            args.smooth,
            False,
        ),
        (
            "learning_rate_vs_step.png",
            "global_step",
            "lr",
            "Learning rate schedule",
            "Training step",
            "Learning rate",
            1,
            False,
        ),
        (
            "eta_vs_step.png",
            "global_step",
            "eta_min",
            "Estimated remaining epoch time",
            "Training step",
            "ETA (minutes)",
            1,
            False,
        ),
        (
            "loss_reduction_pct_vs_step.png",
            "global_step",
            "loss",
            "Loss reduction relative to first logged point",
            "Training step",
            "Loss reduction (%)",
            args.smooth,
            True,
        ),
    ]

    for (
        filename,
        x_key,
        y_key,
        title,
        xlabel,
        ylabel,
        smooth,
        relative,
    ) in plots:
        output = output_dir / filename

        if save_line_plot(
            runs,
            x_key,
            y_key,
            output,
            title,
            xlabel,
            ylabel,
            smooth=smooth,
            relative_to_first=relative,
        ):
            generated.append(output)

    summary_path = output_dir / "summary.csv"
    write_summary(runs, summary_path)

    print()
    print("=" * 72)
    print("MiniMind pretraining comparison")
    print("=" * 72)

    for label, records in runs.items():
        first = records[0]
        last = records[-1]
        loss_drop = first["loss"] - last["loss"]
        loss_drop_pct = 100.0 * loss_drop / first["loss"]

        print(f"{label}")
        print(f"  points          : {len(records)}")
        print(f"  steps           : {first['global_step']} -> {last['global_step']}")
        print(f"  progress        : {last['progress_pct']:.2f}%")
        print(f"  loss            : {first['loss']:.4f} -> {last['loss']:.4f}")
        print(f"  loss reduction  : {loss_drop:.4f} ({loss_drop_pct:.2f}%)")
        print(f"  learning rate   : {first['lr']:.8f} -> {last['lr']:.8f}")
        print(f"  last ETA        : {last['eta_min']:.1f} min")
        print()

    print("Generated files:")
    for path in generated:
        print(f"  {path}")

    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
