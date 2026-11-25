
#!/usr/bin/env python3
"""
analyze_mapnet_runs.py 

Analyze MapNet training runs and their evaluations.

Expected structure:

experiments/
└── Bebop/
    └── v2_3/
        ├── bebop_mapnet_baseline/
        │   ├── log.txt
        │   ├── best_model.pth.tar
        │   └── epoch_*.pth.tar
        └── ...

eval_results/
└── Bebop/
    └── v2_3/
        ├── bebop_mapnet_baseline/
        │   ├── metrics.json
        │   ├── per_image_errors.csv
        │   └── *.pkl
        └── ...

Outputs:

eval_results/Bebop/v2_3/training_analysis/
    01_training_loss.png
    02_checkpoint_timeline.png
    03_position_error_comparison.png
    04_orientation_error_comparison.png
    05_pose_error_distributions.png
    summary.csv
    report.md

No checkpoints are loaded. No inference is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "Bebop"
DEFAULT_SCENE = "v2_3"

DEFAULT_TOP = 6


# ============================================================
# GENERAL UTILITIES
# ============================================================

def safe_float(value):
    """Convert a value to float, returning NaN if invalid."""
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_int(value):
    """Convert a value to int, returning None if invalid."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_split(value):
    """Normalize split names."""
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ("val", "validation", "test"):
        return "val"

    if value in ("train", "training"):
        return "train"

    return value


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# TRAINING LOG PARSER
# ============================================================

def extract_epoch_and_loss(line, split):
    """
    Extract epoch and loss from MapNet log lines.

    Supports variations such as:

        Train ... Epoch 10 ... Loss -8.123
        Val ... Epoch 10, val_loss 0.813981
        Train Epoch: 10 ... loss: -8.123
        Val Epoch: 10 ... loss: 0.813981

    Only lines explicitly identified as Train or Val are considered.
    """

    if not re.search(r"\b" + split + r"\b", line, re.IGNORECASE):
        return None

    epoch_match = re.search(
        r"\bEpoch\s*[:=#]?\s*(\d+)",
        line,
        re.IGNORECASE,
    )

    if not epoch_match:
        return None

    epoch = int(epoch_match.group(1))

    if split == "val":
        loss_patterns = [
            r"\bval[_ ]?loss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bloss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]
    else:
        loss_patterns = [
            r"\btrain[_ ]?loss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bloss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]

    for pattern in loss_patterns:
        matches = re.findall(pattern, line, re.IGNORECASE)

        if matches:
            # If a line contains multiple loss values, use the last one.
            loss = safe_float(matches[-1])

            if np.isfinite(loss):
                return epoch, loss

    return None


def read_training_log(log_path):
    """
    Read train and validation losses from a MapNet log.

    If multiple records exist for an epoch, the last record is retained.
    """

    train_history = {}
    val_history = {}

    if not log_path.is_file():
        return train_history, val_history

    with log_path.open("r", errors="replace") as file:
        for line in file:

            train_result = extract_epoch_and_loss(line, "train")

            if train_result is not None:
                epoch, loss = train_result
                train_history[epoch] = loss

            val_result = extract_epoch_and_loss(line, "val")

            if val_result is not None:
                epoch, loss = val_result
                val_history[epoch] = loss

    return train_history, val_history


def load_run_history(run_dir):
    """Load train/validation histories and identify the best epoch."""

    log_path = run_dir / "log.txt"

    train_history, val_history = read_training_log(log_path)

    train_epochs = np.array(sorted(train_history), dtype=int)
    train_losses = np.array(
        [train_history[e] for e in train_epochs],
        dtype=float,
    )

    val_epochs = np.array(sorted(val_history), dtype=int)
    val_losses = np.array(
        [val_history[e] for e in val_epochs],
        dtype=float,
    )

    if len(val_epochs):
        best_index = int(np.argmin(val_losses))

        best_epoch = int(val_epochs[best_index])
        best_val_loss = float(val_losses[best_index])
        final_val_loss = float(val_losses[-1])
    else:
        best_epoch = None
        best_val_loss = np.nan
        final_val_loss = np.nan

    return {
        "train_epochs": train_epochs,
        "train_loss": train_losses,
        "val_epochs": val_epochs,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "num_train_epochs": int(len(train_epochs)),
        "num_val_epochs": int(len(val_epochs)),
        "log_path": str(log_path),
    }


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def discover_checkpoints(run_dir):
    """
    Discover checkpoint files without loading them.

    Recognized:
        epoch_000.pth.tar
        epoch_010.pth.tar
        epoch_100.pth.tar
        best_model.pth.tar
    """

    checkpoints = []

    for path in run_dir.glob("epoch_*.pth.tar"):
        match = re.search(r"epoch_(\d+)", path.name)

        if match:
            checkpoints.append(int(match.group(1)))

    checkpoints = sorted(set(checkpoints))

    return {
        "epochs": checkpoints,
        "num_checkpoints": len(checkpoints),
        "has_best_model": (run_dir / "best_model.pth.tar").is_file(),
    }


# ============================================================
# EVALUATION FILES
# ============================================================

def load_metrics_json(eval_dir):
    """Load metrics.json from an evaluation directory."""

    path = eval_dir / "metrics.json"

    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    except (json.JSONDecodeError, OSError) as error:
        print(f"  [!] Could not read {path}: {error}")
        return {}


def load_per_image_errors(eval_dir):
    """Load per-image errors indexed by valid_index."""

    path = eval_dir / "per_image_errors.csv"

    if not path.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as error:
        print(f"  [!] Could not read {path}: {error}")
        return pd.DataFrame()

    required = {
        "valid_index",
        "translation_error_m",
        "orientation_error_deg",
    }

    if not required.issubset(df.columns):
        print(f"  [!] Invalid columns in {path}")
        return pd.DataFrame()

    df = df[list(required)].copy()

    df["valid_index"] = pd.to_numeric(
        df["valid_index"],
        errors="coerce",
    )

    df["translation_error_m"] = pd.to_numeric(
        df["translation_error_m"],
        errors="coerce",
    )

    df["orientation_error_deg"] = pd.to_numeric(
        df["orientation_error_deg"],
        errors="coerce",
    )

    df = df.dropna()
    df["valid_index"] = df["valid_index"].astype(int)

    # Avoid ambiguous matches if a CSV contains duplicate indices.
    df = df.drop_duplicates(
        subset="valid_index",
        keep="last",
    )

    return df.sort_values("valid_index").reset_index(drop=True)


def get_json_metrics(metrics):
    """Extract metrics from the metrics.json structure."""

    position = metrics.get("position", {})
    orientation = metrics.get("orientation", {})
    metadata = metrics.get("metadata", {})

    return {
        "num_images": safe_int(metrics.get("num_images")),
        "position_mean_m": safe_float(position.get("mean_m")),
        "position_median_m": safe_float(position.get("median_m")),
        "position_rmse_m": safe_float(position.get("rmse_m")),
        "position_std_m": safe_float(position.get("std_m")),
        "position_p95_m": safe_float(position.get("p95_m")),
        "orientation_mean_deg": safe_float(orientation.get("mean_deg")),
        "orientation_median_deg": safe_float(orientation.get("median_deg")),
        "orientation_std_deg": safe_float(orientation.get("std_deg")),
        "orientation_p95_deg": safe_float(orientation.get("p95_deg")),
        "split": normalize_split(metadata.get("split")),
        "checkpoint_name": metadata.get("checkpoint_name"),
    }


# ============================================================
# PER-IMAGE COMPARISON
# ============================================================

def summarize_error_dataframe(df, prefix=""):
    """Calculate summary statistics from already-computed errors."""

    if df.empty:
        return {}

    t = df["translation_error_m"].to_numpy(dtype=float)
    q = df["orientation_error_deg"].to_numpy(dtype=float)

    result = {
        "num_images": int(len(df)),
        "position_mean_m": float(np.mean(t)),
        "position_median_m": float(np.median(t)),
        "position_rmse_m": float(np.sqrt(np.mean(t ** 2))),
        "position_std_m": float(np.std(t)),
        "position_p95_m": float(np.percentile(t, 95)),
        "orientation_mean_deg": float(np.mean(q)),
        "orientation_median_deg": float(np.median(q)),
        "orientation_std_deg": float(np.std(q)),
        "orientation_p95_deg": float(np.percentile(q, 95)),
    }

    if prefix:
        return {
            f"{prefix}{key}": value
            for key, value in result.items()
        }

    return result


def get_common_errors(evaluations):
    """
    Find the intersection of valid image indices across compatible runs.

    Runs with a different split are not mixed.
    """

    valid_evaluations = {
        name: data
        for name, data in evaluations.items()
        if not data["errors"].empty
    }

    if not valid_evaluations:
        return {}, set()

    split_groups = {}

    for name, data in valid_evaluations.items():
        split = data["split"] or "unknown"
        split_groups.setdefault(split, []).append(name)

    # The comparison is only valid within a common split.
    if len(split_groups) > 1:
        print(
            "\n[!] Different evaluation splits detected. "
            "Common-image comparison will be performed separately "
            "for each split."
        )

    common_data = {}
    common_indices_by_split = {}

    for split, names in split_groups.items():

        index_sets = [
            set(
                evaluations[name]["errors"]["valid_index"].tolist()
            )
            for name in names
        ]

        common_indices = set.intersection(*index_sets)

        common_indices_by_split[split] = common_indices

        for name in names:
            df = evaluations[name]["errors"]

            common_df = df[
                df["valid_index"].isin(common_indices)
            ].copy()

            common_data[name] = common_df

    return common_data, common_indices_by_split


# ============================================================
# RUN DISCOVERY
# ============================================================

def discover_runs(experiments_dir):
    """Discover immediate experiment directories containing log/checkpoints."""

    if not experiments_dir.is_dir():
        raise FileNotFoundError(
            f"Experiments directory does not exist: {experiments_dir}"
        )

    runs = []

    for directory in sorted(experiments_dir.iterdir()):
        if not directory.is_dir():
            continue

        has_log = (directory / "log.txt").is_file()
        has_checkpoints = any(directory.glob("epoch_*.pth.tar"))

        if has_log or has_checkpoints:
            runs.append(directory)

    return runs


def collect_data(experiments_dir, evaluations_dir):
    """Collect training and evaluation information for every run."""

    run_dirs = discover_runs(experiments_dir)

    if not run_dirs:
        raise RuntimeError(
            f"No training runs found in {experiments_dir}"
        )

    runs = {}
    evaluations = {}

    print_section("DISCOVERED TRAINING RUNS")

    for run_dir in run_dirs:

        name = run_dir.name

        history = load_run_history(run_dir)
        checkpoints = discover_checkpoints(run_dir)

        runs[name] = {
            **history,
            **checkpoints,
            "directory": run_dir,
        }

        best_text = (
            f"{history['best_val_loss']:.6f} "
            f"@ epoch {history['best_epoch']}"
            if history["best_epoch"] is not None
            else "not available"
        )

        print(
            f"{name}\n"
            f"  Train epochs: {history['num_train_epochs']}\n"
            f"  Val epochs:   {history['num_val_epochs']}\n"
            f"  Best val:     {best_text}\n"
            f"  Checkpoints:  {checkpoints['num_checkpoints']}\n"
            f"  Best model:   {checkpoints['has_best_model']}"
        )

        eval_dir = evaluations_dir / name

        metrics = load_metrics_json(eval_dir)
        errors = load_per_image_errors(eval_dir)

        if not metrics and errors.empty:
            print("  [!] No evaluation artifacts found.")
            continue

        json_summary = get_json_metrics(metrics)

        metadata_split = json_summary["split"]

        # CSV errors are the source for per-image comparisons.
        # JSON provides the original evaluation summary.
        csv_summary = summarize_error_dataframe(errors)

        if errors.empty:
            print("  [!] per_image_errors.csv unavailable.")

        if metrics and not errors.empty:
            # Check that the number of samples agrees.
            json_n = json_summary["num_images"]

            if json_n is not None and json_n != len(errors):
                print(
                    f"  [!] Image count mismatch: metrics.json={json_n}, "
                    f"CSV={len(errors)}"
                )

        evaluations[name] = {
            "directory": eval_dir,
            "metrics": metrics,
            "json_summary": json_summary,
            "errors": errors,
            "split": metadata_split,
            "csv_summary": csv_summary,
        }

        print(
            f"  Evaluation: {len(errors)} per-image records | "
            f"split={metadata_split or 'unknown'}"
        )

    if not runs:
        raise RuntimeError("No usable training runs found.")

    common_data, common_indices = get_common_errors(evaluations)

    print_section("COMMON IMAGE COMPARISON")

    for split, indices in common_indices.items():
        print(f"Split: {split} | Common images: {len(indices)}")

    # Build one row per training run.
    rows = []

    for name, run in runs.items():

        evaluation = evaluations.get(name)

        row = {
            "experiment": name,
            "best_epoch": run["best_epoch"],
            "best_val_loss": run["best_val_loss"],
            "final_val_loss": run["final_val_loss"],
            "num_train_epochs": run["num_train_epochs"],
            "num_val_epochs": run["num_val_epochs"],
            "num_checkpoints": run["num_checkpoints"],
            "checkpoint_epochs": ",".join(
                str(x) for x in run["epochs"]
            ),
            "has_best_model": run["has_best_model"],
            "evaluation_available": evaluation is not None,
            "split": None,
            "num_images": np.nan,
        }

        if evaluation is not None:

            json_summary = evaluation["json_summary"]
            csv_summary = evaluation["csv_summary"]

            row["split"] = evaluation["split"]
            row["num_images"] = (
                len(evaluation["errors"])
                if not evaluation["errors"].empty
                else json_summary["num_images"]
            )

            # Full evaluation metrics from metrics.json.
            for key in (
                "position_mean_m",
                "position_median_m",
                "position_rmse_m",
                "position_std_m",
                "position_p95_m",
                "orientation_mean_deg",
                "orientation_median_deg",
                "orientation_std_deg",
                "orientation_p95_deg",
            ):
                row[key] = json_summary[key]

            # Common-index metrics, when available.
            common_df = common_data.get(name, pd.DataFrame())

            common_summary = summarize_error_dataframe(
                common_df,
                prefix="common_",
            )

            row.update(common_summary)

            row["num_common_images"] = (
                len(common_df) if not common_df.empty else 0
            )

            # Compare CSV-derived full metrics with JSON.
            # This is a consistency check, not a new pose calculation.
            if not evaluation["errors"].empty:
                row["csv_position_mean_m"] = csv_summary.get(
                    "position_mean_m", np.nan
                )
                row["csv_orientation_mean_deg"] = csv_summary.get(
                    "orientation_mean_deg", np.nan
                )

        rows.append(row)

    summary = pd.DataFrame(rows)

    return runs, evaluations, summary, common_data, common_indices


# ============================================================
# PLOT 01 — TRAINING AND VALIDATION LOSS
# ============================================================

def plot_training_loss(runs, output):

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, run in runs.items():

        if len(run["train_epochs"]):
            ax.plot(
                run["train_epochs"],
                run["train_loss"],
                linewidth=1.2,
                alpha=0.65,
                linestyle="--",
                label=f"{name} train",
            )

        if len(run["val_epochs"]):
            ax.plot(
                run["val_epochs"],
                run["val_loss"],
                linewidth=1.8,
                label=f"{name} val",
            )

            if run["best_epoch"] is not None:
                ax.scatter(
                    run["best_epoch"],
                    run["best_val_loss"],
                    s=35,
                    zorder=5,
                )

    ax.set_title("MapNet — Training and validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# PLOT 02 — CHECKPOINT TIMELINE
# ============================================================

def plot_checkpoint_timeline(runs, output):

    names = list(runs.keys())

    fig, ax = plt.subplots(
        figsize=(12, max(4, 0.65 * len(names) + 2))
    )

    for y, name in enumerate(names):

        run = runs[name]
        checkpoints = run["epochs"]

        if checkpoints:
            ax.scatter(
                checkpoints,
                [y] * len(checkpoints),
                marker="s",
                s=45,
                label="Periodic checkpoint" if y == 0 else None,
            )

        if run["best_epoch"] is not None:
            ax.scatter(
                [run["best_epoch"]],
                [y],
                marker="*",
                s=150,
                label="Best validation epoch" if y == 0 else None,
                zorder=5,
            )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Epoch")
    ax.set_title("MapNet — Checkpoint timeline")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# PLOT 03 — POSITION ERRORS
# ============================================================

def plot_position_comparison(summary, output):

    column = "common_position_mean_m"

    # Prefer common-image metrics when available.
    data = summary.copy()

    data["plot_position_mean"] = data[column]

    missing = data["plot_position_mean"].isna()

    data.loc[missing, "plot_position_mean"] = data.loc[
        missing, "position_mean_m"
    ]

    data = data.dropna(subset=["plot_position_mean"])

    if data.empty:
        print("[!] Position comparison omitted: no metrics.")
        return

    data = data.sort_values("plot_position_mean")

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(data))
    width = 0.2

    series = [
        ("plot_position_mean", "Mean"),
        ("position_median_m", "Median"),
        ("position_rmse_m", "RMSE"),
        ("position_p95_m", "P95"),
    ]

    for i, (column_name, label) in enumerate(series):

        if column_name not in data.columns:
            continue

        values = data[column_name].to_numpy(dtype=float)

        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(data["experiment"], rotation=25, ha="right")
    ax.set_ylabel("Position error (m)")
    ax.set_title("MapNet — Position error comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# PLOT 04 — ORIENTATION ERRORS
# ============================================================

def plot_orientation_comparison(summary, output):

    data = summary.copy()

    data["plot_orientation_mean"] = data[
        "common_orientation_mean_deg"
    ]

    missing = data["plot_orientation_mean"].isna()

    data.loc[missing, "plot_orientation_mean"] = data.loc[
        missing, "orientation_mean_deg"
    ]

    data = data.dropna(subset=["plot_orientation_mean"])

    if data.empty:
        print("[!] Orientation comparison omitted: no metrics.")
        return

    data = data.sort_values("plot_orientation_mean")

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(data))
    width = 0.25

    series = [
        ("plot_orientation_mean", "Mean"),
        ("orientation_median_deg", "Median"),
        ("orientation_p95_deg", "P95"),
    ]

    for i, (column_name, label) in enumerate(series):

        values = data[column_name].to_numpy(dtype=float)

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(data["experiment"], rotation=25, ha="right")
    ax.set_ylabel("Orientation error (degrees)")
    ax.set_title("MapNet — Orientation error comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# PLOT 05 — PER-IMAGE ERROR DISTRIBUTIONS
# ============================================================

def plot_error_distributions(evaluations, common_data, output):

    position_data = []
    orientation_data = []
    labels = []

    for name, evaluation in evaluations.items():

        # Use common indices if available.
        df = common_data.get(name)

        if df is None or df.empty:
            df = evaluation["errors"]

        if df.empty:
            continue

        labels.append(name)
        position_data.append(
            df["translation_error_m"].to_numpy(dtype=float)
        )
        orientation_data.append(
            df["orientation_error_deg"].to_numpy(dtype=float)
        )

    if not labels:
        print("[!] Error distributions omitted: no per-image data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].boxplot(
        position_data,
        tick_labels=labels,
        showfliers=True,
    )

    axes[0].set_title("Position error distribution")
    axes[0].set_ylabel("Error (m)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(
        orientation_data,
        tick_labels=labels,
        showfliers=True,
    )

    axes[1].set_title("Orientation error distribution")
    axes[1].set_ylabel("Error (degrees)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "MapNet — Per-image pose error distributions"
    )

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


# ============================================================
# CSV AND REPORT
# ============================================================

def save_summary(summary, output):

    summary.to_csv(
        output,
        index=False,
        float_format="%.8f",
    )


def format_number(value, decimals=4):

    value = safe_float(value)

    if not np.isfinite(value):
        return "—"

    return f"{value:.{decimals}f}"


def save_report(
    summary,
    runs,
    evaluations,
    common_indices,
    output,
    dataset,
    scene,
):

    lines = [
        f"# MapNet training analysis — {dataset} {scene}",
        "",
        f"- Training runs discovered: **{len(runs)}**",
        f"- Runs with evaluation artifacts: **{len(evaluations)}**",
        "",
        "## Evaluation comparison",
        "",
        "| Run | Split | Images | Common images | "
        "Position mean (m) | Orientation mean (°) | "
        "Best val_loss | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():

        lines.append(
            f"| {row['experiment']} "
            f"| {row.get('split') or '—'} "
            f"| {format_number(row.get('num_images'), 0)} "
            f"| {format_number(row.get('num_common_images'), 0)} "
            f"| {format_number(row.get('position_mean_m'))} "
            f"| {format_number(row.get('orientation_mean_deg'))} "
            f"| {format_number(row.get('best_val_loss'), 6)} "
            f"| {format_number(row.get('best_epoch'), 0)} |"
        )

    lines += [
        "",
        "## Common-image comparison",
        "",
    ]

    if common_indices:
        for split, indices in common_indices.items():
            lines.append(
                f"- Split `{split}`: **{len(indices)}** common images."
            )
    else:
        lines.append(
            "- No common-image comparison was possible."
        )

    lines += [
        "",
        "## Best evaluated runs",
        "",
    ]

    for column, label, unit in [
        ("common_position_mean_m", "Position mean", "m"),
        ("common_orientation_mean_deg", "Orientation mean", "°"),
    ]:

        valid = summary.dropna(subset=[column])

        if not valid.empty:
            best = valid.loc[valid[column].idxmin()]

            lines.append(
                f"- **{label}:** {best['experiment']} "
                f"({best[column]:.4f} {unit}, "
                f"{int(best['num_common_images'])} common images)"
            )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- `val_loss` and pose errors are different metrics.",
        "- Pose errors are read from the evaluation artifacts.",
        "- Common-image statistics use the intersection of `valid_index`.",
        "- Runs with different splits are never mixed in a common comparison.",
        "- If common-image metrics are unavailable, the original "
        "evaluation metrics are retained.",
        "- Checkpoints are discovered by filename; their weights "
        "are never loaded.",
        "- The best validation epoch is identified from `log.txt`.",
        "",
    ]

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze MapNet runs across dataset versions."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name, e.g. Bebop.",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=DEFAULT_SCENE,
        help="Dataset version/scene, e.g. v2_3.",
    )

    parser.add_argument(
        "--experiments",
        type=Path,
        default=None,
        help="Optional explicit experiments directory.",
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Optional explicit evaluations directory.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory.",
    )

    args = parser.parse_args()

    dataset_folder = (
        "7scenes" if args.dataset == "7Scenes" else args.dataset
    )

    experiments_dir = (
        args.experiments.expanduser().resolve()
        if args.experiments
        else PROJECT_ROOT / "experiments" / args.dataset / args.scene
    )

    evaluations_dir = (
        args.eval_dir.expanduser().resolve()
        if args.eval_dir
        else PROJECT_ROOT / "eval_results" / dataset_folder / args.scene
    )

    output_dir = (
        args.out.expanduser().resolve()
        if args.out
        else evaluations_dir / "training_analysis"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("MAPNET TRAINING ANALYSIS")

    print(f"Project root:     {PROJECT_ROOT}")
    print(f"Dataset:          {args.dataset}")
    print(f"Scene/version:    {args.scene}")
    print(f"Experiments:      {experiments_dir}")
    print(f"Evaluations:      {evaluations_dir}")
    print(f"Output:           {output_dir}")

    runs, evaluations, summary, common_data, common_indices = collect_data(
        experiments_dir,
        evaluations_dir,
    )

    print_section("GENERATING PLOTS")

    plot_training_loss(
        runs,
        output_dir / "01_training_loss.png",
    )

    plot_checkpoint_timeline(
        runs,
        output_dir / "02_checkpoint_timeline.png",
    )

    plot_position_comparison(
        summary,
        output_dir / "03_position_error_comparison.png",
    )

    plot_orientation_comparison(
        summary,
        output_dir / "04_orientation_error_comparison.png",
    )

    plot_error_distributions(
        evaluations,
        common_data,
        output_dir / "05_pose_error_distributions.png",
    )

    save_summary(
        summary,
        output_dir / "summary.csv",
    )

    save_report(
        summary,
        runs,
        evaluations,
        common_indices,
        output_dir / "report.md",
        args.dataset,
        args.scene,
    )

    print_section("ANALYSIS COMPLETED")

    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
