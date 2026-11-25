#!/usr/bin/env python3
"""
Analyze MapNet training runs and their evaluations.

Normal analysis:
    - Reads training logs, metrics.json, per_image_errors.csv and .pkl.
    - Does NOT load checkpoints or run inference.
    - Excludes bebop_mapnet_queue_smoke from performance analysis.
    - Uses common validation images for fair comparisons.

Optional historical checkpoint evaluation:
    --evaluate-checkpoints --config-file PATH

Outputs:
    01_training_validation_loss.png
    02_checkpoint_timeline.png
    03_position_error_comparison.png
    04_orientation_error_comparison.png
    05_pose_error_boxplots.png
    06_position_x_real_vs_predicted.png
    07_position_y_real_vs_predicted.png
    08_position_z_real_vs_predicted.png
    09_position_3d_real_vs_predicted.png
    10_position_error_vs_epoch.png
    11_orientation_error_vs_epoch.png
    12_trajectory_comparison_3d.png
    summary.csv
    checkpoint_metrics.csv
    report.md

Checkpoint evaluations are saved under:
    eval_results/<dataset>/<scene>/<run>/checkpoint_evaluations/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = "Bebop"
DEFAULT_SCENE = "v2_3"

EXCLUDED_RUNS = {"bebop_mapnet_queue_smoke"}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_split(value):
    """Normalize split labels without merging validation and test."""
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ("val", "validation"):
        return "val"

    if value in ("train", "training"):
        return "train"

    if value == "test":
        return "test"

    return value


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def save_figure(fig, output):
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output.name}")


# ============================================================
# TRAINING LOG PARSER
# ============================================================

def extract_epoch_and_loss(line, split):
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
        patterns = [
            r"\bval[_ ]?loss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bloss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]
    else:
        patterns = [
            r"\btrain[_ ]?loss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"\bloss\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]

    for pattern in patterns:
        matches = re.findall(pattern, line, re.IGNORECASE)
        if matches:
            loss = safe_float(matches[-1])
            if np.isfinite(loss):
                return epoch, loss

    return None


def read_training_log(log_path):
    train_history = {}
    val_history = {}

    if not log_path.is_file():
        return train_history, val_history

    with log_path.open("r", errors="replace") as file:
        for line in file:
            result = extract_epoch_and_loss(line, "train")
            if result:
                epoch, loss = result
                train_history[epoch] = loss

            result = extract_epoch_and_loss(line, "val")
            if result:
                epoch, loss = result
                val_history[epoch] = loss

    return train_history, val_history


def load_run_history(run_dir):
    train_history, val_history = read_training_log(run_dir / "log.txt")

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
        "num_train_epochs": len(train_epochs),
        "num_val_epochs": len(val_epochs),
        "log_path": str(run_dir / "log.txt"),
    }


# ============================================================
# CHECKPOINT DISCOVERY AND OPTIONAL EVALUATION
# ============================================================

def discover_checkpoints(run_dir):
    checkpoints = []

    for path in run_dir.glob("epoch_*.pth.tar"):
        match = re.search(r"epoch_(\d+)", path.name)
        if match:
            checkpoints.append({
                "epoch": int(match.group(1)),
                "path": path,
                "kind": "periodic",
            })

    checkpoints.sort(key=lambda item: item["epoch"])

    best_path = run_dir / "best_model.pth.tar"
    has_best_model = best_path.is_file()

    return {
        "checkpoints": checkpoints,
        "epochs": [item["epoch"] for item in checkpoints],
        "num_checkpoints": len(checkpoints),
        "has_best_model": has_best_model,
    }


def discover_config_file(run_dir, explicit_config=None):
    """Find the configuration file required by eval.py."""

    if explicit_config is not None:
        path = explicit_config.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    candidates = []
    for pattern in ("*.ini", "*.cfg", "*.conf"):
        candidates.extend(run_dir.glob(pattern))

    candidates = sorted(set(p.resolve() for p in candidates))

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No .ini/.cfg/.conf file found in {run_dir}. "
            "Pass --config-file explicitly."
        )

    raise RuntimeError(
        f"Multiple config files found in {run_dir}: "
        f"{[str(p) for p in candidates]}. "
        "Pass --config-file explicitly."
    )


def evaluate_one_checkpoint(
    checkpoint_path,
    output_dir,
    dataset,
    scene,
    config_file,
    eval_script,
    device,
):
    """Run the repository's eval.py on one checkpoint."""

    metrics_path = output_dir / "metrics.json"
    errors_path = output_dir / "per_image_errors.csv"

    if metrics_path.is_file() and errors_path.is_file():
        print(f"  [skip] Existing evaluation: {output_dir.name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(eval_script),
        "--dataset", dataset,
        "--scene", scene,
        "--model", "mapnet",
        "--weights", str(checkpoint_path.resolve()),
        "--config_file", str(config_file),
        "--device", str(device),
        "--val",
        "--output_dir", str(output_dir.resolve()),
    ]

    print("\nRunning checkpoint evaluation:")
    print(" ", checkpoint_path.name)

    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def evaluate_run_checkpoints(
    run_name,
    run,
    eval_run_dir,
    dataset,
    scene,
    config_file,
    eval_script,
    device,
    checkpoint_stride,
    include_best,
):
    """Evaluate periodic checkpoints and optionally best_model."""

    checkpoint_root = eval_run_dir / "checkpoint_evaluations"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    config = discover_config_file(run["directory"], config_file)

    checkpoints = run["checkpoints"]

    if checkpoint_stride > 1:
        checkpoints = [
            item for item in checkpoints
            if item["epoch"] % checkpoint_stride == 0
        ]

    for item in checkpoints:
        epoch = item["epoch"]
        output_dir = checkpoint_root / f"epoch_{epoch:04d}"

        evaluate_one_checkpoint(
            item["path"],
            output_dir,
            dataset,
            scene,
            config,
            eval_script,
            device,
        )

    if include_best and run["has_best_model"]:
        output_dir = checkpoint_root / "best_model"

        evaluate_one_checkpoint(
            run["directory"] / "best_model.pth.tar",
            output_dir,
            dataset,
            scene,
            config,
            eval_script,
            device,
        )


def load_checkpoint_evaluations(eval_run_dir):
    """Read metrics produced for historical checkpoints."""

    root = eval_run_dir / "checkpoint_evaluations"
    rows = []

    if not root.is_dir():
        return rows

    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue

        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            continue

        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        position = metrics.get("position", {})
        orientation = metrics.get("orientation", {})
        metadata = metrics.get("metadata", {})

        match = re.fullmatch(r"epoch_(\d+)", directory.name)
        epoch = int(match.group(1)) if match else None

        rows.append({
            "experiment": eval_run_dir.name,
            "checkpoint": directory.name,
            "epoch": epoch,
            "is_best_model": directory.name == "best_model",
            "split": normalize_split(metadata.get("split")),
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
            "metrics_path": str(metrics_path),
        })

    return rows


# ============================================================
# EVALUATION ARTIFACTS
# ============================================================

def load_metrics_json(eval_dir):
    path = eval_dir / "metrics.json"

    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"  [!] Could not read {path}: {error}")
        return {}


def load_per_image_errors(eval_dir):
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

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df["valid_index"] = df["valid_index"].astype(int)
    df = df.drop_duplicates(subset="valid_index", keep="last")

    return df.sort_values("valid_index").reset_index(drop=True)


def load_pose_pickle(eval_dir):
    """Load saved predictions and targets from eval.py."""

    candidates = sorted(eval_dir.glob("*.pkl"))

    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            f"  [!] Multiple .pkl files in {eval_dir}; "
            f"using {candidates[0].name}"
        )

    import pickle

    try:
        with candidates[0].open("rb") as file:
            data = pickle.load(file)
    except Exception as error:
        print(f"  [!] Could not load {candidates[0]}: {error}")
        return None

    required = {"targ_poses", "pred_poses", "valid_indices"}

    if not isinstance(data, dict) or not required.issubset(data):
        print(f"  [!] Unexpected .pkl structure: {candidates[0]}")
        return None

    target = np.asarray(data["targ_poses"], dtype=float)
    prediction = np.asarray(data["pred_poses"], dtype=float)
    indices = np.asarray(data["valid_indices"], dtype=int)

    if (
        target.ndim != 2
        or prediction.ndim != 2
        or target.shape[1] < 3
        or prediction.shape[1] < 3
        or len(target) != len(prediction)
        or len(target) != len(indices)
    ):
        print(f"  [!] Invalid pose array dimensions in {candidates[0]}")
        return None

    order = np.argsort(indices)

    return {
        "target": target[order],
        "prediction": prediction[order],
        "indices": indices[order],
        "path": candidates[0],
    }


def get_json_metrics(metrics):
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
# SUMMARY STATISTICS FROM EXISTING PER-IMAGE ERRORS
# ============================================================

def summarize_error_dataframe(df, prefix=""):
    if df.empty:
        return {}

    t = finite_values(df["translation_error_m"])
    q = finite_values(df["orientation_error_deg"])

    if not len(t) or not len(q):
        return {}

    result = {
        "num_images": int(min(len(t), len(q))),
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
        return {f"{prefix}{key}": value for key, value in result.items()}

    return result


def get_common_errors(evaluations):
    """
    Intersect valid_index independently within each split.

    Different splits are never merged.
    """

    groups = {}

    for name, data in evaluations.items():
        if data["errors"].empty:
            continue

        split = data["split"] or "unknown"
        groups.setdefault(split, []).append(name)

    common_data = {}
    common_indices = {}

    for split, names in groups.items():
        if len(names) < 2:
            continue

        sets = [
            set(evaluations[name]["errors"]["valid_index"].tolist())
            for name in names
        ]

        indices = set.intersection(*sets)
        common_indices[split] = indices

        for name in names:
            df = evaluations[name]["errors"]
            common_data[name] = df[
                df["valid_index"].isin(indices)
            ].copy()

    return common_data, common_indices


# ============================================================
# RUN DISCOVERY AND DATA COLLECTION
# ============================================================

def discover_runs(experiments_dir):
    if not experiments_dir.is_dir():
        raise FileNotFoundError(
            f"Experiments directory does not exist: {experiments_dir}"
        )

    runs = []

    for directory in sorted(experiments_dir.iterdir()):
        if not directory.is_dir():
            continue

        if directory.name in EXCLUDED_RUNS:
            print(f"[excluded] {directory.name}")
            continue

        has_log = (directory / "log.txt").is_file()
        has_checkpoints = any(directory.glob("epoch_*.pth.tar"))

        if has_log or has_checkpoints:
            runs.append(directory)

    return runs


def collect_data(experiments_dir, evaluations_dir):
    run_dirs = discover_runs(experiments_dir)

    if not run_dirs:
        raise RuntimeError(f"No training runs found in {experiments_dir}")

    runs = {}
    evaluations = {}

    print_section("DISCOVERED TRAINING RUNS")

    for run_dir in run_dirs:
        name = run_dir.name
        history = load_run_history(run_dir)
        checkpoint_info = discover_checkpoints(run_dir)

        runs[name] = {
            **history,
            **checkpoint_info,
            "directory": run_dir,
        }

        best_text = (
            f"{history['best_val_loss']:.6f} @ epoch {history['best_epoch']}"
            if history["best_epoch"] is not None
            else "not available"
        )

        print(
            f"{name}\n"
            f"  Train epochs: {history['num_train_epochs']}\n"
            f"  Val epochs:   {history['num_val_epochs']}\n"
            f"  Best val:     {best_text}\n"
            f"  Checkpoints:  {checkpoint_info['num_checkpoints']}\n"
            f"  Best model:   {checkpoint_info['has_best_model']}"
        )

        eval_dir = evaluations_dir / name
        metrics = load_metrics_json(eval_dir)
        errors = load_per_image_errors(eval_dir)
        poses = load_pose_pickle(eval_dir)

        if not metrics and errors.empty and poses is None:
            print("  [!] No evaluation artifacts found.")
            continue

        json_summary = get_json_metrics(metrics)

        if metrics and not errors.empty:
            json_n = json_summary["num_images"]
            if json_n is not None and json_n != len(errors):
                print(
                    f"  [!] Image count mismatch: "
                    f"metrics.json={json_n}, CSV={len(errors)}"
                )

        evaluations[name] = {
            "directory": eval_dir,
            "metrics": metrics,
            "json_summary": json_summary,
            "errors": errors,
            "poses": poses,
            "split": json_summary["split"],
        }

        print(
            f"  Evaluation: {len(errors)} per-image records | "
            f"split={json_summary['split'] or 'unknown'} | "
            f"poses={'yes' if poses is not None else 'no'}"
        )

    common_data, common_indices = get_common_errors(evaluations)

    print_section("COMMON IMAGE COMPARISON")
    if common_indices:
        for split, indices in common_indices.items():
            print(f"Split: {split} | Common images: {len(indices)}")
    else:
        print("No common-image groups with at least two comparable runs.")

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
            "checkpoint_epochs": ",".join(map(str, run["epochs"])),
            "has_best_model": run["has_best_model"],
            "evaluation_available": evaluation is not None,
            "split": None,
            "num_images": np.nan,
            "num_common_images": 0,
        }

        metric_names = [
            "position_mean_m",
            "position_median_m",
            "position_rmse_m",
            "position_std_m",
            "position_p95_m",
            "orientation_mean_deg",
            "orientation_median_deg",
            "orientation_std_deg",
            "orientation_p95_deg",
        ]

        for metric in metric_names:
            row[metric] = np.nan
            row[f"common_{metric}"] = np.nan

        if evaluation is not None:
            row["split"] = evaluation["split"]
            errors = evaluation["errors"]

            row["num_images"] = (
                len(errors)
                if not errors.empty
                else evaluation["json_summary"]["num_images"]
            )

            common_df = common_data.get(name, pd.DataFrame())

            # If at least two runs share this split, use common-image metrics.
            # Otherwise, use the complete evaluation CSV.
            source_df = common_df if not common_df.empty else errors

            source_summary = summarize_error_dataframe(source_df)

            for metric in metric_names:
                row[metric] = source_summary.get(metric, np.nan)

            row["num_common_images"] = (
                len(common_df) if not common_df.empty else 0
            )
            row["num_images_used_for_metrics"] = len(source_df)

            # Keep original JSON values separately for audit/consistency.
            json_summary = evaluation["json_summary"]
            for metric in metric_names:
                row[f"json_{metric}"] = json_summary.get(metric, np.nan)

        rows.append(row)

    return (
        runs,
        evaluations,
        pd.DataFrame(rows),
        common_data,
        common_indices,
    )


# ============================================================
# PLOT 01 — TRAINING AND VALIDATION LOSS
# ============================================================

def plot_training_loss(runs, output):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for name, run in runs.items():
        if len(run["train_epochs"]):
            axes[0].plot(
                run["train_epochs"],
                run["train_loss"],
                linewidth=1.2,
                label=name,
            )

        if len(run["val_epochs"]):
            axes[1].plot(
                run["val_epochs"],
                run["val_loss"],
                linewidth=1.4,
                label=name,
            )

            if run["best_epoch"] is not None:
                axes[1].scatter(
                    [run["best_epoch"]],
                    [run["best_val_loss"]],
                    marker="*",
                    s=75,
                    zorder=5,
                )

    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    save_figure(fig, output)


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

        if run["epochs"]:
            ax.scatter(
                run["epochs"],
                [y] * len(run["epochs"]),
                marker="s",
                s=35,
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

    save_figure(fig, output)


# ============================================================
# PLOT 03 AND 04 — SIMPLIFIED COMPARISONS
# ============================================================

def plot_single_metric_comparison(
    summary,
    metric,
    label,
    ylabel,
    output,
):
    data = summary.dropna(subset=[metric]).copy()

    if data.empty:
        print(f"[!] {label} comparison omitted: no metrics.")
        return

    data = data.sort_values(metric)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(data["experiment"], data[metric])

    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, output)


# ============================================================
# PLOT 05 — ERROR BOXPLOTS
# ============================================================

def plot_error_distributions(evaluations, common_data, output):
    position_data = []
    orientation_data = []
    labels = []

    for name, evaluation in evaluations.items():
        df = common_data.get(name)

        if df is None or df.empty:
            df = evaluation["errors"]

        if df.empty:
            continue

        labels.append(name)
        position_data.append(df["translation_error_m"].to_numpy(float))
        orientation_data.append(df["orientation_error_deg"].to_numpy(float))

    if not labels:
        print("[!] Error boxplots omitted: no per-image data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].boxplot(
        position_data,
        tick_labels=labels,
        showfliers=True,
    )
    axes[0].set_title("Position error")
    axes[0].set_ylabel("Error (m)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].boxplot(
        orientation_data,
        tick_labels=labels,
        showfliers=True,
    )
    axes[1].set_title("Orientation error")
    axes[1].set_ylabel("Error (degrees)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("MapNet — Per-image pose error distributions")
    save_figure(fig, output)


# ============================================================
# PLOTS 06–09 — REAL VS PREDICTED POSITION
# ============================================================

def get_pose_evaluations(evaluations):
    return {
        name: data["poses"]
        for name, data in evaluations.items()
        if data["poses"] is not None
    }


def plot_axis_real_vs_predicted(pose_data, axis, axis_name, output):
    fig, ax = plt.subplots(figsize=(8, 8))
    all_values = []

    for name, poses in pose_data.items():
        target = poses["target"][:, axis]
        prediction = poses["prediction"][:, axis]

        ax.scatter(
            target,
            prediction,
            s=10,
            alpha=0.45,
            label=name,
        )

        all_values.extend(target.tolist())
        all_values.extend(prediction.tolist())

    if not all_values:
        plt.close(fig)
        return

    low = float(np.nanmin(all_values))
    high = float(np.nanmax(all_values))
    margin = max((high - low) * 0.05, 0.05)

    ax.plot(
        [low - margin, high + margin],
        [low - margin, high + margin],
        linestyle="--",
        linewidth=1,
        label="Ideal prediction",
    )

    ax.set_xlim(low - margin, high + margin)
    ax.set_ylim(low - margin, high + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Real {axis_name} (m)")
    ax.set_ylabel(f"Predicted {axis_name} (m)")
    ax.set_title(f"Real vs predicted position — {axis_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    save_figure(fig, output)


def plot_3d_real_vs_predicted(pose_data, output):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, poses in pose_data.items():
        target = poses["target"][:, :3]
        prediction = poses["prediction"][:, :3]

        ax.scatter(
            target[:, 0], target[:, 1], target[:, 2],
            s=9, alpha=0.45, label=f"{name} real",
        )
        ax.scatter(
            prediction[:, 0], prediction[:, 1], prediction[:, 2],
            s=9, alpha=0.45, marker="^", label=f"{name} predicted",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D real vs predicted positions")
    ax.legend(fontsize=7)

    save_figure(fig, output)


# ============================================================
# PLOT 12 — TRAJECTORY COMPARISON
# ============================================================

def plot_trajectory_comparison(pose_data, output):
    if not pose_data:
        print("[!] Trajectory plot omitted: no pose pickle files.")
        return

    n = len(pose_data)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, poses in pose_data.items():
        target = poses["target"][:, :3]
        prediction = poses["prediction"][:, :3]

        # valid_indices are sorted when the pickle is loaded.
        ax.plot(
            target[:, 0], target[:, 1], target[:, 2],
            linewidth=1.5,
            label=f"{name} real",
        )
        ax.plot(
            prediction[:, 0], prediction[:, 1], prediction[:, 2],
            linewidth=1.2,
            linestyle="--",
            label=f"{name} predicted",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("MapNet — Ordered trajectory comparison")
    ax.legend(fontsize=7)

    save_figure(fig, output)


# ============================================================
# PLOTS 10–11 — GEOMETRIC ERROR VS CHECKPOINT EPOCH
# ============================================================

def plot_checkpoint_metric(
    checkpoint_df,
    metric,
    ylabel,
    title,
    output,
):
    if checkpoint_df.empty:
        print(f"[!] {title} omitted: no checkpoint evaluations.")
        return

    data = checkpoint_df.dropna(subset=["epoch", metric]).copy()

    if data.empty:
        print(f"[!] {title} omitted: no periodic checkpoint metrics.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    for name, group in data.groupby("experiment"):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group[metric],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=name,
        )

    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    save_figure(fig, output)


# ============================================================
# CSV AND REPORT
# ============================================================

def save_summary(summary, output):
    summary.to_csv(output, index=False, float_format="%.8f")
    print(f"  Saved: {output.name}")


def save_checkpoint_metrics(rows, output):
    df = pd.DataFrame(rows)

    if df.empty:
        # Still create a CSV with a useful schema.
        df = pd.DataFrame(columns=[
            "experiment", "checkpoint", "epoch", "is_best_model",
            "split", "num_images",
            "position_mean_m", "position_median_m",
            "position_rmse_m", "position_std_m", "position_p95_m",
            "orientation_mean_deg", "orientation_median_deg",
            "orientation_std_deg", "orientation_p95_deg",
        ])

    df.to_csv(output, index=False, float_format="%.8f")
    print(f"  Saved: {output.name}")

    return df


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
    checkpoint_df,
    output,
    dataset,
    scene,
):
    lines = [
        f"# MapNet training analysis — {dataset} {scene}",
        "",
        f"- Training runs included: **{len(runs)}**",
        f"- Runs with evaluation artifacts: **{len(evaluations)}**",
        f"- Excluded smoke tests: `{', '.join(sorted(EXCLUDED_RUNS))}`",
        "",
        "## Evaluation comparison",
        "",
        "| Run | Split | Images used | Common images | "
        "Position mean (m) | RMSE (m) | P95 (m) | "
        "Orientation mean (°) | P95 (°) | Best val_loss | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['experiment']} "
            f"| {row.get('split') or '—'} "
            f"| {format_number(row.get('num_images'), 0)} "
            f"| {format_number(row.get('num_common_images'), 0)} "
            f"| {format_number(row.get('position_mean_m'))} "
            f"| {format_number(row.get('position_rmse_m'))} "
            f"| {format_number(row.get('position_p95_m'))} "
            f"| {format_number(row.get('orientation_mean_deg'))} "
            f"| {format_number(row.get('orientation_p95_deg'))} "
            f"| {format_number(row.get('best_val_loss'), 6)} "
            f"| {format_number(row.get('best_epoch'), 0)} |"
        )

    lines += ["", "## Common-image comparison", ""]

    if common_indices:
        for split, indices in common_indices.items():
            lines.append(
                f"- Split `{split}`: **{len(indices)}** common images."
            )
    else:
        lines.append("- No multi-run common-image comparison was possible.")

    lines += ["", "## Best runs on comparable images", ""]

    for column, label, unit in [
        ("position_mean_m", "Position mean", "m"),
        ("position_rmse_m", "Position RMSE", "m"),
        ("position_p95_m", "Position P95", "m"),
        ("orientation_mean_deg", "Orientation mean", "°"),
        ("orientation_p95_deg", "Orientation P95", "°"),
    ]:
        valid = summary.dropna(subset=[column])
        if not valid.empty:
            best = valid.loc[valid[column].idxmin()]
            lines.append(
                f"- **{label}:** {best['experiment']} "
                f"({best[column]:.4f} {unit})"
            )

    lines += ["", "## Historical checkpoint evaluation", ""]

    if checkpoint_df.empty:
        lines.append(
            "No historical checkpoints have been evaluated yet. "
            "Run with `--evaluate-checkpoints` to generate these results."
        )
    else:
        lines.append(
            f"- Historical checkpoint evaluations available: "
            f"**{len(checkpoint_df)}**."
        )
        lines.append(
            "- Checkpoint error curves use the metrics generated by `eval.py`."
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Training and validation losses are optimization losses, "
        "not geometric pose errors.",
        "- Position and orientation errors are read from `eval.py` artifacts.",
        "- Common-image statistics are calculated from the intersection "
        "of `valid_index` within each split.",
        "- Validation and test splits are kept separate.",
        "- RMSE penalizes large errors more strongly than the mean.",
        "- P95 is the error value below which 95% of evaluated images fall.",
        "- The 3D position error is the Euclidean distance between "
        "predicted and target XYZ positions.",
        "- Trajectory plots connect poses in ascending `valid_index` order. "
        "They do not infer missing timestamps or sequence boundaries.",
        "- The smoke-test run is excluded from this analysis.",
        "",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze MapNet runs across dataset versions."
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--experiments", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)

    parser.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help="Run eval.py on periodic checkpoints and best_model.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Config file passed to eval.py. If omitted, "
             "a unique .ini/.cfg/.conf file is searched in each run.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=PROJECT_ROOT / "geomapnet" / "scripts" / "eval.py",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device argument forwarded to eval.py. Use -1 for CPU "
             "if supported by the local PyTorch setup.",
    )
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=1,
        help="Evaluate every Nth periodic checkpoint. Default: 1 (all).",
    )
    parser.add_argument(
        "--no-best-checkpoint",
        action="store_true",
        help="Do not evaluate best_model.pth.tar.",
    )

    args = parser.parse_args()

    if args.checkpoint_stride < 1:
        parser.error("--checkpoint-stride must be >= 1")

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
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Dataset:       {args.dataset}")
    print(f"Scene/version: {args.scene}")
    print(f"Experiments:   {experiments_dir}")
    print(f"Evaluations:   {evaluations_dir}")
    print(f"Output:        {output_dir}")
    print(f"Excluded:      {', '.join(sorted(EXCLUDED_RUNS))}")

    runs, evaluations, summary, common_data, common_indices = collect_data(
        experiments_dir,
        evaluations_dir,
    )

    # Optional historical checkpoint evaluation.
    if args.evaluate_checkpoints:
        print_section("HISTORICAL CHECKPOINT EVALUATION")

        eval_script = args.eval_script.expanduser().resolve()
        if not eval_script.is_file():
            raise FileNotFoundError(f"eval.py not found: {eval_script}")

        for run_name, run in runs.items():
            evaluate_run_checkpoints(
                run_name,
                run,
                evaluations_dir / run_name,
                args.dataset,
                args.scene,
                args.config_file,
                eval_script,
                args.device,
                args.checkpoint_stride,
                not args.no_best_checkpoint,
            )

    # Load historical checkpoint results whether newly generated or existing.
    checkpoint_rows = []
    for run_name in runs:
        checkpoint_rows.extend(
            load_checkpoint_evaluations(evaluations_dir / run_name)
        )

    checkpoint_df = pd.DataFrame(checkpoint_rows)

    print_section("GENERATING PLOTS")

    plot_training_loss(
        runs,
        output_dir / "01_training_validation_loss.png",
    )

    plot_checkpoint_timeline(
        runs,
        output_dir / "02_checkpoint_timeline.png",
    )

    plot_single_metric_comparison(
        summary,
        "position_mean_m",
        "MapNet — Mean position error",
        "Mean position error (m)",
        output_dir / "03_position_error_comparison.png",
    )

    plot_single_metric_comparison(
        summary,
        "orientation_mean_deg",
        "MapNet — Mean orientation error",
        "Mean orientation error (degrees)",
        output_dir / "04_orientation_error_comparison.png",
    )

    plot_error_distributions(
        evaluations,
        common_data,
        output_dir / "05_pose_error_boxplots.png",
    )

    pose_data = get_pose_evaluations(evaluations)

    if pose_data:
        plot_axis_real_vs_predicted(
            pose_data, 0, "X",
            output_dir / "06_position_x_real_vs_predicted.png",
        )
        plot_axis_real_vs_predicted(
            pose_data, 1, "Y",
            output_dir / "07_position_y_real_vs_predicted.png",
        )
        plot_axis_real_vs_predicted(
            pose_data, 2, "Z",
            output_dir / "08_position_z_real_vs_predicted.png",
        )
        plot_3d_real_vs_predicted(
            pose_data,
            output_dir / "09_position_3d_real_vs_predicted.png",
        )
        plot_trajectory_comparison(
            pose_data,
            output_dir / "12_trajectory_comparison_3d.png",
        )
    else:
        print("[!] Position and trajectory plots omitted: no .pkl files.")

    plot_checkpoint_metric(
        checkpoint_df,
        "position_mean_m",
        "Mean position error (m)",
        "Position error vs checkpoint epoch",
        output_dir / "10_position_error_vs_epoch.png",
    )

    plot_checkpoint_metric(
        checkpoint_df,
        "orientation_mean_deg",
        "Mean orientation error (degrees)",
        "Orientation error vs checkpoint epoch",
        output_dir / "11_orientation_error_vs_epoch.png",
    )

    print_section("SAVING TABLES AND REPORT")

    save_summary(summary, output_dir / "summary.csv")

    checkpoint_df = save_checkpoint_metrics(
        checkpoint_rows,
        output_dir / "checkpoint_metrics.csv",
    )
    print("  Saved: checkpoint_metrics.csv")

    save_report(
        summary,
        runs,
        evaluations,
        common_indices,
        checkpoint_df,
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
