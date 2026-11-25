#!/usr/bin/env python3
# view_mapnet_plot.py
"""
Open a saved MapNet evaluation PKL as an interactive 3D Matplotlib plot.

Reproduces the original NVIDIA MapNet visualization:
- Blue lines: prediction-to-ground-truth correspondences.
- Red points: predicted positions.
- Green points: ground-truth positions.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_results(pkl_path):
    """Load saved predictions and ground-truth poses."""
    pkl_path = Path(pkl_path).expanduser().resolve()

    if not pkl_path.is_file():
        raise FileNotFoundError(
            "Evaluation file not found: {}".format(pkl_path)
        )

    with pkl_path.open("rb") as f:
        results = pickle.load(f)

    required_keys = {"pred_poses", "targ_poses"}
    missing = required_keys.difference(results.keys())

    if missing:
        raise KeyError(
            "Missing keys in PKL file: {}".format(sorted(missing))
        )

    pred = np.asarray(results["pred_poses"])
    targ = np.asarray(results["targ_poses"])

    if pred.shape != targ.shape or pred.ndim != 2 or pred.shape[1] < 3:
        raise ValueError(
            "Invalid pose arrays: pred={}, targ={}".format(
                pred.shape, targ.shape
            )
        )

    print("Loaded:", pkl_path)
    print("Evaluated poses:", len(pred))

    return pred, targ


def plot_trajectory(pred, targ, max_points=1000):
    """Reproduce the original interactive 3D MapNet plot."""

    # Same downsampling principle used by eval.py.
    ss = max(1, int(len(pred) / max_points))

    pred_xyz = pred[::ss, :3]
    targ_xyz = targ[::ss, :3]

    x = np.vstack((pred_xyz[:, 0], targ_xyz[:, 0]))
    y = np.vstack((pred_xyz[:, 1], targ_xyz[:, 1]))
    z = np.vstack((pred_xyz[:, 2], targ_xyz[:, 2]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    # Blue correspondence lines.
    for xx, yy, zz in zip(x.T, y.T, z.T):
        ax.plot(xx, yy, zs=zz, c="b")

    # Predicted positions.
    ax.scatter(
        x[0, :], y[0, :], zs=z[0, :],
        c="r", depthshade=0
    )

    # Ground-truth positions.
    ax.scatter(
        x[1, :], y[1, :], zs=z[1, :],
        c="g", depthshade=0
    )

    # Original NVIDIA viewing angle.
    ax.view_init(azim=119, elev=13)

    ax.set_title("MapNet: predicted vs ground truth")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    print("Opening interactive Matplotlib window...")
    print("Close the window to return to the terminal.")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Open a saved MapNet evaluation as an interactive 3D plot."
    )

    parser.add_argument(
        "pkl",
        type=str,
        help="Path to the evaluation .pkl file."
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum approximate number of plotted positions (default: 1000)."
    )

    args = parser.parse_args()

    if args.max_points < 1:
        parser.error("--max-points must be greater than zero")

    pred, targ = load_results(args.pkl)
    plot_trajectory(pred, targ, args.max_points)


if __name__ == "__main__":
    main()
