
#!/usr/bin/env python3
"""Shared evaluation metrics for PoseNet / MapNet."""

import numpy as np

from common.pose_utils import quaternion_angular_error


def compute_pose_errors(pred_poses, targ_poses):
    """
    Compute per-pose translation and orientation errors.

    Both inputs must contain [x, y, z, qx, qy, qz, qw].
    Translation is assumed to be expressed in meters.
    Quaternions are assumed to be already converted and normalized.
    """
    pred_poses = np.asarray(pred_poses, dtype=float)
    targ_poses = np.asarray(targ_poses, dtype=float)

    if pred_poses.shape != targ_poses.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{pred_poses.shape} vs {targ_poses.shape}"
        )

    if pred_poses.ndim != 2 or pred_poses.shape[1] != 7:
        raise ValueError(
            "Expected pose arrays with shape (N, 7), "
            f"received {pred_poses.shape}"
        )

    if len(pred_poses) == 0:
        raise ValueError("Cannot evaluate an empty set of poses.")

    if not np.isfinite(pred_poses).all():
        raise ValueError("Predicted poses contain NaN or infinite values.")

    if not np.isfinite(targ_poses).all():
        raise ValueError("Target poses contain NaN or infinite values.")

    translation_errors = np.linalg.norm(
        pred_poses[:, :3] - targ_poses[:, :3],
        axis=1
    )

    orientation_errors = np.asarray([
        quaternion_angular_error(pred_q, target_q)
        for pred_q, target_q in zip(
            pred_poses[:, 3:],
            targ_poses[:, 3:]
        )
    ], dtype=float)

    return translation_errors, orientation_errors


def summarize_errors(translation_errors, orientation_errors):
    """Generate JSON-serializable summary statistics."""
    translation_errors = np.asarray(translation_errors, dtype=float)
    orientation_errors = np.asarray(orientation_errors, dtype=float)

    if len(translation_errors) == 0:
        raise ValueError("Translation error array is empty.")

    if len(translation_errors) != len(orientation_errors):
        raise ValueError(
            "Translation and orientation error counts do not match."
        )

    if not (
        np.isfinite(translation_errors).all()
        and np.isfinite(orientation_errors).all()
    ):
        raise ValueError("Error arrays contain NaN or infinite values.")

    return {
        "num_images": int(len(translation_errors)),
        "position": {
            "mean_m": float(np.mean(translation_errors)),
            "median_m": float(np.median(translation_errors)),
            "rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
            "std_m": float(np.std(translation_errors)),
            "p95_m": float(np.percentile(translation_errors, 95)),
        },
        "orientation": {
            "mean_deg": float(np.mean(orientation_errors)),
            "median_deg": float(np.median(orientation_errors)),
            "std_deg": float(np.std(orientation_errors)),
            "p95_deg": float(np.percentile(orientation_errors, 95)),
        },
    }
