
#!/usr/bin/env python3
"""
Evalúa los experimentos MapNet convirtiendo sus poses W2C a:

- Posición de cámara en coordenadas del mundo (metros).
- Orientación del cuerpo del dron en coordenadas FLU.

Calcula métricas tanto sobre todos los índices válidos de cada
experimento como sobre el conjunto común de los siete experimentos.

Convenciones:
- Entrada MapNet: W2C, quaternion scalar-first [qw, qx, qy, qz].
- Salida: posición C_w y orientación R_wb.
- R_BODY_FROM_CAMERA: transformación de coordenadas de cámara
  óptica a coordenadas del cuerpo FLU.
"""

import pickle
import csv
from pathlib import Path

import numpy as np
import transforms3d.quaternions as txq


BASE_DIR = Path(__file__).resolve().parents[2]

EXPERIMENTS = [
    "baseline",
    "500",
    "lr5e-5",
    "seed42",
    "skip20",
    "skip5",
    "wd1e-4",
]

OUTPUT_DIR = BASE_DIR / "eval_results" / "c2w_comparison"

# Cámara óptica [right, down, forward] -> cuerpo FLU [forward, left, up].
R_BODY_FROM_CAMERA = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
])


def w2c_to_c2w_body(pose):
    """
    Convierte una pose W2C de MapNet a posición de cámara
    y orientación del cuerpo del dron en el mundo.

    pose = [tx, ty, tz, qw, qx, qy, qz]

    Retorna:
        position_world: (3,), metros
        R_world_body: (3, 3)
    """
    t_cw = np.asarray(pose[:3], dtype=np.float64)
    q_cw = np.asarray(pose[3:], dtype=np.float64)

    q_norm = np.linalg.norm(q_cw)
    if q_norm < 1e-12:
        raise ValueError("Quaternion con norma cero.")

    q_cw /= q_norm

    # transforms3d usa quaternion scalar-first: [w, x, y, z].
    R_cw = txq.quat2mat(q_cw)

    # Invertir W2C -> C2W.
    R_wc = R_cw.T
    position_world = -R_wc @ t_cw

    # Orientación de cámara óptica -> orientación del cuerpo FLU.
    R_world_body = R_wc @ R_BODY_FROM_CAMERA.T

    return position_world, R_world_body


def angular_error_deg(R_pred, R_gt):
    """Error geodésico entre dos matrices de rotación."""
    R_error = R_pred.T @ R_gt

    cos_angle = (np.trace(R_error) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def load_experiment(name):
    path = (
        BASE_DIR
        / "eval_results"
        / name
        / "Bebop_bebop_mapnet.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as file:
        data = pickle.load(file)

    pred = np.asarray(data["pred_poses"], dtype=np.float64)
    targ = np.asarray(data["targ_poses"], dtype=np.float64)
    indices = np.asarray(data["valid_indices"], dtype=np.int64)

    if pred.shape != targ.shape:
        raise ValueError(f"{name}: pred_poses y targ_poses no coinciden.")

    if pred.ndim != 2 or pred.shape[1] != 7:
        raise ValueError(f"{name}: se esperaba un array N x 7.")

    if len(indices) != len(pred):
        raise ValueError(f"{name}: cantidad de índices incorrecta.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name}: hay índices duplicados.")

    return pred, targ, indices


def evaluate_poses(pred, targ, indices):
    rows = []

    for p, gt, index in zip(pred, targ, indices):
        pred_position, pred_rotation = w2c_to_c2w_body(p)
        gt_position, gt_rotation = w2c_to_c2w_body(gt)

        translation_error = np.linalg.norm(
            pred_position - gt_position
        )

        rotation_error = angular_error_deg(
            pred_rotation,
            gt_rotation
        )

        rows.append({
            "valid_index": int(index),
            "pred_x_m": pred_position[0],
            "pred_y_m": pred_position[1],
            "pred_z_m": pred_position[2],
            "gt_x_m": gt_position[0],
            "gt_y_m": gt_position[1],
            "gt_z_m": gt_position[2],
            "translation_error_m": translation_error,
            "orientation_error_deg": rotation_error,
        })

    return rows


def calculate_statistics(rows):
    translation = np.array([
        row["translation_error_m"] for row in rows
    ])

    orientation = np.array([
        row["orientation_error_deg"] for row in rows
    ])

    return {
        "n": len(rows),
        "position_mean_m": np.mean(translation),
        "position_median_m": np.median(translation),
        "position_rmse_m": np.sqrt(np.mean(translation ** 2)),
        "position_p95_m": np.percentile(translation, 95),
        "orientation_mean_deg": np.mean(orientation),
        "orientation_median_deg": np.median(orientation),
        "orientation_p95_deg": np.percentile(orientation, 95),
    }


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("EVALUACIÓN MAPNET: W2C -> C2W + ORIENTACIÓN BODY FLU")
    print("=" * 75)

    loaded = {}
    evaluated = {}

    # Cargar y evaluar cada experimento.
    for name in EXPERIMENTS:
        pred, targ, indices = load_experiment(name)

        loaded[name] = {
            "pred": pred,
            "targ": targ,
            "indices": indices,
        }

        evaluated[name] = evaluate_poses(pred, targ, indices)

        print(
            f"{name:12s}: {len(indices):3d} poses, "
            f"índices [{indices.min()}, {indices.max()}]"
        )

    # Intersección de índices válidos de los siete experimentos.
    common_indices = set(loaded[EXPERIMENTS[0]]["indices"].tolist())

    for name in EXPERIMENTS[1:]:
        common_indices &= set(loaded[name]["indices"].tolist())

    common_indices = sorted(common_indices)

    if not common_indices:
        raise RuntimeError("No hay índices comunes entre experimentos.")

    common_set = set(common_indices)

    print("\n" + "=" * 75)
    print(f"ÍNDICES COMUNES: {len(common_indices)}")
    print(
        f"Rango común: {common_indices[0]} - "
        f"{common_indices[-1]}"
    )
    print("=" * 75)

    # Comprobar que las poses reales coinciden en los índices comunes.
    # Comprobar que las poses reales (posición y orientación)
    # coinciden en todos los índices comunes.
    reference_gt = {}

    for name in EXPERIMENTS:
        pred, targ, indices = (
            loaded[name]["pred"],
            loaded[name]["targ"],
            loaded[name]["indices"],
        )

        for gt_pose, index in zip(targ, indices):
            if index not in common_set:
                continue

            gt_position, gt_rotation = w2c_to_c2w_body(gt_pose)

            if index not in reference_gt:
                reference_gt[index] = (
                    gt_position,
                    gt_rotation,
                )
            else:
                ref_position, ref_rotation = reference_gt[index]

                position_difference = np.linalg.norm(
                    gt_position - ref_position
                )

                orientation_difference = angular_error_deg(
                    gt_rotation,
                    ref_rotation,
                )

                if (
                    position_difference > 1e-5
                    or orientation_difference > 1e-4
                ):
                    raise RuntimeError(
                        f"Ground truth inconsistente: "
                        f"experimento={name}, índice={index}, "
                        f"posición={position_difference:.8f} m, "
                        f"orientación={orientation_difference:.8f} grados"
                    )

    print(
        "Ground truth verificado: posición y orientación "
        "coinciden en los índices comunes."
    )


    print("Ground truth: posiciones coincidentes en índices comunes.")

    # Métricas por experimento: conjunto original y conjunto común.
    summary_rows = []

    for name in EXPERIMENTS:
        rows_all = evaluated[name]

        rows_common = [
            row for row in rows_all
            if row["valid_index"] in common_set
        ]

        # CSV individual con todos los índices y marca de índice común.
        detailed_rows = []

        for row in rows_all:
            detailed = dict(row)
            detailed["is_common"] = (
                row["valid_index"] in common_set
            )
            detailed_rows.append(detailed)

        save_csv(
            OUTPUT_DIR / f"{name}_per_frame.csv",
            detailed_rows
        )

        stats_all = calculate_statistics(rows_all)
        stats_common = calculate_statistics(rows_common)

        summary = {"experiment": name}

        for key, value in stats_all.items():
            summary[f"all_{key}"] = value

        for key, value in stats_common.items():
            summary[f"common_{key}"] = value

        summary_rows.append(summary)

    save_csv(
        OUTPUT_DIR / "summary.csv",
        summary_rows
    )

    # Resumen legible en terminal.
    print("\n" + "=" * 105)
    print("COMPARACIÓN PRINCIPAL: MISMAS IMÁGENES PARA LOS SIETE EXPERIMENTOS")
    print("=" * 105)

    print(
        f"{'Experimento':12s} {'N':>5s} "
        f"{'Pos mean (m)':>14s} {'Pos med (m)':>13s} "
        f"{'Pos RMSE (m)':>14s} "
        f"{'Rot mean (°)':>13s} {'Rot med (°)':>12s}"
    )

    print("-" * 105)

    for row in summary_rows:
        print(
            f"{row['experiment']:12s} "
            f"{row['common_n']:5d} "
            f"{row['common_position_mean_m']:14.4f} "
            f"{row['common_position_median_m']:13.4f} "
            f"{row['common_position_rmse_m']:14.4f} "
            f"{row['common_orientation_mean_deg']:13.4f} "
            f"{row['common_orientation_median_deg']:12.4f}"
        )

    print("\n" + "=" * 75)
    print("RESULTADOS GUARDADOS EN:")
    print(OUTPUT_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
