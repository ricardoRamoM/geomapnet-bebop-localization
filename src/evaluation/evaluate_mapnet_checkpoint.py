
#!/usr/bin/env python3
# evaluate_mapnet_checkpoint.py
"""
Evalúa un checkpoint individual de MapNet a partir del .pkl
generado por geomapnet/scripts/eval.py.

Convierte poses W2C a posición C2W y orientación del cuerpo FLU.
Calcula métricas físicas comparables con ResNet-18.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from evaluate_mapnet_c2w import (
    w2c_to_c2w_body,
    calculate_statistics,
    save_csv,
)


def load_predictions(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    pred = np.asarray(data["pred_poses"], dtype=np.float64)
    targ = np.asarray(data["targ_poses"], dtype=np.float64)
    indices = np.asarray(data["valid_indices"], dtype=np.int64)

    if pred.shape != targ.shape:
        raise ValueError("pred_poses y targ_poses tienen formas distintas.")

    if pred.ndim != 2 or pred.shape[1] != 7:
        raise ValueError(
            f"Se esperaba una matriz N x 7. Forma recibida: {pred.shape}"
        )

    if len(indices) != len(pred):
        raise ValueError("La cantidad de índices no coincide con las poses.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Hay índices duplicados.")

    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(targ)):
        raise ValueError("Hay valores NaN o infinitos en las poses.")

    return pred, targ, indices


def evaluate(pred, targ, indices):
    rows = []

    for p, gt, index in zip(pred, targ, indices):

        pred_position, pred_rotation = w2c_to_c2w_body(p)
        gt_position, gt_rotation = w2c_to_c2w_body(gt)

        position_error = np.linalg.norm(
            pred_position - gt_position
        )

        # Error angular geodésico entre matrices de rotación.
        R_error = pred_rotation.T @ gt_rotation

        cos_angle = (np.trace(R_error) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        orientation_error = np.degrees(np.arccos(cos_angle))

        rows.append({
            "valid_index": int(index),

            "pred_x_m": pred_position[0],
            "pred_y_m": pred_position[1],
            "pred_z_m": pred_position[2],

            "gt_x_m": gt_position[0],
            "gt_y_m": gt_position[1],
            "gt_z_m": gt_position[2],

            "translation_error_m": position_error,
            "orientation_error_deg": orientation_error,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación física FLU de un checkpoint MapNet."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Archivo .pkl generado por eval.py."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directorio donde guardar los resultados."
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("EVALUACIÓN MAPNET: POSICIÓN C2W Y ORIENTACIÓN BODY FLU")
    print("=" * 75)

    pred, targ, indices = load_predictions(args.input)

    print(f"Archivo: {args.input}")
    print(f"Poses evaluadas: {len(indices)}")
    print(f"Rango de índices: {indices.min()} - {indices.max()}")

    rows = evaluate(pred, targ, indices)
    stats = calculate_statistics(rows)

    # Guardar errores y poses por imagen.
    csv_path = args.output / "per_frame_metrics.csv"
    save_csv(csv_path, rows)

    # Guardar resumen estadístico.
    summary_path = args.output / "summary.txt"

    with open(summary_path, "w") as file:
        file.write("EVALUACIÓN MAPNET - FLU\n")
        file.write("=" * 50 + "\n")
        file.write(f"Archivo: {args.input}\n")
        file.write(f"Imágenes evaluadas: {stats['n']}\n\n")

        file.write("ERROR DE POSICIÓN\n")
        file.write(f"Media:   {stats['position_mean_m']:.6f} m\n")
        file.write(f"Mediana: {stats['position_median_m']:.6f} m\n")
        file.write(f"RMSE:    {stats['position_rmse_m']:.6f} m\n")
        file.write(f"P95:     {stats['position_p95_m']:.6f} m\n\n")

        file.write("ERROR DE ORIENTACIÓN\n")
        file.write(f"Media:   {stats['orientation_mean_deg']:.6f} grados\n")
        file.write(f"Mediana: {stats['orientation_median_deg']:.6f} grados\n")
        file.write(f"P95:     {stats['orientation_p95_deg']:.6f} grados\n")

    print("\nRESULTADOS EN SISTEMA FLU")
    print("-" * 75)
    print(f"Imágenes evaluadas:       {stats['n']}")
    print(
        f"Posición media:           "
        f"{stats['position_mean_m']:.4f} m"
    )
    print(
        f"Posición mediana:         "
        f"{stats['position_median_m']:.4f} m"
    )
    print(
        f"Posición RMSE:            "
        f"{stats['position_rmse_m']:.4f} m"
    )
    print(
        f"Orientación media:        "
        f"{stats['orientation_mean_deg']:.4f} grados"
    )
    print(
        f"Orientación mediana:      "
        f"{stats['orientation_median_deg']:.4f} grados"
    )

    print("\nArchivos guardados:")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
