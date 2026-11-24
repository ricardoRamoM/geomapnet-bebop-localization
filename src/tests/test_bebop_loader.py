#!/usr/bin/env python3
"""Prueba de inicialización del loader Bebop y recálculo de pose_stats.txt.

Ejecutar desde:
    /data/research/mapnet/geomapnet

Comando:
    python test_bebop_loader.py
"""

import os
import numpy as np

from dataset_loaders.bebop import Bebop


def main():
    data_path = "../data/deepslam_data/Bebop"
    scene = "bebop"

    print("=== Prueba del loader Bebop ===")
    print("Directorio:", os.path.realpath(os.path.join(data_path, scene)))

    train_set = Bebop(
        scene=scene,
        data_path=data_path,
        train=True,
        transform=None,
    )

    print("\n--- Resumen entrenamiento ---")
    print("Imágenes:", len(train_set))
    print("Secuencias:", train_set.sequence_names)
    print("Rangos:", train_set.sequence_ranges)
    print("Pose shape:", train_set.poses.shape)
    print("mean_t:", train_set.mean_t)
    print("std_t :", train_set.std_t)

    expected_train = 856
    if len(train_set) != expected_train:
        raise RuntimeError(
            "Se esperaban {} imágenes de entrenamiento, se encontraron {}."
            .format(expected_train, len(train_set))
        )

    # Al construir train_set se debe haber escrito pose_stats.txt.
    stats_path = os.path.join(data_path, scene, "pose_stats.txt")
    stats = np.loadtxt(stats_path)
    if stats.shape != (2, 3):
        raise RuntimeError(
            "pose_stats.txt debe tener forma (2, 3), tiene {}.".format(stats.shape)
        )

    print("\npose_stats.txt actualizado:")
    print(stats)

    val_set = Bebop(
        scene=scene,
        data_path=data_path,
        train=False,
        transform=None,
    )

    print("\n--- Resumen validación ---")
    print("Imágenes:", len(val_set))
    print("Secuencias:", val_set.sequence_names)
    print("Rangos:", val_set.sequence_ranges)
    print("Pose shape:", val_set.poses.shape)

    expected_val = 304
    if len(val_set) != expected_val:
        raise RuntimeError(
            "Se esperaban {} imágenes de validación, se encontraron {}."
            .format(expected_val, len(val_set))
        )

    if not np.allclose(train_set.mean_t, val_set.mean_t):
        raise RuntimeError("Train y validación no están usando el mismo mean_t.")
    if not np.allclose(train_set.std_t, val_set.std_t):
        raise RuntimeError("Train y validación no están usando el mismo std_t.")

    print("\nOK: loader inicializado y estadísticas compartidas correctamente.")


if __name__ == "__main__":
    main()
