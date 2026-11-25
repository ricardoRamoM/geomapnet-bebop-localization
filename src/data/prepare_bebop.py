#!/usr/bin/env python3
# prepare_bebop.py
""" 
python src/data/prepare_bebop.py --version v2_3

Prepara el dataset Parrot Bebop 2 para el cargador MapNet.

Entrada:
    /data/research/bebop_raw/
        poses_split.csv
        frames_all/
        frames/

Salida:
    /data/research/bebop_data/
        <secuencia>/
            <imagen>.jpg
            poses.txt
        TrainSplit.txt
        TestSplit.txt
        stats.txt
        pose_stats.txt

Formato de poses.txt:
    filename r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

Las 12 cantidades numéricas son las primeras tres filas de T_c2w
(camera-to-world), aplanadas por filas.

Convenciones:
- Posición C_w: centro de cámara expresado en mundo FLU, en metros.
- Cuaternión qx,qy,qz,qw: orientación body-to-world.
- R_BODY_FROM_CAMERA: transforma vectores de cámara óptica
  [right, down, forward] a cuerpo FLU [forward, left, up].

El archivo poses.txt contiene exclusivamente poses C2W.
No se recalcula COLMAP->FLU. El CSV debe estar ya alineado en FLU.
"""

from __future__ import annotations

import csv
import math
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import argparse

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

#RAW_ROOT_BASE = Path("/data/research/bebop_raw")
#OUT_ROOT_BASE = Path("/data/research/bebop_data")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT_BASE = PROJECT_ROOT / "bebop_raw"
OUT_ROOT_BASE = PROJECT_ROOT / "bebop_data"


# Se asignan en main() según --version.
RAW_ROOT = None
OUT_ROOT = None
POSE_CSV = None
SEQUENCES = {}

# La versión se selecciona mediante --version.
# Los directorios específicos se configuran en main().

SEQUENCES_BY_VERSION = {
    "v2_0": {
        "bebop_001_val": "test",
        "bebop_002_train": "train",
        "bebop_003_train": "train",
        "bebop_004_train": "train",
        "bebop_005_train": "train",
        "ref": None,
    },

    "v2_2": {
        "bebop_001_val": "test",
        "bebop_002_train": "train",
        "bebop_003_train": "train",
        "bebop_004_train": "train",
        "bebop_005_train": "train",

        "bebop_test_001": "train",
        "bebop_test_002": "train",
        "bebop_test_003": "train",
        "bebop_test_004": "test",
        "bebop_test_005": "train",
        "bebop_test_006": "train",
        "bebop_test_007": "train",
        "bebop_test_008": "test",
        "bebop_test_009": "train",
        "bebop_test_010": "test",
        "bebop_test_011": "train",

        "ref": None,
    },

    "v2_3": {
            "bebop_001_val": "test",
            "bebop_002_train": "train",
            "bebop_003_train": "train",
            "bebop_004_train": "train",
            "bebop_005_train": "train",
    
            "bebop_test_001": "train",
            "bebop_test_002": "train",
            "bebop_test_003": "train",
            "bebop_test_004": "test",
            "bebop_test_005": "train",
            "bebop_test_006": "train",
            "bebop_test_007": "train",
            "bebop_test_008": "test",
            "bebop_test_009": "train",
            "bebop_test_010": "test",
            "bebop_test_011": "train",
    
            "ref": None,
        },
}

# Convierte cámara óptica [right, down, forward] a cuerpo FLU
# [forward, left, up].
R_BODY_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

USE_SYMLINK = False
REUSE_EXISTING_IMAGES = True
RGB_STATS_TRAIN_ONLY = True

# ---------------------------------------------------------------------------


def first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Falta una columna requerida entre: {names}")


def parse_float(row: dict[str, str], names: tuple[str, ...], line: int) -> float:
    raw = first_present(row, names)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Valor no numérico en línea {line}, columnas {names}: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Valor no finito en línea {line}: {raw!r}")
    return value


def quaternion_xyzw_to_rotation(qx: float, qy: float,
                                qz: float, qw: float) -> np.ndarray:
    """Matriz de rotación body-to-world a partir de cuaternión xyzw."""
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Cuaternión de norma casi cero.")
    x, y, z, w = q / norm

    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )



def make_camera_to_world_pose(pose: dict) -> np.ndarray:
    """
    Construye T_wc (camera-to-world).

    R_wc = R_wb R_bc
    t_wc = C_w

    La traslación es la posición del centro de cámara
    expresada en el sistema de coordenadas mundo FLU.
    """
    R_wb = quaternion_xyzw_to_rotation(
        pose["qx"], pose["qy"], pose["qz"], pose["qw"]
    )

    R_wc = R_wb @ R_BODY_FROM_CAMERA

    C_w = np.asarray(
        [pose["x"], pose["y"], pose["z"]],
        dtype=np.float64
    )

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = C_w

    if not np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-6):
        raise ValueError(f"Rotación CTW no ortonormal para {pose['image_name']}")

    if not np.isclose(np.linalg.det(R_wc), 1.0, atol=1e-6):
        raise ValueError(f"Determinante CTW inválido para {pose['image_name']}")

    return T_wc


def read_pose_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {path}")

    poses: dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"El CSV no tiene encabezado: {path}")

        required = {"image_name", "video_name"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Faltan columnas {sorted(missing)}. "
                f"Columnas disponibles: {reader.fieldnames}"
            )

        for line, row in enumerate(reader, start=2):
            image_name = first_present(row, ("image_name",))
            video_name = first_present(row, ("video_name",))
            if image_name in poses:
                raise ValueError(f"Imagen duplicada en línea {line}: {image_name}")

            pose = {
                "image_name": image_name,
                "video_name": video_name,
                "x": parse_float(row, ("x", "x_m"), line),
                "y": parse_float(row, ("y", "y_m"), line),
                "z": parse_float(row, ("z", "z_m"), line),
                "qx": parse_float(row, ("qx",), line),
                "qy": parse_float(row, ("qy",), line),
                "qz": parse_float(row, ("qz",), line),
                "qw": parse_float(row, ("qw",), line),
                "csv_line": line,
            }

            qnorm = np.linalg.norm(
                [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]
            )
            if qnorm < 1e-10:
                raise ValueError(f"Cuaternión inválido en línea {line}: {image_name}")

            # Normalizar cuaternión para eliminar pequeñas desviaciones numéricas.
            for key in ("qx", "qy", "qz", "qw"):
                pose[key] /= qnorm

            # Orden temporal si el CSV incluye frame_index.
            frame_index = row.get("frame_index", "")
            try:
                pose["frame_index"] = int(frame_index) if frame_index.strip() else None
            except ValueError:
                pose["frame_index"] = None

            poses[image_name] = pose

    if not poses:
        raise ValueError("El CSV no contiene poses.")
    return poses


def sequence_for_pose(pose: dict) -> str | None:
    video_name = pose["video_name"]

    if video_name in SEQUENCES:
        return video_name

    # Las imágenes de referencia pueden aparecer identificadas
    # directamente por el nombre del archivo en video_name.
    if video_name.startswith("ref_frame"):
        return "ref"

    return None


def find_source_image(image_name: str, video_name: str,
                      sequence: str) -> Path:
    candidates = [
        RAW_ROOT / "frames_all" / image_name,
        RAW_ROOT / "frames" / video_name / image_name,
        RAW_ROOT / "frames" / sequence / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No se encontró {image_name}. Rutas revisadas:\n  {searched}"
    )


def sort_key(pose: dict):
    # frame_index es preferible para ordenar las secuencias de vídeo.
    if pose["frame_index"] is not None:
        return (0, pose["frame_index"], pose["image_name"])
    return (1, pose["image_name"])


def prepare_sequences(poses: dict[str, dict]):
    grouped: dict[str, list[dict]] = defaultdict(list)
    ignored = []

    for pose in poses.values():
        sequence = sequence_for_pose(pose)
        if sequence is None:
            ignored.append((pose["image_name"], pose["video_name"]))
        else:
            grouped[sequence].append(pose)

    train_sequences = []
    test_sequences = []
    output_images: dict[str, list[Path]] = {}
    all_c2w: dict[str, list[np.ndarray]] = {}

    for sequence, split in SEQUENCES.items():
        rows = sorted(grouped.get(sequence, []), key=sort_key)

        # Las secuencias de referencia son opcionales.
        if not rows and split is None:
            print(f"[{sequence:16s}] omitida: no contiene poses.")
            continue

        if not rows:
            raise RuntimeError(
                f"No hay poses para la secuencia {sequence!r}."
            )

        destination_dir = OUT_ROOT / sequence
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Elimina el archivo antiguo para evitar confundirlo con el nuevo
        # formato C2W.
        legacy_pose_file = destination_dir / "poses_CTW.txt"
        if legacy_pose_file.exists():
            legacy_pose_file.unlink()

        image_paths = []
        c2w_matrices = []

        poses_path = destination_dir / "poses.txt"

        with poses_path.open("w", encoding="utf-8") as pose_file:
            pose_file.write(
                "# C2W: filename "
                "r11 r12 r13 tx "
                "r21 r22 r23 ty "
                "r31 r32 r33 tz\n"
            )

            for pose in rows:
                image_name = pose["image_name"]

                source = find_source_image(
                    image_name,
                    pose["video_name"],
                    sequence,
                )

                destination = destination_dir / image_name

                if destination.is_symlink() or destination.exists():
                    if not REUSE_EXISTING_IMAGES:
                        destination.unlink()

                        if USE_SYMLINK:
                            destination.symlink_to(source.resolve())
                        else:
                            shutil.copy2(source, destination)
                else:
                    if USE_SYMLINK:
                        destination.symlink_to(source.resolve())
                    else:
                        shutil.copy2(source, destination)

                try:
                    with Image.open(destination) as image:
                        image.verify()
                except Exception as exc:
                    raise RuntimeError(
                        f"Imagen corrupta o ilegible: {destination}"
                    ) from exc

                # Única representación de pose: camera-to-world.
                T_c2w = make_camera_to_world_pose(pose)
                matrix_c2w = T_c2w[:3, :4]

                values = matrix_c2w.reshape(-1)

                pose_file.write(
                    image_name + " "
                    + " ".join(f"{value:.12f}" for value in values)
                    + "\n"
                )

                image_paths.append(destination)
                c2w_matrices.append(matrix_c2w.copy())

        output_images[sequence] = image_paths
        all_c2w[sequence] = c2w_matrices

        if split == "train":
            train_sequences.append(sequence)
        elif split == "test":
            test_sequences.append(sequence)
        elif split is None:
            # Se prepara, pero no se incorpora a TrainSplit/TestSplit.
            pass
        else:
            raise ValueError(
                f"Split no reconocido: {sequence} -> {split}"
            )

        split_label = split if split is not None else "excluded"

        print(
            f"[{sequence:16s}] split={split_label:8s} | "
            f"{len(image_paths):4d} imágenes"
        )

    if ignored:
        print(
            f"Aviso: {len(ignored)} filas ignoradas por no pertenecer a "
            f"SEQUENCES. Ejemplos: {ignored[:5]}"
        )

    return (
        train_sequences,
        test_sequences,
        output_images,
        all_c2w,
    )


def write_split_files(train_sequences: list[str],
                      test_sequences: list[str]) -> None:
    (OUT_ROOT / "TrainSplit.txt").write_text(
        "\n".join(train_sequences) + "\n", encoding="utf-8"
    )
    (OUT_ROOT / "TestSplit.txt").write_text(
        "\n".join(test_sequences) + "\n", encoding="utf-8"
    )


def write_rgb_stats(train_sequences, test_sequences, output_images) -> None:
    sequences = (
        train_sequences
        if RGB_STATS_TRAIN_ONLY
        else train_sequences + test_sequences
    )
    description = (
        "solo entrenamiento"
        if RGB_STATS_TRAIN_ONLY
        else "entrenamiento + test"
    )

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    print(f"\nCalculando estadísticas RGB ({description})...")
    for sequence in sequences:
        for image_path in output_images[sequence]:
            with Image.open(image_path) as image:
                array = np.asarray(
                    image.convert("RGB"), dtype=np.float64
                ) / 255.0
            pixels = array.reshape(-1, 3)
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += np.square(pixels).sum(axis=0)
            pixel_count += pixels.shape[0]

    if pixel_count == 0:
        raise RuntimeError("No hay píxeles para calcular stats.txt.")

    mean = channel_sum / pixel_count
    variance = np.maximum(
        channel_sq_sum / pixel_count - mean**2, 1e-12
    )
    std = np.sqrt(variance)

    # IMPORTANTE: train.py aplica sqrt(stats[1]); por eso guardamos VARIANZA,
    # no desviación estándar, en la segunda fila.
    with (OUT_ROOT / "stats.txt").open("w", encoding="utf-8") as file:
        file.write(" ".join(f"{v:.12e}" for v in mean) + "\n")
        file.write(" ".join(f"{v:.12e}" for v in variance) + "\n")

    print(f"RGB mean     : {mean}")
    print(f"RGB variance : {variance}")
    print(f"RGB std      : {std}")
    print(f"Píxeles      : {pixel_count:,}")


def write_pose_stats(train_sequences, all_c2w) -> None:
    """
    Calcula media y desviación estándar de las posiciones de cámara
    en coordenadas mundo (C2W).

    En una matriz C2W, la última columna de la matriz 3x4
    representa C_w = [x, y, z] en el mundo FLU.
    """
    translations = []

    for sequence in train_sequences:
        for matrix_3x4 in all_c2w[sequence]:
            translations.append(matrix_3x4[:, 3])

    translations = np.asarray(translations, dtype=np.float64)

    if translations.size == 0:
        raise RuntimeError(
            "No hay posiciones C2W de las secuencias de entrenamiento."
        )

    mean_t = translations.mean(axis=0)
    std_t = translations.std(axis=0)
    std_t = np.maximum(std_t, 1e-6)

    with (OUT_ROOT / "pose_stats.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(" ".join(f"{v:.12f}" for v in mean_t) + "\n")
        file.write(" ".join(f"{v:.12f}" for v in std_t) + "\n")

    print(f"\nPose C2W mean_Cw: {mean_t}")
    print(f"Pose C2W std_Cw : {std_t}")
    print("pose_stats.txt calculado solo con secuencias TrainSplit.")


def main() -> None:
    global RAW_ROOT, OUT_ROOT, POSE_CSV, SEQUENCES

    parser = ArgumentParser(
        description="Prepara el dataset Bebop para MapNet."
    )
    parser.add_argument(
        "--version",
        required=True,
        choices=["v2_0", "v2_2", "v2_3"],
        help="Versión del dataset que se desea preparar."
    )
    args = parser.parse_args()

    version = args.version

    RAW_ROOT = RAW_ROOT_BASE / version
    OUT_ROOT = OUT_ROOT_BASE / version
    POSE_CSV = RAW_ROOT / "poses_split.csv"
    SEQUENCES = SEQUENCES_BY_VERSION[version]

    print("=== Preparación Bebop para MapNet ===")
    print(f"Versión: {version}")
    print(f"Origen : {RAW_ROOT}")
    print(f"CSV    : {POSE_CSV}")
    print(f"Salida : {OUT_ROOT}\n")

    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de entrada: {RAW_ROOT}"
        )

    if not (RAW_ROOT / "frames").is_dir():
        raise FileNotFoundError(
            f"No existe el directorio frames: {RAW_ROOT / 'frames'}"
        )

    if not (RAW_ROOT / "frames_all").is_dir():
        raise FileNotFoundError(
            f"No existe el directorio frames_all: {RAW_ROOT / 'frames_all'}"
        )

    if not POSE_CSV.is_file():
        raise FileNotFoundError(f"No existe el CSV: {POSE_CSV}")

    if not np.allclose(
        R_BODY_FROM_CAMERA @ R_BODY_FROM_CAMERA.T,
        np.eye(3), atol=1e-8
    ):
        raise RuntimeError("R_BODY_FROM_CAMERA no es ortonormal.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    poses = read_pose_csv(POSE_CSV)
    print(f"Filas de poses leídas: {len(poses)}")

    (
        train_sequences,
        test_sequences,
        output_images,
        all_c2w,
    ) = prepare_sequences(poses)

    write_split_files(train_sequences, test_sequences)
    write_rgb_stats(train_sequences, test_sequences, output_images)
    write_pose_stats(train_sequences, all_c2w)

    train_count = sum(len(output_images[s]) for s in train_sequences)
    test_count = sum(len(output_images[s]) for s in test_sequences)

    print("\n=== RESUMEN ===")
    print(f"Train sequences : {train_sequences}")
    print(f"Test sequences  : {test_sequences}")
    print(f"Train images    : {train_count}")
    print(f"Test images     : {test_count}")
    print(f"Total           : {train_count + test_count}")
    print(f"Output          : {OUT_ROOT}")
    print("\nPreparación terminada.")

if __name__ == "__main__":
    main()