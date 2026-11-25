
#!/usr/bin/env python3
"""
Verificador independiente del flujo C2W de MapNet para Parrot Bebop.

Ejemplos:
    python src/tests/verify_c2w_mapnet_pipeline.py --versions v2_2 v2_3
    python src/tests/verify_c2w_mapnet_pipeline.py --all
    python src/tests/verify_c2w_mapnet_pipeline.py --versions v2_3 \
        --raw-root /data/research/bebop_raw \
        --processed-root /data/research/bebop_data

Características:
- Solo lectura: no crea ni modifica archivos.
- Compara poses.txt contra las hipótesis C2W y W2C.
- Clasifica cada versión geométricamente.
- Comprueba TrainSplit.txt y TestSplit.txt contra SEQUENCES_BY_VERSION.
- Comprueba la exclusión de referencias.
- Audita estáticamente el código activo de MapNet.
- No utiliza evaluadores antiguos.

Convenciones:
- CSV: posición FLU y quaternion xyzw.
- FLU: X adelante, Y izquierda, Z arriba.
- Cámara COLMAP: X derecha, Y abajo, Z adelante.
- poses.txt: matriz C2W de cámara, 3x4, aplanada por filas.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_ROOT = Path("/data/research/bebop_raw")
DEFAULT_PROCESSED_ROOT = Path("/data/research/bebop_data")

PREPARE = REPO_ROOT / "src/data/prepare_bebop.py"

POSE_UTILS = REPO_ROOT / "geomapnet/common/pose_utils.py"
BEBOP_LOADER = REPO_ROOT / "geomapnet/dataset_loaders/bebop.py"
COMPOSITE_LOADER = REPO_ROOT / "geomapnet/dataset_loaders/composite.py"
CRITERION = REPO_ROOT / "geomapnet/common/criterion.py"
MODEL = REPO_ROOT / "geomapnet/models/posenet.py"
TRAIN = REPO_ROOT / "geomapnet/scripts/train.py"
EVAL = REPO_ROOT / "geomapnet/scripts/eval.py"

R_BODY_FROM_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

class Report:
    def __init__(self, version: str):
        self.version = version
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.infos = 0
        self.geometry_classification = "NO VERIFICADA"

    def header(self, title: str):
        print("\n" + "=" * 76)
        print(title)
        print("=" * 76)

    def ok(self, message: str):
        self.passed += 1
        print(f"[PASS] {message}")

    def fail(self, message: str):
        self.failed += 1
        print(f"[FAIL] {message}")

    def warn(self, message: str):
        self.warnings += 1
        print(f"[WARN] {message}")

    def info(self, message: str):
        self.infos += 1
        print(f"[INFO] {message}")

    def summary(self):
        print("\n" + "-" * 76)
        print(
            f"RESUMEN {self.version}: "
            f"{self.passed} PASS | {self.failed} FAIL | "
            f"{self.warnings} WARN | {self.infos} INFO"
        )
        if self.geometry_classification != "NO VERIFICADA":
            print(
                f"Clasificación geométrica: "
                f"{self.geometry_classification}"
            )
        print("-" * 76)


# ---------------------------------------------------------------------
# Quaternion and rotation utilities
# ---------------------------------------------------------------------

def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion inválido o de norma cero.")

    return q / norm


def quaternion_xyzw_to_rotation(q: np.ndarray) -> np.ndarray:
    """Quaternion xyzw -> matriz de rotación body-to-world."""
    x, y, z, w = normalize_quaternion_xyzw(q)

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    relative = R1.T @ R2
    value = (np.trace(relative) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def make_camera_c2w(
    position_flu: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """
    Construye C2W de cámara a partir de la pose C2W del cuerpo FLU.

    R_world_camera = R_world_body @ R_body_camera
    """
    R_world_body = quaternion_xyzw_to_rotation(quaternion_xyzw)
    R_world_camera = R_world_body @ R_BODY_FROM_CAMERA

    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_world_camera
    T_c2w[:3, 3] = np.asarray(position_flu, dtype=np.float64)

    return T_c2w


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "filename": (
        "filename", "file_name", "image", "image_name",
        "image_filename", "frame", "name",
    ),
    "x": ("x", "pos_x", "position_x", "x_m"),
    "y": ("y", "pos_y", "position_y", "y_m"),
    "z": ("z", "pos_z", "position_z", "z_m"),
    "qx": ("qx", "q_x"),
    "qy": ("qy", "q_y"),
    "qz": ("qz", "q_z"),
    "qw": ("qw", "q_w"),
    "split": ("split", "subset", "partition"),
    "sequence": ("video_name", "sequence", "sequence_name", "video"),
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def resolve_columns(fieldnames: list[str]) -> dict[str, Optional[str]]:
    normalized = {
        normalize_column_name(name): name
        for name in fieldnames
    }

    result = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        result[canonical] = None

        for alias in aliases:
            key = normalize_column_name(alias)

            if key in normalized:
                result[canonical] = normalized[key]
                break

    required = (
        "filename", "x", "y", "z",
        "qx", "qy", "qz", "qw",
    )

    missing = [
        key for key in required
        if result[key] is None
    ]

    if missing:
        raise ValueError(
            "No se pudieron identificar columnas obligatorias: "
            + ", ".join(missing)
            + f". Columnas encontradas: {fieldnames}"
        )

    return result


def is_reference_row(row: dict) -> bool:
    filename = row["basename"].lower()
    sequence = (row.get("sequence") or "").lower()

    return (
        filename.startswith("ref_frame")
        or filename.startswith("reference")
        or sequence.startswith("ref_frame")
        or sequence.startswith("reference")
    )


def read_source_csv(csv_path: Path):
    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(f"CSV sin encabezados: {csv_path}")

        columns = resolve_columns(reader.fieldnames)
        rows = []

        for line_number, row in enumerate(reader, start=2):
            filename = (row[columns["filename"]] or "").strip()

            if not filename:
                continue

            try:
                position = np.array(
                    [
                        float(row[columns["x"]]),
                        float(row[columns["y"]]),
                        float(row[columns["z"]]),
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row[columns["qx"]]),
                        float(row[columns["qy"]]),
                        float(row[columns["qz"]]),
                        float(row[columns["qw"]]),
                    ],
                    dtype=np.float64,
                )

                if not np.all(np.isfinite(position)):
                    raise ValueError("posición no finita")

                if not np.all(np.isfinite(quaternion)):
                    raise ValueError("quaternion no finito")

                q_norm = np.linalg.norm(quaternion)

                if q_norm < 1e-12:
                    raise ValueError("quaternion de norma cero")

                rows.append(
                    {
                        "filename": filename,
                        "basename": Path(filename).name,
                        "position": position,
                        "quaternion": quaternion / q_norm,
                        "split": (
                            (row.get(columns["split"]) or "")
                            .strip()
                            .lower()
                            if columns["split"] else ""
                        ),
                        "sequence": (
                            (row.get(columns["sequence"]) or "")
                            .strip()
                            if columns["sequence"] else ""
                        ),
                        "line": line_number,
                    }
                )

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Fila CSV {line_number}, {filename}: {exc}"
                ) from exc

    return rows, columns


# ---------------------------------------------------------------------
# Dataset discovery and poses.txt parsing
# ---------------------------------------------------------------------

def find_source_csv(
    raw_root: Path,
    version: str,
) -> Optional[Path]:

    candidates = [
        raw_root / version / "poses_split.csv",
        raw_root / version / "poses.csv",
        raw_root / "poses_split.csv",
        raw_root / "poses.csv",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def find_processed_version(
    processed_root: Path,
    version: str,
) -> Optional[Path]:

    candidates = [
        processed_root / version,
        processed_root / "Bebop" / version,
        processed_root / "Bebop" / "bebop" / version,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            if list(candidate.rglob("poses.txt")):
                return candidate

    return None


def parse_poses_file(path: Path):
    """
    Lee:
        image.jpg value1 value2 ... value12

    Los doce valores representan una matriz 3x4 por filas.
    """
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 13:
                raise ValueError(
                    f"{path}:{line_number}: se esperaban 13 campos "
                    f"(nombre + 12 valores), se encontraron {len(parts)}."
                )

            filename = parts[0]
            values = np.asarray(parts[1:], dtype=np.float64)

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{path}:{line_number}: valores no finitos."
                )

            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = values.reshape(3, 4)

            records.append(
                {
                    "filename": filename,
                    "basename": Path(filename).name,
                    "T": T,
                    "path": path,
                    "line": line_number,
                }
            )

    return records


def collect_processed_poses(version_dir: Path):
    all_records = []
    poses_files = sorted(version_dir.rglob("poses.txt"))

    for path in poses_files:
        all_records.extend(parse_poses_file(path))

    return poses_files, all_records


# ---------------------------------------------------------------------
# Geometric checks
# ---------------------------------------------------------------------

def verify_rotation_matrix(
    R: np.ndarray,
    tolerance: float = 1e-5,
):
    orthogonality_error = np.linalg.norm(
        R.T @ R - np.eye(3)
    )
    determinant = np.linalg.det(R)

    valid = (
        orthogonality_error <= tolerance
        and abs(determinant - 1.0) <= tolerance
    )

    return valid, orthogonality_error, determinant


def compare_csv_and_poses(
    report: Report,
    csv_rows: list[dict],
    pose_records: list[dict],
    position_tolerance: float,
    angle_tolerance_deg: float,
):
    report.header("DIAGNÓSTICO GEOMÉTRICO C2W / W2C")

    by_basename = {}

    for record in pose_records:
        by_basename.setdefault(
            record["basename"], []
        ).append(record)

    c2w_position_errors = []
    c2w_angle_errors = []
    w2c_position_errors = []
    w2c_angle_errors = []

    missing = []
    ambiguous = []
    matched_names = set()

    for row in csv_rows:
        candidates = by_basename.get(row["basename"], [])

        if not candidates:
            missing.append(row)
            continue

        if len(candidates) > 1:
            exact = [
                item for item in candidates
                if item["filename"] == row["filename"]
            ]

            if len(exact) == 1:
                candidate = exact[0]
            else:
                ambiguous.append(row["filename"])
                continue
        else:
            candidate = candidates[0]

        matched_names.add(candidate["basename"])

        expected_c2w = make_camera_c2w(
            row["position"],
            row["quaternion"],
        )

        expected_w2c = np.linalg.inv(expected_c2w)
        actual = candidate["T"]

        c2w_position_errors.append(
            np.linalg.norm(
                actual[:3, 3] - expected_c2w[:3, 3]
            )
        )

        c2w_angle_errors.append(
            rotation_error_deg(
                actual[:3, :3],
                expected_c2w[:3, :3],
            )
        )

        w2c_position_errors.append(
            np.linalg.norm(
                actual[:3, 3] - expected_w2c[:3, 3]
            )
        )

        w2c_angle_errors.append(
            rotation_error_deg(
                actual[:3, :3],
                expected_w2c[:3, :3],
            )
        )

    # Las referencias también se comparan geométricamente.
    # Que estén excluidas de TrainSplit/TestSplit no significa
    # necesariamente que estén ausentes de poses.txt.

    if missing:
        missing_references = [
            row for row in missing
            if is_reference_row(row)
        ]

        missing_normal = [
            row for row in missing
            if not is_reference_row(row)
        ]

        if missing_normal:
            report.fail(
                f"{len(missing_normal)} imágenes normales del CSV "
                f"no aparecen en poses.txt. "
                f"Ejemplos: "
                f"{[r['filename'] for r in missing_normal[:5]]}"
            )

        if missing_references:
            report.fail(
                f"{len(missing_references)} referencias del CSV "
                "no aparecen en poses.txt. "
                "Se esperaba que las referencias conservaran "
                "su correspondencia geométrica."
            )

    if ambiguous:
        report.fail(
            f"{len(ambiguous)} imágenes tienen correspondencia "
            f"ambigua. Ejemplos: {ambiguous[:5]}"
        )

    matched = len(c2w_position_errors)

    if matched == 0:
        report.fail("No se pudo comparar ninguna pose.")
        report.geometry_classification = "NO VERIFICADA"
        return

    c2w_pos_max = float(max(c2w_position_errors))
    c2w_ang_max = float(max(c2w_angle_errors))
    w2c_pos_max = float(max(w2c_position_errors))
    w2c_ang_max = float(max(w2c_angle_errors))

    c2w_ok = (
        c2w_pos_max <= position_tolerance
        and c2w_ang_max <= angle_tolerance_deg
    )

    w2c_ok = (
        w2c_pos_max <= position_tolerance
        and w2c_ang_max <= angle_tolerance_deg
    )

    print("\nComparación contra C2W esperado:")
    print(f"  Error posición máximo: {c2w_pos_max:.6e} m")
    print(
        f"  Error posición medio:  "
        f"{np.mean(c2w_position_errors):.6e} m"
    )
    print(f"  Error angular máximo:  {c2w_ang_max:.6f} grados")
    print(
        f"  Error angular medio:   "
        f"{np.mean(c2w_angle_errors):.6f} grados"
    )

    print("\nComparación contra W2C esperado:")
    print(f"  Error posición máximo: {w2c_pos_max:.6e} m")
    print(
        f"  Error posición medio:  "
        f"{np.mean(w2c_position_errors):.6e} m"
    )
    print(f"  Error angular máximo:  {w2c_ang_max:.6f} grados")
    print(
        f"  Error angular medio:   "
        f"{np.mean(w2c_angle_errors):.6f} grados"
    )

    print("\n" + "=" * 68)

    if c2w_ok and not w2c_ok:
        report.geometry_classification = "C2W CONFIRMADO"
        print("CLASIFICACIÓN: C2W CONFIRMADO")
        report.ok(
            "poses.txt conserva las matrices C2W esperadas."
        )
        report.ok(
            "La hipótesis W2C queda descartada."
        )

    elif w2c_ok and not c2w_ok:
        report.geometry_classification = "W2C DETECTADO"
        print("CLASIFICACIÓN: W2C DETECTADO")
        report.fail(
            "poses.txt contiene las matrices W2C, no las C2W."
        )

    elif c2w_ok and w2c_ok:
        report.geometry_classification = "AMBIGUA"
        print("CLASIFICACIÓN: AMBIGUA")
        report.warn(
            "Ambas hipótesis cumplen las tolerancias. "
            "No se puede distinguir C2W de W2C."
        )

    else:
        report.geometry_classification = (
            "OTRA TRANSFORMACIÓN DETECTADA"
        )
        print("CLASIFICACIÓN: OTRA TRANSFORMACIÓN DETECTADA")
        report.fail(
            "Las poses no coinciden ni con C2W ni con W2C esperado."
        )

    print("=" * 68)

    if matched == len(csv_rows) and not ambiguous and not missing:
        report.ok(
            f"Correspondencia completa: {matched}/{len(csv_rows)} poses."
        )
    else:
        report.info(
            f"Correspondencias geométricas: {matched}/{len(csv_rows)}."
        )


def verify_pose_file_structure(
    report: Report,
    pose_records: list[dict],
):
    report.header("ESTRUCTURA DE poses.txt")

    if not pose_records:
        report.fail("No se encontraron registros en poses.txt.")
        return

    report.ok(
        f"Se leyeron {len(pose_records)} matrices 3x4."
    )

    errors = []

    for record in pose_records:
        T = record["T"]
        valid, ortho_error, determinant = verify_rotation_matrix(
            T[:3, :3]
        )

        if not valid:
            errors.append(
                (
                    record["filename"],
                    ortho_error,
                    determinant,
                )
            )

    if errors:
        report.fail(
            f"{len(errors)} matrices tienen rotación no válida. "
            f"Ejemplo: {errors[0]}"
        )
    else:
        report.ok(
            "Todas las rotaciones son ortogonales y det(R)=+1."
        )

    translations = np.asarray(
        [record["T"][:3, 3] for record in pose_records]
    )

    if np.all(np.isfinite(translations)):
        report.ok("Todas las traslaciones son finitas.")
    else:
        report.fail("Se encontraron traslaciones no finitas.")


# ---------------------------------------------------------------------
# Preparation configuration and split checks
# ---------------------------------------------------------------------

def read_sequence_configuration(
    prepare_path: Path,
) -> Optional[dict]:
    """
    Lee SEQUENCES_BY_VERSION mediante AST.

    No importa prepare_bebop.py ni ejecuta su main().
    """
    if not prepare_path.is_file():
        return None

    source = prepare_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "SEQUENCES_BY_VERSION"
            ):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return None

                if isinstance(value, dict):
                    return value

    return None


def read_split_file(path: Path) -> set[str]:
    names = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 1:
                raise ValueError(
                    f"Formato inesperado en {path}: {line!r}. "
                    "Se esperaba una secuencia por línea."
                )

            names.add(parts[0])

    return names


def find_split_file(
    version_dir: Path,
    filename: str,
) -> Optional[Path]:
    direct = version_dir / filename

    if direct.is_file():
        return direct

    matches = [
        path for path in version_dir.rglob(filename)
        if path.is_file()
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def normalize_csv_sequence(row: dict) -> str:
    sequence = (row.get("sequence") or "").strip()

    if not sequence:
        return ""

    if is_reference_row(row):
        return "ref"

    return Path(sequence).name


def verify_csv_splits(
    report: Report,
    rows: list[dict],
):
    """
    Describe los valores split del CSV, pero no los confunde
    con los splits efectivos de MapNet.
    """
    report.header("METADATOS DE SPLIT DEL CSV")

    split_counts = {}

    for row in rows:
        split = row["split"] or "(sin split)"
        split_counts[split] = split_counts.get(split, 0) + 1

    for split, count in sorted(split_counts.items()):
        report.info(f"CSV split={split}: {count} filas")

    references = [
        row for row in rows
        if is_reference_row(row)
    ]

    normal_rows = [
        row for row in rows
        if not is_reference_row(row)
    ]

    if references:
        report.ok(
            f"Referencias identificadas por filename/sequence: "
            f"{len(references)}."
        )

        ref_splits = sorted({
            row["split"] or "(sin split)"
            for row in references
        })

        report.info(
            "Etiquetas CSV de las referencias: "
            + ", ".join(ref_splits)
        )

    else:
        report.info("No se encontraron filas de referencia.")

    if normal_rows:
        sequences = {
            normalize_csv_sequence(row)
            for row in normal_rows
            if normalize_csv_sequence(row)
        }

        report.ok(
            f"Filas normales del CSV: {len(normal_rows)}; "
            f"secuencias de vídeo: {len(sequences)}."
        )

    if references and all(
        row["split"] == "train"
        for row in references
    ):
        report.info(
            "Las referencias tienen split=train en el CSV. "
            "La exclusión real se verifica mediante "
            "SEQUENCES_BY_VERSION y los archivos MapNet."
        )


def verify_processed_splits(
    report: Report,
    version: str,
    version_dir: Path,
    csv_rows: list[dict],
    prepare_path: Path,
):
    report.header("VALIDACIÓN DE SPLITS MAPNET")

    configuration = read_sequence_configuration(prepare_path)

    if configuration is None:
        report.fail(
            "No se pudo leer SEQUENCES_BY_VERSION "
            "de prepare_bebop.py."
        )
        return

    if version not in configuration:
        report.fail(
            f"La versión {version} no aparece en "
            "SEQUENCES_BY_VERSION."
        )
        return

    version_config = configuration[version]

    expected_train = {
        name
        for name, split in version_config.items()
        if split == "train"
    }

    expected_test = {
        name
        for name, split in version_config.items()
        if split == "test"
    }

    expected_references = {
        name
        for name, split in version_config.items()
        if split is None
    }

    train_path = find_split_file(
        version_dir,
        "TrainSplit.txt",
    )
    test_path = find_split_file(
        version_dir,
        "TestSplit.txt",
    )

    if train_path is None or test_path is None:
        report.fail(
            "No se encontraron ambos archivos "
            "TrainSplit.txt y TestSplit.txt."
        )
        return

    try:
        actual_train = read_split_file(train_path)
        actual_test = read_split_file(test_path)
    except Exception as exc:
        report.fail(f"Error leyendo archivos de split: {exc}")
        return

    report.info(f"TrainSplit: {train_path}")
    report.info(f"TestSplit:  {test_path}")

    print(f"\nTrain esperado: {len(expected_train)} secuencias")
    print(f"Train real:     {len(actual_train)} secuencias")
    print(f"Test esperado:  {len(expected_test)} secuencias")
    print(f"Test real:      {len(actual_test)} secuencias")

    missing_train = sorted(expected_train - actual_train)
    extra_train = sorted(actual_train - expected_train)

    missing_test = sorted(expected_test - actual_test)
    extra_test = sorted(actual_test - expected_test)

    overlap = sorted(actual_train.intersection(actual_test))

    if not missing_train and not extra_train:
        report.ok(
            f"TrainSplit.txt coincide exactamente con "
            f"SEQUENCES_BY_VERSION ({len(actual_train)} secuencias)."
        )
    else:
        report.fail(
            "TrainSplit.txt no coincide con la configuración. "
            f"Faltantes={missing_train}; adicionales={extra_train}"
        )

    if not missing_test and not extra_test:
        report.ok(
            f"TestSplit.txt coincide exactamente con "
            f"SEQUENCES_BY_VERSION ({len(actual_test)} secuencias)."
        )
    else:
        report.fail(
            "TestSplit.txt no coincide con la configuración. "
            f"Faltantes={missing_test}; adicionales={extra_test}"
        )

    if overlap:
        report.fail(
            "Hay secuencias presentes simultáneamente en train y test: "
            + repr(overlap)
        )
    else:
        report.ok(
            "No hay secuencias compartidas entre train y test."
        )

    # Verificar cobertura del CSV.
    csv_sequences = {
        normalize_csv_sequence(row)
        for row in csv_rows
        if not is_reference_row(row)
        and normalize_csv_sequence(row)
    }

    configured_sequences = expected_train | expected_test

    missing_config = sorted(csv_sequences - configured_sequences)
    absent_csv = sorted(configured_sequences - csv_sequences)

    if not missing_config and not absent_csv:
        report.ok(
            "Las secuencias normales del CSV coinciden con "
            "la configuración de preparación."
        )
    else:
        report.fail(
            "La configuración y las secuencias del CSV difieren. "
            f"CSV sin configuración={missing_config}; "
            f"configuración sin CSV={absent_csv}"
        )

    # Verificar que las referencias estén declaradas como None.
    reference_rows = [
        row for row in csv_rows
        if is_reference_row(row)
    ]

    if reference_rows:
        if "ref" in expected_references:
            report.ok(
                "La configuración declara ref=None: "
                "las referencias no pertenecen a train ni test."
            )
        else:
            report.fail(
                "Hay referencias en el CSV, pero no se encontró "
                "la entrada ref=None en SEQUENCES_BY_VERSION."
            )

        reference_sequences = {
            normalize_csv_sequence(row)
            for row in reference_rows
        }

        leaked = sorted(
            reference_sequences.intersection(
                actual_train | actual_test
            )
        )

        if leaked:
            report.fail(
                "Referencias incluidas en los archivos de split: "
                + repr(leaked)
            )
        else:
            report.ok(
                f"Las {len(reference_rows)} referencias del CSV "
                "están excluidas de TrainSplit.txt y TestSplit.txt."
            )

        if all(row["split"] == "train" for row in reference_rows):
            report.info(
                "Las referencias conservan split=train en el CSV "
                "fuente, pero están excluidas de los splits MapNet."
            )

    else:
        report.info(
            "No hay referencias en el CSV para comprobar exclusión."
        )

    # La configuración no debe tener secuencias duplicadas.
    configured_all = (
        list(expected_train)
        + list(expected_test)
        + list(expected_references)
    )

    if len(configured_all) == len(set(configured_all)):
        report.ok(
            "La configuración no contiene secuencias duplicadas."
        )


# ---------------------------------------------------------------------
# Static source audit
# ---------------------------------------------------------------------

def read_source(path: Path) -> Optional[str]:
    if not path.is_file():
        return None

    return path.read_text(encoding="utf-8")


def extract_function(
    source: str,
    function_name: str,
) -> str:
    """Extrae una función o método respetando su indentación."""
    pattern = re.compile(
        rf"(?m)^([ \t]*)def\s+{re.escape(function_name)}\s*\("
    )

    match = pattern.search(source)

    if not match:
        return ""

    indentation = match.group(1)
    start = match.start()

    next_definition = re.search(
        rf"(?m)^{re.escape(indentation)}(?:def|class)\s+\w+",
        source[match.end():],
    )

    end = (
        match.end() + next_definition.start()
        if next_definition
        else len(source)
    )

    return source[start:end]


def check_source_file(
    report: Report,
    path: Path,
) -> Optional[str]:

    source = read_source(path)

    if source is None:
        report.fail(f"No se encontró código fuente: {path}")
        return None

    report.ok(
        f"Fuente disponible: {path.relative_to(REPO_ROOT)}"
    )

    return source


def verify_prepare_source(
    report: Report,
    source: Optional[str],
):
    report.header("PREPARACIÓN DEL DATASET")

    if not source:
        return

    checks = [
        (
            "SEQUENCES_BY_VERSION",
            "Configuración de secuencias por versión.",
        ),
        (
            "R_BODY_FROM_CAMERA",
            "Conversión de cámara óptica a cuerpo FLU.",
        ),
        (
            "quaternion_xyzw_to_rotation",
            "Conversión quaternion xyzw a rotación.",
        ),
        (
            "make_camera_to_world_pose",
            "Construcción de matriz C2W de cámara.",
        ),
        (
            "T_c2w",
            "Uso explícito de matriz C2W.",
        ),
        (
            "reshape",
            "Aplanado/reestructuración de matriz de pose.",
        ),
        (
            "write_split_files",
            "Escritura de archivos de split.",
        ),
    ]

    for token, description in checks:
        if token in source:
            report.ok(description)
        else:
            report.fail(
                f"No se encontró en prepare_bebop.py: {token}"
            )

    function = extract_function(
        source,
        "make_camera_to_world_pose",
    )

    if function:
        required = (
            "quaternion_xyzw_to_rotation",
            "R_BODY_FROM_CAMERA",
            "T_wc",
        )

        missing = [
            item for item in required
            if item not in function
        ]

        if not missing:
            report.ok(
                "make_camera_to_world_pose() construye "
                "la orientación de cámara a partir del cuerpo "
                "y R_BODY_FROM_CAMERA."
            )
        else:
            report.fail(
                "No se confirmó la construcción C2W en "
                "make_camera_to_world_pose(): "
                + repr(missing)
            )

        if re.search(
            r"np\.linalg\.inv|torch\.inverse",
            function,
        ):
            report.fail(
                "make_camera_to_world_pose() contiene "
                "una inversión matricial."
            )
        else:
            report.ok(
                "make_camera_to_world_pose() no invierte "
                "la matriz de pose."
            )

    else:
        report.fail(
            "No se encontró make_camera_to_world_pose()."
        )


def verify_active_source_pipeline(report: Report):
    report.header("AUDITORÍA ESTÁTICA DEL CÓDIGO ACTIVO")

    sources = {
        "prepare": check_source_file(report, PREPARE),
        "pose_utils": check_source_file(report, POSE_UTILS),
        "bebop_loader": check_source_file(report, BEBOP_LOADER),
        "composite": check_source_file(report, COMPOSITE_LOADER),
        "criterion": check_source_file(report, CRITERION),
        "model": check_source_file(report, MODEL),
        "train": check_source_file(report, TRAIN),
        "eval": check_source_file(report, EVAL),
    }

    # 1. Preparación.
    verify_prepare_source(
        report,
        sources["prepare"],
    )

    # 2. process_poses.
    source = sources["pose_utils"]

    if source:
        function = extract_function(
            source,
            "process_poses",
        )

        if not function:
            report.fail("No se encontró process_poses().")
        else:
            required = (
                "poses_in[i].reshape((3, 4))[:3, :3]",
                "txq.mat2quat(np.dot(align_R, R))",
                "qlog(q)",
            )

            missing = [
                item for item in required
                if item not in function
            ]

            if missing:
                report.fail(
                    "process_poses() no contiene las operaciones "
                    "esperadas: " + repr(missing)
                )
            else:
                report.ok(
                    "process_poses() reconstruye R 3x4, aplica "
                    "align_R y convierte a log-quaternion."
                )

            if re.search(
                r"np\.linalg\.inv|torch\.inverse|np\.transpose",
                function,
            ):
                report.fail(
                    "process_poses() contiene una inversión "
                    "o transposición explícita."
                )
            else:
                report.ok(
                    "process_poses() no invierte ni transpone "
                    "las matrices de pose."
                )

            if "poses_in[:, [3, 7, 11]]" in function:
                report.ok(
                    "process_poses() extrae la traslación C2W "
                    "de los índices [3, 7, 11]."
                )
            else:
                report.fail(
                    "No se confirmó la extracción de traslación "
                    "C2W en process_poses()."
                )

    # 3. Loader Bebop.
    source = sources["bebop_loader"]

    if source:
        function = extract_function(
            source,
            "_read_poses_file",
        )

        required = (
            "filenames.append(fields[0])",
            "poses.append(pose)",
            "return filenames, poses",
        )

        missing = [
            item for item in required
            if item not in function
        ]

        if function and not missing:
            report.ok(
                "_read_poses_file() conserva los 12 valores "
                "de cada matriz sin invertir ni transponer."
            )
            report.info(
                "El reshape(3,4) se realiza posteriormente "
                "en process_poses()."
            )
        else:
            report.fail(
                "No se confirmó la lectura directa de poses: "
                + repr(missing)
            )

        if re.search(
            r"np\.linalg\.inv|torch\.inverse|np\.transpose|\.T\b",
            function,
        ):
            report.fail(
                "_read_poses_file() contiene una inversión "
                "o transposición explícita."
            )
        else:
            report.ok(
                "El lector Bebop no invierte ni transpone poses."
            )

        if "len(fields) != 13" in function:
            report.ok(
                "El loader valida el nombre y los 12 valores "
                "de cada matriz."
            )

    # 4. Composite.
    source = sources["composite"]

    if source:
        if (
            "_build_bebop_centers" in source
            and "sequence_ranges" in source
        ):
            report.ok(
                "Composite construye centros Bebop respetando "
                "los límites de secuencia."
            )
        else:
            report.fail(
                "No se confirmó el control de límites "
                "entre secuencias en composite.py."
            )

        if "torch.stack" in source:
            report.ok(
                "Composite apila las poses para formar clips."
            )
        else:
            report.fail(
                "No se detectó el apilado de poses en composite.py."
            )

    # 5. Criterion.
    source = sources["criterion"]

    if source:
        if "calc_vos_simple" in source:
            report.ok(
                "MapNetCriterion utiliza calc_vos_simple()."
            )
        else:
            report.fail(
                "No se detectó calc_vos_simple() en criterion.py."
            )

        if "abs_loss" in source:
            report.ok(
                "Criterion contiene el término de pose absoluta."
            )
        else:
            report.fail(
                "No se encontró el término de pose absoluta."
            )

    # 6. Modelo.
    source = sources["model"]

    if source:
        for token, description in (
            ("fc_xyz", "cabeza de traslación"),
            ("fc_wpqr", "cabeza de orientación log-quaternion"),
        ):
            if token in source:
                report.ok(
                    f"Modelo contiene {description}: {token}."
                )
            else:
                report.fail(
                    f"No se detectó {token} en posenet.py."
                )

    # 7. Entrenamiento.
    source = sources["train"]

    if source:
        if "torch.from_numpy(x).float()" in source:
            report.ok(
                "train.py convierte las etiquetas a tensor float."
            )
        else:
            report.fail(
                "No se encontró la conversión esperada "
                "de target_transform."
            )

        if (
            "MapNetCriterion" in source
            and "srq = section.getfloat('gamma')" in source
        ):
            report.ok(
                "train.py asigna gamma a srq para MapNetCriterion."
            )
        else:
            report.fail(
                "No se confirmó la asignación gamma -> srq."
            )

        if (
            "MF(train=True" in source
            and "MF(train=False" in source
        ):
            report.ok(
                "train.py construye MF por separado para train "
                "y validation."
            )
        else:
            report.fail(
                "No se confirmó la construcción de MF train/val."
            )

    # 8. Evaluación.
    source = sources["eval"]

    if source:
        required = (
            "qexp(p[3:])",
            "(output[:, :3] * pose_s) + pose_m",
            "(target[:, :3] * pose_s) + pose_m",
            "data_set.get_indices(batch_idx)",
        )

        missing = [
            item for item in required
            if item not in source
        ]

        if not missing:
            report.ok(
                "eval.py expande log-quaternion, desnormaliza "
                "traslaciones y recupera el índice central del clip."
            )
        else:
            report.fail(
                "Faltan operaciones esperadas en eval.py: "
                + repr(missing)
            )

        if re.search(
            r"np\.linalg\.inv|torch\.inverse",
            source,
        ):
            report.fail(
                "eval.py contiene una inversión matricial explícita. "
                "Revisar su contexto geométrico."
            )
        else:
            report.ok(
                "eval.py no contiene inversión matricial explícita."
            )

        if "zip(x.T, y.T, z.T)" in source:
            report.ok(
                "La transposición x.T/y.T/z.T se limita "
                "a organizar las trayectorias para visualización."
            )

    report.info(
        "Los evaluadores antiguos excluidos de esta auditoría "
        "no se han leído ni utilizado."
    )


# ---------------------------------------------------------------------
# Version execution
# ---------------------------------------------------------------------

def verify_version(
    version: str,
    raw_root: Path,
    processed_root: Path,
    position_tolerance: float,
    angle_tolerance_deg: float,
    prepare_path: Path,
):
    report = Report(version)

    report.header(
        f"VERIFICACIÓN C2W MAPNET - {version}"
    )

    csv_path = find_source_csv(
        raw_root,
        version,
    )

    version_dir = find_processed_version(
        processed_root,
        version,
    )

    if csv_path is None:
        report.fail(
            f"No se encontró poses_split.csv para {version}. "
            f"Raíz consultada: {raw_root}"
        )
        report.summary()
        return report

    if version_dir is None:
        report.fail(
            f"No se encontró el dataset procesado de {version}. "
            f"Raíz consultada: {processed_root}"
        )
        report.summary()
        return report

    report.ok(f"CSV fuente: {csv_path}")
    report.ok(f"Dataset procesado: {version_dir}")

    try:
        csv_rows, columns = read_source_csv(csv_path)

        report.ok(
            f"CSV leído correctamente: {len(csv_rows)} filas útiles."
        )

        report.info(
            f"Columnas detectadas: {columns}"
        )

    except Exception as exc:
        report.fail(f"Error leyendo CSV: {exc}")
        report.summary()
        return report

    try:
        poses_files, pose_records = collect_processed_poses(
            version_dir
        )
    except Exception as exc:
        report.fail(f"Error leyendo poses.txt: {exc}")
        report.summary()
        return report

    if not poses_files:
        report.fail("No se encontraron archivos poses.txt.")
        report.summary()
        return report

    report.ok(
        f"Archivos poses.txt encontrados: {len(poses_files)}"
    )

    report.info(
        f"Registros de poses encontrados: {len(pose_records)}"
    )

    verify_pose_file_structure(
        report,
        pose_records,
    )

    compare_csv_and_poses(
        report,
        csv_rows,
        pose_records,
        position_tolerance,
        angle_tolerance_deg,
    )

    verify_csv_splits(
        report,
        csv_rows,
    )

    verify_processed_splits(
        report,
        version,
        version_dir,
        csv_rows,
        prepare_path,
    )

    report.summary()
    return report


def discover_versions(
    raw_root: Path,
    processed_root: Path,
) -> list[str]:

    versions = set()

    for root in (raw_root, processed_root):
        if not root.is_dir():
            continue

        for child in root.iterdir():
            if not child.is_dir():
                continue

            if (
                (child / "poses_split.csv").is_file()
                or (child / "poses.txt").is_file()
                or list(child.rglob("poses.txt"))
            ):
                versions.add(child.name)

    return sorted(versions)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verifica el flujo C2W de MapNet "
            "para varias versiones Bebop."
        )
    )

    selection = parser.add_mutually_exclusive_group(
        required=True
    )

    selection.add_argument(
        "--versions",
        nargs="+",
        help="Versiones a verificar, por ejemplo: v2_2 v2_3.",
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Descubre y verifica las versiones disponibles.",
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )

    parser.add_argument(
        "--prepare",
        type=Path,
        default=PREPARE,
        help="Ruta a src/data/prepare_bebop.py.",
    )

    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=1e-5,
        help="Tolerancia absoluta de posición en metros.",
    )

    parser.add_argument(
        "--angle-tolerance-deg",
        type=float,
        default=1e-3,
        help="Tolerancia angular en grados.",
    )

    parser.add_argument(
        "--skip-source-audit",
        action="store_true",
        help="Omite las comprobaciones estáticas del código MapNet.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 76)
    print("VERIFICADOR C2W - MAPNET / BEBOP")
    print("=" * 76)

    print(f"Repositorio:      {REPO_ROOT}")
    print(f"Raw root:         {args.raw_root}")
    print(f"Processed root:   {args.processed_root}")
    print(f"Preparación:      {args.prepare}")
    print(f"Tolerancia pos.:  {args.position_tolerance:g} m")
    print(f"Tolerancia rot.:  {args.angle_tolerance_deg:g} grados")
    print("Modo:             SOLO LECTURA")

    if args.all:
        versions = discover_versions(
            args.raw_root,
            args.processed_root,
        )
    else:
        versions = args.versions

    if not versions:
        print("\n[FAIL] No se encontraron versiones para verificar.")
        return 2

    print(f"Versiones: {', '.join(versions)}")

    reports = []

    for version in versions:
        reports.append(
            verify_version(
                version=version,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                position_tolerance=args.position_tolerance,
                angle_tolerance_deg=args.angle_tolerance_deg,
                prepare_path=args.prepare,
            )
        )

    if not args.skip_source_audit:
        source_report = Report("CÓDIGO ACTIVO")

        verify_active_source_pipeline(
            source_report
        )

        source_report.summary()
        reports.append(source_report)

    total_pass = sum(
        item.passed for item in reports
    )

    total_fail = sum(
        item.failed for item in reports
    )

    total_warn = sum(
        item.warnings for item in reports
    )

    print("\n" + "=" * 76)
    print("RESUMEN GLOBAL")
    print("=" * 76)

    print(f"Versiones procesadas: {len(versions)}")
    print(f"PASS: {total_pass}")
    print(f"FAIL: {total_fail}")
    print(f"WARN: {total_warn}")

    for item in reports:
        if item.version != "CÓDIGO ACTIVO":
            print(
                f"  {item.version}: "
                f"{item.geometry_classification}"
            )

    all_geometry_c2w = all(
        item.geometry_classification == "C2W CONFIRMADO"
        for item in reports
        if item.version != "CÓDIGO ACTIVO"
    )

    source_audited = (
        args.skip_source_audit
        or any(
            item.version == "CÓDIGO ACTIVO"
            for item in reports
        )
    )

    if total_fail:
        print(
            "\nRESULTADO: FALLÓ. "
            "Revisar las comprobaciones [FAIL]."
        )
        return 1

    if total_warn:
        print(
            "\nRESULTADO: SIN FALLOS, PERO CON ADVERTENCIAS."
        )
        return 0

    if all_geometry_c2w and source_audited:
        print(
            "\nRESULTADO: C2W END-TO-END VERIFICADO "
            "PARA LAS VERSIONES SELECCIONADAS."
        )
    elif all_geometry_c2w:
        print(
            "\nRESULTADO: DATOS C2W VERIFICADOS. "
            "La auditoría estática del código fue omitida."
        )
    else:
        print(
            "\nRESULTADO: TODAS LAS PRUEBAS PASARON, "
            "PERO NO TODAS LAS VERSIONES FUERON CLASIFICADAS C2W."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
