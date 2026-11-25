
#!/usr/bin/env python3
"""
test_bebop.py

Pruebas del dataset Parrot Bebop y de integración con MapNet.

PRUEBA 1 - DATASET (por defecto)
    - Carga de TrainSplit y TestSplit.
    - Verificación de secuencias y cantidades.
    - Validación de poses C2W.
    - Validación de matrices de rotación.
    - Validación de pose_stats.txt.
    - Estadísticas compartidas entre train y test.

PRUEBA 2 - INTEGRACIÓN MAPNET (opcional: --integration)
    - Lectura de configuración.
    - Estadísticas RGB.
    - Carga de train y validation mediante MF.
    - Comprobación de batches.
    - Forward pass de ResNet34 + MapNet.
    - MapNetCriterion.
    - Backward pass y gradientes.
    - Forward pass de validación.

No realiza entrenamiento ni guarda checkpoints.

Ejemplos:

1. Comprobar una nueva versión del dataset:

    python src/tests/test_bebop.py --scene v2_3

2. Comprobar también las cantidades esperadas:

    python src/tests/test_bebop.py \
        --scene v2_3 \
        --expected-train 1497 \
        --expected-val 588

3. Ejecutar también la integración MapNet:

    python src/tests/test_bebop.py \
        --scene v2_3 \
        --integration \
        --config configs/Bebop/v2_3/bebop_mapnet_baseline.ini

4. Cambiar el tamaño del batch de integración:

    python src/tests/test_bebop.py \
        --scene v2_3 \
        --integration \
        --batch-size 2

IMPORTANTE:
- Los datos se leen desde /data/research/bebop_data.
- poses.txt debe contener matrices C2W.
- Las poses ya están alineadas a FLU.
- No debe aplicarse una segunda alineación.
"""

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
GEOMAPNET_DIR = ROOT / "geomapnet"

# Ubicación real de los datasets.
DATA_ROOT = Path("/data/research/bebop_data")

# Rutas de GeoMapNet.
sys.path.insert(0, str(GEOMAPNET_DIR))
sys.path.insert(0, str(GEOMAPNET_DIR / "scripts"))

from dataset_loaders.bebop import Bebop


# ============================================================
# UTILIDADES
# ============================================================

def check(condition, message):
    """Lanza un error si una condición no se cumple."""
    if not condition:
        raise RuntimeError(message)


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# PRUEBA 1: LOADER Y DATASET C2W
# ============================================================

def test_dataset(args):

    print_section("PRUEBA 1: DATASET BEBOP")

    scene_root = DATA_ROOT / args.scene

    check(
        scene_root.is_dir(),
        f"No se encontró el dataset: {scene_root}",
    )

    print("Repositorio:", ROOT)
    print("Dataset:", scene_root)
    print("Convención esperada: C2W")
    print("Alineación esperada: FLU, sin realineación adicional")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_set = Bebop(
        scene=args.scene,
        data_path=str(DATA_ROOT),
        train=True,
        transform=None,
        skip_images=True,
    )

    print("\n--- TRAIN ---")
    print("Imágenes:", len(train_set))
    print("Secuencias:", train_set.sequence_names)
    print("Rangos:", train_set.sequence_ranges)
    print("Pose shape:", train_set.poses.shape)
    print("mean_t:", train_set.mean_t)
    print("std_t:", train_set.std_t)

    check(
        len(train_set) > 0,
        "El dataset de entrenamiento está vacío.",
    )

    if args.expected_train is not None:
        check(
            len(train_set) == args.expected_train,
            (
                f"Se esperaban {args.expected_train} imágenes "
                f"de entrenamiento y se encontraron {len(train_set)}."
            ),
        )

    # --------------------------------------------------------
    # TEST / VALIDATION
    # --------------------------------------------------------

    test_set = Bebop(
        scene=args.scene,
        data_path=str(DATA_ROOT),
        train=False,
        transform=None,
        skip_images=True,
    )

    print("\n--- TEST / VALIDATION ---")
    print("Imágenes:", len(test_set))
    print("Secuencias:", test_set.sequence_names)
    print("Rangos:", test_set.sequence_ranges)
    print("Pose shape:", test_set.poses.shape)

    check(
        len(test_set) > 0,
        "El dataset de validación está vacío.",
    )

    if args.expected_val is not None:
        check(
            len(test_set) == args.expected_val,
            (
                f"Se esperaban {args.expected_val} imágenes "
                f"de validación y se encontraron {len(test_set)}."
            ),
        )

    # --------------------------------------------------------
    # DIMENSIONES Y VALORES DE POSES PROCESADAS
    # --------------------------------------------------------

    check(
        train_set.poses.ndim == 2
        and train_set.poses.shape[1] == 6,
        f"Dimensiones inesperadas de train_set.poses: "
        f"{train_set.poses.shape}",
    )

    check(
        test_set.poses.ndim == 2
        and test_set.poses.shape[1] == 6,
        f"Dimensiones inesperadas de test_set.poses: "
        f"{test_set.poses.shape}",
    )

    check(
        np.isfinite(train_set.poses).all(),
        "Las poses procesadas de train contienen NaN o Inf.",
    )

    check(
        np.isfinite(test_set.poses).all(),
        "Las poses procesadas de test contienen NaN o Inf.",
    )

    # --------------------------------------------------------
    # MATRICES C2W ORIGINALES
    # --------------------------------------------------------

    check(
        hasattr(train_set, "raw_poses"),
        "El loader no expone raw_poses. "
        "No se puede comprobar la estructura C2W.",
    )

    check(
        hasattr(test_set, "raw_poses"),
        "El loader de test no expone raw_poses.",
    )

    raw_train = np.asarray(train_set.raw_poses)
    raw_test = np.asarray(test_set.raw_poses)

    check(
        raw_train.shape == (len(train_set), 12),
        f"raw_poses train debe tener forma (N,12), "
        f"pero tiene {raw_train.shape}.",
    )

    check(
        raw_test.shape == (len(test_set), 12),
        f"raw_poses test debe tener forma (N,12), "
        f"pero tiene {raw_test.shape}.",
    )

    check(
        np.isfinite(raw_train).all(),
        "Las matrices C2W de train contienen NaN o Inf.",
    )

    check(
        np.isfinite(raw_test).all(),
        "Las matrices C2W de test contienen NaN o Inf.",
    )

    # Cada matriz 3x4 tiene esta estructura:
    #
    # R00 R01 R02 tx
    # R10 R11 R12 ty
    # R20 R21 R22 tz
    #
    # La última columna representa el centro de cámara C_w.

    def validate_c2w_matrices(raw_poses, split_name):

        rotations = raw_poses[:, [
            0, 1, 2,
            4, 5, 6,
            8, 9, 10,
        ]].reshape(-1, 3, 3)

        centers = raw_poses[:, [3, 7, 11]]

        identity = np.eye(3)

        orthogonality_errors = np.linalg.norm(
            np.matmul(
                np.transpose(rotations, (0, 2, 1)),
                rotations,
            ) - identity,
            axis=(1, 2),
        )

        determinants = np.linalg.det(rotations)

        max_orthogonality_error = float(
            np.max(orthogonality_errors)
        )

        max_determinant_error = float(
            np.max(np.abs(determinants - 1.0))
        )

        print(f"\n--- Validación C2W: {split_name} ---")
        print("Matrices:", len(raw_poses))
        print("Centros de cámara:", centers.shape)
        print(
            "Error máximo de ortogonalidad:",
            f"{max_orthogonality_error:.8f}",
        )
        print(
            "Error máximo de determinante:",
            f"{max_determinant_error:.8f}",
        )

        check(
            max_orthogonality_error < 1e-3,
            f"Las rotaciones de {split_name} no son ortogonales.",
        )

        check(
            max_determinant_error < 1e-3,
            f"Las rotaciones de {split_name} no tienen "
            f"determinante cercano a +1.",
        )

        check(
            np.isfinite(centers).all(),
            f"Los centros C2W de {split_name} no son finitos.",
        )

        return centers

    train_centers = validate_c2w_matrices(
        raw_train,
        "TRAIN",
    )

    test_centers = validate_c2w_matrices(
        raw_test,
        "TEST",
    )

    print("\nRango de centros de cámara C2W:")
    print("Train min:", train_centers.min(axis=0))
    print("Train max:", train_centers.max(axis=0))
    print("Test min :", test_centers.min(axis=0))
    print("Test max :", test_centers.max(axis=0))

    # --------------------------------------------------------
    # ESTADÍSTICAS DE POSICIÓN
    # --------------------------------------------------------

    stats_path = scene_root / "pose_stats.txt"

    check(
        stats_path.is_file(),
        f"No se encontró pose_stats.txt: {stats_path}",
    )

    stats = np.loadtxt(stats_path)

    check(
        stats.shape == (2, 3),
        f"pose_stats.txt debe tener forma (2,3), "
        f"pero tiene {stats.shape}.",
    )

    check(
        np.isfinite(stats).all(),
        "pose_stats.txt contiene NaN o Inf.",
    )

    check(
        np.all(stats[1] > 0),
        "Las desviaciones estándar deben ser positivas.",
    )

    check(
        np.allclose(train_set.mean_t, stats[0]),
        "mean_t del loader no coincide con pose_stats.txt.",
    )

    check(
        np.allclose(train_set.std_t, stats[1]),
        "std_t del loader no coincide con pose_stats.txt.",
    )

    check(
        np.allclose(train_set.mean_t, test_set.mean_t),
        "Train y test no comparten mean_t.",
    )

    check(
        np.allclose(train_set.std_t, test_set.std_t),
        "Train y test no comparten std_t.",
    )

    print("\n--- Estadísticas C2W ---")
    print("Archivo:", stats_path)
    print("Mean:", stats[0])
    print("Std :", stats[1])

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print_section("RESULTADO DATASET")

    print("Train:", len(train_set))
    print("Test :", len(test_set))
    print("Matrices C2W: OK")
    print("Rotaciones válidas: OK")
    print("Centros de cámara: OK")
    print("Estadísticas compartidas: OK")
    print("Dimensiones de poses: OK")

    print("\nPRUEBA DEL DATASET SUPERADA.")

    return train_set, test_set


# ============================================================
# PRUEBA 2: INTEGRACIÓN MAPNET
# ============================================================

def test_mapnet(args):

    import torch

    from torch.utils.data import DataLoader
    from torchvision import transforms, models
    from torchvision.models import ResNet34_Weights

    import set_paths

    from models.posenet import PoseNet, MapNet
    from common.criterion import MapNetCriterion
    from dataset_loaders.composite import MF

    print_section("PRUEBA 2: INTEGRACIÓN MAPNET")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    scene_root = DATA_ROOT / args.scene

    # --------------------------------------------------------
    # CONFIGURACIÓN
    # --------------------------------------------------------

    if args.config is None:

        config_path = (
            ROOT
            / "configs"
            / "Bebop"
            / args.scene
            / "bebop_mapnet.ini"
        )

    else:

        config_path = Path(args.config).expanduser()

        if not config_path.is_absolute():
            config_path = ROOT / config_path

    config_path = config_path.resolve()

    check(
        config_path.is_file(),
        f"No existe la configuración: {config_path}",
    )

    settings = configparser.ConfigParser()
    settings.read(config_path)

    check(
        "training" in settings
        and "hyperparameters" in settings,
        "La configuración necesita training y hyperparameters.",
    )

    training = settings["training"]
    hyper = settings["hyperparameters"]

    seed = training.getint("seed")

    dropout = hyper.getfloat("dropout")
    beta = hyper.getfloat("beta")
    gamma = hyper.getfloat("gamma")

    skip = hyper.getint("skip")
    steps = hyper.getint("steps")
    real = hyper.getboolean("real")
    variable_skip = hyper.getboolean("variable_skip")

    check(steps > 0, "steps debe ser positivo.")
    check(skip > 0, "skip debe ser positivo.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("\nConfiguración:", config_path)
    print("Seed:", seed)
    print("Skip:", skip)
    print("Steps:", steps)
    print("Dropout:", dropout)
    print("Beta:", beta)
    print("Gamma:", gamma)
    print("Real:", real)
    print("Variable skip:", variable_skip)

    # --------------------------------------------------------
    # ESTADÍSTICAS RGB
    # --------------------------------------------------------

    stats_file = scene_root / "stats.txt"

    check(
        stats_file.is_file(),
        f"No existe stats.txt: {stats_file}",
    )

    rgb_stats = np.loadtxt(stats_file)

    check(
        rgb_stats.shape == (2, 3),
        f"stats.txt debe tener forma (2,3), "
        f"pero tiene {rgb_stats.shape}.",
    )

    check(
        np.isfinite(rgb_stats).all(),
        "stats.txt contiene NaN o Inf.",
    )

    check(
        np.all(rgb_stats[1] > 0),
        "Las varianzas RGB deben ser positivas.",
    )

    rgb_mean = rgb_stats[0]
    rgb_std = np.sqrt(rgb_stats[1])

    print("\nEstadísticas RGB")
    print("Mean:", rgb_mean)
    print("Variance:", rgb_stats[1])
    print("Std:", rgb_std)

    # --------------------------------------------------------
    # TRANSFORMACIONES
    # --------------------------------------------------------

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=rgb_mean,
            std=rgb_std,
        ),
    ])

    target_transform = transforms.Lambda(
        lambda x: torch.from_numpy(x).float()
    )

    # --------------------------------------------------------
    # DATASET MF
    # --------------------------------------------------------

    dataset_kwargs = dict(
        scene=args.scene,
        data_path=str(DATA_ROOT),
        transform=data_transform,
        target_transform=target_transform,
        seed=seed,
        dataset="Bebop",
        skip=skip,
        steps=steps,
        variable_skip=variable_skip,
    )

    train_set = MF(
        train=True,
        real=real,
        **dataset_kwargs,
    )

    val_set = MF(
        train=False,
        real=real,
        **dataset_kwargs,
    )

    print("\nTrain clips:", len(train_set))
    print("Validation clips:", len(val_set))

    check(len(train_set) > 0, "Train MF está vacío.")
    check(len(val_set) > 0, "Validation MF está vacío.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    images, targets = next(iter(train_loader))

    print("\nBatch de entrenamiento")
    print("Images:", tuple(images.shape))
    print("Targets:", tuple(targets.shape))
    print("Images dtype:", images.dtype)
    print("Targets dtype:", targets.dtype)

    check(
        images.ndim == 5,
        f"Se esperaban imágenes [B,T,C,H,W], recibido: "
        f"{images.shape}",
    )

    check(
        tuple(images.shape[1:3]) == (steps, 3),
        f"Dimensiones de imágenes inesperadas: {images.shape}",
    )

    check(
        tuple(targets.shape) == (images.shape[0], steps, 6),
        f"Dimensiones de poses inesperadas: {targets.shape}",
    )

    check(
        images.dtype == torch.float32,
        f"Tipo de imágenes inesperado: {images.dtype}",
    )

    check(
        targets.dtype == torch.float32,
        f"Tipo de targets inesperado: {targets.dtype}",
    )

    check(
        torch.isfinite(images).all().item(),
        "Las imágenes contienen NaN o Inf.",
    )

    check(
        torch.isfinite(targets).all().item(),
        "Los targets contienen NaN o Inf.",
    )

    images = images.to(device)
    targets = targets.to(device)

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    print("\nConstruyendo ResNet34 + MapNet...")

    feature_extractor = models.resnet34(
        weights=ResNet34_Weights.DEFAULT
    )

    posenet = PoseNet(
        feature_extractor,
        droprate=dropout,
        pretrained=True,
        filter_nans=False,
    )

    model = MapNet(mapnet=posenet).to(device)

    criterion = MapNetCriterion(
        sax=0.0,
        saq=beta,
        srx=0.0,
        srq=gamma,
        learn_beta=False,
        learn_gamma=False,
    ).to(device)

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    print("\nEjecutando forward pass...")

    model.train()
    model.zero_grad(set_to_none=True)

    predictions = model(images)

    check(
        isinstance(predictions, torch.Tensor),
        "La salida de MapNet no es un tensor.",
    )

    check(
        predictions.shape == targets.shape,
        f"Predicciones {predictions.shape} "
        f"no coinciden con targets {targets.shape}.",
    )

    check(
        torch.isfinite(predictions).all().item(),
        "Las predicciones contienen NaN o Inf.",
    )

    loss = criterion(predictions, targets)

    print("Predictions:", tuple(predictions.shape))
    print(f"Loss: {loss.item():.6f}")

    check(
        loss.numel() == 1,
        "La pérdida no es escalar.",
    )

    check(
        torch.isfinite(loss).item(),
        "La pérdida contiene NaN o Inf.",
    )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    print("\nEjecutando backward pass...")

    loss.backward()

    gradients_found = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients_found += 1

            if not torch.isfinite(parameter.grad).all().item():
                invalid_gradients += 1

    print("Parámetros con gradiente:", gradients_found)
    print("Gradientes inválidos:", invalid_gradients)

    check(
        gradients_found > 0,
        "No se encontraron gradientes.",
    )

    check(
        invalid_gradients == 0,
        "Se encontraron gradientes NaN o Inf.",
    )

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    print("\nEjecutando validación...")

    model.eval()

    val_images, val_targets = next(iter(val_loader))

    val_images = val_images.to(device)
    val_targets = val_targets.to(device)

    with torch.no_grad():

        val_predictions = model(val_images)
        val_loss = criterion(val_predictions, val_targets)

    print("Validation images:", tuple(val_images.shape))
    print("Validation targets:", tuple(val_targets.shape))
    print("Validation predictions:", tuple(val_predictions.shape))
    print(f"Validation loss: {val_loss.item():.6f}")

    check(
        val_predictions.shape == val_targets.shape,
        "Las dimensiones de validación no coinciden.",
    )

    check(
        torch.isfinite(val_predictions).all().item(),
        "Las predicciones de validación contienen NaN o Inf.",
    )

    check(
        torch.isfinite(val_loss).item(),
        "La pérdida de validación contiene NaN o Inf.",
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print_section("RESULTADO INTEGRACIÓN MAPNET")

    print("Dataset MF: OK")
    print("Estadísticas RGB: OK")
    print("ResNet34: OK")
    print("MapNet forward: OK")
    print("MapNetCriterion: OK")
    print("Backward pass: OK")
    print("Gradientes: OK")
    print("Validación: OK")

    print("\nPRUEBA DE INTEGRACIÓN MAPNET SUPERADA.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Pruebas del dataset Bebop y MapNet."
    )

    parser.add_argument(
        "--scene",
        default="v2_3",
        help="Versión del dataset.",
    )

    parser.add_argument(
        "--expected-train",
        type=int,
        default=None,
        help="Cantidad esperada de imágenes train.",
    )

    parser.add_argument(
        "--expected-val",
        type=int,
        default=None,
        help="Cantidad esperada de imágenes test/validation.",
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Ejecuta también la prueba de integración MapNet.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Configuración MapNet relativa a la raíz o absoluta.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size para la prueba de integración.",
    )

    args = parser.parse_args()

    check(
        args.batch_size > 0,
        "batch-size debe ser positivo.",
    )

    # Siempre se comprueba el dataset.
    test_dataset(args)

    # La integración es opcional.
    if args.integration:
        test_mapnet(args)

    print_section("TODAS LAS PRUEBAS SOLICITADAS HAN TERMINADO")

    print("Dataset:", args.scene)
    print("Loader C2W: OK")

    if args.integration:
        print("Integración MapNet: OK")
    else:
        print("Integración MapNet: no solicitada")


if __name__ == "__main__":
    main()
