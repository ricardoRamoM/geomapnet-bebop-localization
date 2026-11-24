#!/usr/bin/env python3
"""
scripts/test_bebop_mapnet.py

Prueba de integración de MapNet con el dataset Parrot Bebop.

Comprueba:
1. Carga del dataset Bebop mediante MF.
2. Dimensiones y tipos de los tensores.
3. Forward pass de ResNet34 + MapNet.
4. Cálculo de MapNetCriterion.
5. Backward pass y validez de los gradientes.

No realiza entrenamiento ni guarda checkpoints.
"""

import os
import sys

# Añadir la raíz del repositorio al path de Python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import set_paths

import os.path as osp

import configparser
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights

from models.posenet import PoseNet, MapNet
from common.criterion import MapNetCriterion
from dataset_loaders.composite import MF


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = "Bebop"
SCENE = "bebop"

CONFIG_FILE = "scripts/configs/bebop_mapnet.ini"

BATCH_SIZE = 2
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():

    print("\n========================================")
    print("   TEST DE INTEGRACIÓN MAPNET - BEBOP")
    print("========================================\n")

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # 1. Leer configuración
    # --------------------------------------------------------

    check(
        osp.isfile(CONFIG_FILE),
        f"No existe el archivo de configuración: {CONFIG_FILE}"
    )

    settings = configparser.ConfigParser()
    settings.read(CONFIG_FILE)

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

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[1/6] Configuración cargada")
    print(f"  Seed: {seed}")
    print(f"  Skip: {skip}")
    print(f"  Steps: {steps}")
    print(f"  Beta: {beta}")
    print(f"  Gamma: {gamma}")

    # --------------------------------------------------------
    # 2. Cargar estadísticas RGB
    # --------------------------------------------------------

    stats_file = osp.join(
        "..",
        "data",
        "deepslam_data",
        DATASET,
        SCENE,
        "stats.txt"
    )

    check(
        osp.isfile(stats_file),
        f"No existe stats.txt: {stats_file}"
    )

    stats = np.loadtxt(stats_file)

    check(
        stats.shape == (2, 3),
        f"Dimensiones inesperadas de stats.txt: {stats.shape}"
    )

    print("\n[2/6] Estadísticas RGB")
    print(f"  Archivo: {stats_file}")
    print(f"  Mean: {stats[0]}")
    print(f"  Variance: {stats[1]}")

    # --------------------------------------------------------
    # 3. Transformaciones
    # --------------------------------------------------------

    tforms = [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=stats[0],
            std=np.sqrt(stats[1])
        )
    ]

    data_transform = transforms.Compose(tforms)

    target_transform = transforms.Lambda(
        lambda x: torch.from_numpy(x).float()
    )

    # --------------------------------------------------------
    # 4. Cargar dataset
    # --------------------------------------------------------

    data_dir = osp.join(
        "..",
        "data",
        "deepslam_data",
        DATASET
    )

    dataset_kwargs = dict(
        scene=SCENE,
        data_path=data_dir,
        transform=data_transform,
        target_transform=target_transform,
        seed=seed,
        dataset=DATASET,
        skip=skip,
        steps=steps,
        variable_skip=variable_skip
    )

    train_set = MF(
        train=True,
        real=real,
        **dataset_kwargs
    )

    val_set = MF(
        train=False,
        real=real,
        **dataset_kwargs
    )

    print("\n[3/6] Dataset cargado")
    print(f"  Train clips: {len(train_set)}")
    print(f"  Validation clips: {len(val_set)}")

    check(len(train_set) > 0, "Train dataset vacío.")
    check(len(val_set) > 0, "Validation dataset vacío.")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # 5. Comprobar batch
    # --------------------------------------------------------

    images, targets = next(iter(train_loader))

    print("\n[4/6] Comprobación del batch")

    print(f"  Images shape: {tuple(images.shape)}")
    print(f"  Images dtype: {images.dtype}")
    print(f"  Targets shape: {tuple(targets.shape)}")
    print(f"  Targets dtype: {targets.dtype}")

    expected_images = (BATCH_SIZE, steps, 3)

    check(
        images.ndim == 5,
        f"Se esperaban imágenes [B,T,C,H,W], recibido: {images.shape}"
    )

    check(
        tuple(images.shape[:3]) == expected_images,
        f"Dimensiones de imágenes inesperadas: {images.shape}"
    )

    check(
        targets.shape == (BATCH_SIZE, steps, 6),
        f"Dimensiones de poses inesperadas: {targets.shape}"
    )

    check(
        torch.isfinite(images).all().item(),
        "Las imágenes contienen NaN o Inf."
    )

    check(
        torch.isfinite(targets).all().item(),
        "Las poses contienen NaN o Inf."
    )

    images = images.to(DEVICE)
    targets = targets.to(DEVICE)

    # --------------------------------------------------------
    # 6. Construir ResNet34 + MapNet
    # --------------------------------------------------------

    print("\n[5/6] Construyendo modelo")

    feature_extractor = models.resnet34(
        weights=ResNet34_Weights.DEFAULT
    )

    posenet = PoseNet(
        feature_extractor,
        droprate=dropout,
        pretrained=True,
        filter_nans=False
    )

    model = MapNet(mapnet=posenet)
    model = model.to(DEVICE)

    criterion = MapNetCriterion(
        sax=0.0,
        saq=beta,
        srx=0.0,
        srq=gamma,
        learn_beta=False,
        learn_gamma=False
    ).to(DEVICE)

    # --------------------------------------------------------
    # 7. Forward pass
    # --------------------------------------------------------

    model.train()
    model.zero_grad(set_to_none=True)

    predictions = model(images)

    print("\nForward pass")
    print(f"  Prediction shape: {tuple(predictions.shape)}")

    check(
        isinstance(predictions, torch.Tensor),
        "La salida de MapNet no es un tensor."
    )

    check(
        predictions.shape == targets.shape,
        (
            f"Predicciones {predictions.shape} "
            f"no coinciden con targets {targets.shape}"
        )
    )

    check(
        torch.isfinite(predictions).all().item(),
        "Las predicciones contienen NaN o Inf."
    )

    # --------------------------------------------------------
    # 8. Calcular pérdida
    # --------------------------------------------------------

    loss = criterion(predictions, targets)

    print("\nMapNetCriterion")
    print(f"  Loss: {loss.item():.6f}")

    check(
        loss.numel() == 1,
        f"La pérdida debería contener un único valor. Shape: {loss.shape}"
    )

    check(
        torch.isfinite(loss).item(),
        "La pérdida contiene NaN o Inf."
    )

    # --------------------------------------------------------
    # 9. Backward pass
    # --------------------------------------------------------

    loss.backward()

    gradients_found = 0
    invalid_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients_found += 1

            if not torch.isfinite(parameter.grad).all().item():
                invalid_gradients += 1

    print("\nBackward pass")
    print(f"  Parámetros con gradiente: {gradients_found}")
    print(f"  Gradientes inválidos: {invalid_gradients}")

    check(
        gradients_found > 0,
        "No se encontraron gradientes."
    )

    check(
        invalid_gradients == 0,
        "Se encontraron gradientes NaN o Inf."
    )

    # --------------------------------------------------------
    # 10. Comprobación rápida de validación
    # --------------------------------------------------------

    model.eval()

    val_images, val_targets = next(iter(val_loader))

    val_images = val_images.to(DEVICE)
    val_targets = val_targets.to(DEVICE)

    with torch.no_grad():

        val_predictions = model(val_images)
        val_loss = criterion(val_predictions, val_targets)

    print("\nValidación")
    print(f"  Images: {tuple(val_images.shape)}")
    print(f"  Targets: {tuple(val_targets.shape)}")
    print(f"  Predictions: {tuple(val_predictions.shape)}")
    print(f"  Loss: {val_loss.item():.6f}")

    check(
        val_predictions.shape == val_targets.shape,
        "Las dimensiones de validación no coinciden."
    )

    check(
        torch.isfinite(val_loss).item(),
        "La pérdida de validación contiene NaN o Inf."
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print("\n========================================")
    print("       TODAS LAS PRUEBAS SUPERADAS")
    print("========================================")

    print("Dataset Bebop: OK")
    print("ResNet34: OK")
    print("MapNet forward: OK")
    print("MapNetCriterion: OK")
    print("Backward pass: OK")
    print("Validación: OK")

    print("\nEl modelo está listo para pasar a la etapa")
    print("de entrenamiento formal.\n")


if __name__ == "__main__":
    main()