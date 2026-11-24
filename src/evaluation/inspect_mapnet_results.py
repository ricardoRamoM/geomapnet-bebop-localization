#!/usr/bin/env python3
# inspect_mapnet_results.py
import pickle
from pathlib import Path

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

for experiment in EXPERIMENTS:
    pkl_path = (
        BASE_DIR
        / "eval_results"
        / experiment
        / "Bebop_bebop_mapnet.pkl"
    )

    print("\n" + "=" * 65)
    print(f"EXPERIMENTO: {experiment}")
    print(f"ARCHIVO: {pkl_path}")

    if not pkl_path.exists():
        print("ERROR: archivo no encontrado")
        continue

    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    for key, value in data.items():
        print(f"\n{key}:")
        print(f"  Tipo: {type(value)}")
        print(f"  Shape: {getattr(value, 'shape', 'N/A')}")

        if hasattr(value, "__len__"):
            print(f"  Longitud: {len(value)}")

        if key == "valid_indices":
            print(f"  Primeros índices: {value[:10]}")
        elif key in ("pred_poses", "targ_poses"):
            print(f"  Primera pose: {value[0]}")
