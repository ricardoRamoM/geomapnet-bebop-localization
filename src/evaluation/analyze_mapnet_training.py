
#!/usr/bin/env python3
# analyze_mapnet_training.py
"""
Analiza automáticamente el entrenamiento MapNet más reciente.

Busca:
    /data/research/mapnet/geomapnet/scripts/logs/

Genera:
    /data/research/mapnet/eval_results/training_analysis/<experimento>/

Uso:
    python src/evaluation/analyze_mapnet_training.py

    python src/evaluation/analyze_mapnet_training.py --list-all

    python src/evaluation/analyze_mapnet_training.py --experiment NOMBRE
"""

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

MAPNET_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = MAPNET_ROOT / "geomapnet" / "scripts" / "logs"
OUTPUT_ROOT = MAPNET_ROOT / "eval_results" / "training_analysis"


# ============================================================
# REGEX
# ============================================================

VAL_PATTERN = re.compile(
    r"Val .*?Epoch\s+(\d+),\s*val_loss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

CHECKPOINT_EPOCH_PATTERN = re.compile(
    r"(?:epoch|checkpoint)[_-]?(\d+)",
    re.IGNORECASE
)


# ============================================================
# DISCOVERY
# ============================================================

def discover_experiments():
    """Encuentra automáticamente directorios con log.txt."""

    if not LOGS_ROOT.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de logs: {LOGS_ROOT}"
        )

    experiments = []

    for log_path in LOGS_ROOT.rglob("log.txt"):
        experiments.append({
            "name": log_path.parent.name,
            "directory": log_path.parent,
            "log": log_path,
            "modified": log_path.stat().st_mtime,
        })

    return sorted(
        experiments,
        key=lambda item: item["modified"],
        reverse=True
    )


def select_experiment(experiments, requested_name=None):
    """Selecciona un experimento concreto o el más reciente."""

    if not experiments:
        raise RuntimeError(
            f"No se encontraron archivos log.txt en {LOGS_ROOT}"
        )

    if requested_name:
        matches = [
            exp for exp in experiments
            if exp["name"] == requested_name
        ]

        if not matches:
            available = "\n".join(
                f"  - {exp['name']}" for exp in experiments
            )
            raise ValueError(
                f"No se encontró '{requested_name}'.\n"
                f"Experimentos disponibles:\n{available}"
            )

        return matches[0]

    return experiments[0]


# ============================================================
# LOG ANALYSIS
# ============================================================

def parse_validation_history(log_path):
    """Extrae época y val_loss de todas las validaciones."""

    history = []

    with log_path.open("r", errors="replace") as file:
        for line in file:
            match = VAL_PATTERN.search(line)

            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))

                history.append({
                    "epoch": epoch,
                    "val_loss": val_loss,
                })

    # Si una época aparece más de una vez, conserva el último registro.
    by_epoch = {}
    for item in history:
        by_epoch[item["epoch"]] = item["val_loss"]

    history = [
        {"epoch": epoch, "val_loss": loss}
        for epoch, loss in sorted(by_epoch.items())
    ]

    if not history:
        raise RuntimeError(
            "No se encontraron valores val_loss en el log.\n"
            "Revisa el formato del archivo:\n"
            f"{log_path}"
        )

    return history


# ============================================================
# CHECKPOINT ANALYSIS
# ============================================================

def find_checkpoints(experiment_dir):
    """Encuentra checkpoints y trata de extraer su número de época."""

    checkpoints = []

    for path in experiment_dir.rglob("*.pth*"):
        match = CHECKPOINT_EPOCH_PATTERN.search(path.name)

        epoch = int(match.group(1)) if match else None

        checkpoints.append({
            "path": path,
            "name": path.name,
            "epoch": epoch,
            "size_mb": path.stat().st_size / (1024 ** 2),
            "modified": path.stat().st_mtime,
        })

    return sorted(
        checkpoints,
        key=lambda item: (
            item["epoch"] is None,
            item["epoch"] if item["epoch"] is not None
            else item["modified"]
        )
    )

def rank_saved_checkpoints(history, checkpoints):
    """
    Obtiene el ranking de val_loss únicamente para épocas
    que tienen un checkpoint guardado.

    La validación de la época N ocurre antes de guardar
    epoch_N.pth.tar, por lo que ambos corresponden a los
    mismos pesos.
    """

    validation_by_epoch = {
        item["epoch"]: item["val_loss"]
        for item in history
    }

    saved_epochs = []

    for cp in checkpoints:
        epoch = cp["epoch"]

        # Ignorar checkpoints sin época o sin validación asociada.
        if epoch is None or epoch not in validation_by_epoch:
            continue

        saved_epochs.append({
            "epoch": epoch,
            "val_loss": validation_by_epoch[epoch],
            "checkpoint": cp["path"],
            "size_mb": cp["size_mb"],
        })

    return sorted(
        saved_epochs,
        key=lambda item: item["val_loss"]
    )

def save_saved_checkpoint_ranking(ranking, output_dir):
    """Guarda el ranking de los checkpoints disponibles."""

    path = output_dir / "saved_checkpoints_ranking.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "epoch",
                "val_loss",
                "checkpoint",
                "size_mb",
            ]
        )

        writer.writeheader()

        for rank, item in enumerate(ranking, start=1):
            writer.writerow({
                "rank": rank,
                "epoch": item["epoch"],
                "val_loss": item["val_loss"],
                "checkpoint": item["checkpoint"],
                "size_mb": f"{item['size_mb']:.2f}",
            })

    return path

def analyze_checkpoint_availability(best_epoch, checkpoints):
    """Compara la mejor época de validación con los pesos guardados."""

    exact = [
        cp for cp in checkpoints
        if cp["epoch"] == best_epoch
    ]

    numbered = [
        cp for cp in checkpoints
        if cp["epoch"] is not None
    ]

    if exact:
        return "exact", exact

    if numbered:
        nearest_distance = min(
            abs(cp["epoch"] - best_epoch)
            for cp in numbered
        )

        nearest = [
            cp for cp in numbered
            if abs(cp["epoch"] - best_epoch) == nearest_distance
        ]

        return "nearest", nearest

    return "unknown", checkpoints


# ============================================================
# OUTPUT
# ============================================================

def save_history_csv(history, output_dir):
    path = output_dir / "validation_history.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "val_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    return path

def save_top_epochs_csv(top_epochs, output_dir):
    """Guarda las mejores épocas ordenadas por val_loss."""

    path = output_dir / "top_epochs.csv"

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "epoch", "val_loss"]
        )
        writer.writeheader()

        for rank, item in enumerate(top_epochs, start=1):
            writer.writerow({
                "rank": rank,
                "epoch": item["epoch"],
                "val_loss": item["val_loss"],
            })

    return path


def save_plot(history, best_epoch, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[AVISO] matplotlib no está instalado. Se omite la gráfica.")
        return None

    epochs = [item["epoch"] for item in history]
    losses = [item["val_loss"] for item in history]

    best_loss = min(losses)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, label="Validation loss")
    plt.scatter(
        [best_epoch],
        [best_loss],
        label=f"Best epoch: {best_epoch}",
        zorder=3
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("MapNet - Validation history")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = output_dir / "validation_loss.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return path


def save_summary(
    experiment,
    history,
    best_epoch,
    best_loss,
    top_epochs,
    checkpoints,
    checkpoint_status,
    selected_checkpoints,
    output_dir,
):
    path = output_dir / "summary.txt"

    with path.open("w") as file:
        file.write("MAPNET TRAINING ANALYSIS\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Experiment: {experiment['name']}\n")
        file.write(f"Log: {experiment['log']}\n")
        file.write(f"Training directory: {experiment['directory']}\n")
        file.write(f"Validation epochs: {len(history)}\n")
        file.write(f"First validation epoch: {history[0]['epoch']}\n")
        file.write(f"Last validation epoch: {history[-1]['epoch']}\n\n")

        file.write(f"Best validation epoch: {best_epoch}\n")
        file.write(f"Best validation loss: {best_loss:.8f}\n\n")

        file.write("TOP 10 VALIDATION EPOCHS\n")
        file.write("-" * 65 + "\n")

        for rank, item in enumerate(top_epochs, start=1):
            file.write(
                f"{rank:2d}. Epoch {item['epoch']:4d} "
                f"| val_loss: {item['val_loss']:.8f}\n"
            )

        file.write("\n")

        file.write(f"Available checkpoints: {len(checkpoints)}\n")
        file.write(f"Checkpoint status: {checkpoint_status}\n")

        for cp in selected_checkpoints:
            file.write(f"\nCheckpoint: {cp['path']}\n")
            file.write(f"Checkpoint epoch: {cp['epoch']}\n")
            file.write(f"Size: {cp['size_mb']:.2f} MB\n")

        if checkpoint_status == "exact":
            file.write(
                "\nExiste un checkpoint con el mismo número de época "
                "que la mejor validación. Aun así, debe comprobarse "
                "la correspondencia temporal entre validación y guardado.\n"
            )
        elif checkpoint_status == "nearest":
            file.write(
                "\nADVERTENCIA: no existe un checkpoint identificado "
                "con la mejor época. Los mostrados son los más cercanos "
                "por número de época; no se puede afirmar que sean "
                "los mejores pesos.\n"
            )
        else:
            file.write(
                "\nNo fue posible asociar los checkpoints con épocas "
                "a partir de sus nombres.\n"
            )

    return path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analiza automáticamente entrenamientos MapNet."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Nombre exacto del experimento. Por defecto usa el más reciente."
    )

    parser.add_argument(
        "--list-all",
        action="store_true",
        help="Muestra todos los experimentos encontrados y termina."
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MAPNET - AUTOMATIC TRAINING ANALYSIS")
    print("=" * 70)

    print(f"\nBuscando logs en:\n{LOGS_ROOT}")

    experiments = discover_experiments()

    print(f"\nExperimentos encontrados: {len(experiments)}")

    for exp in experiments:
        date = datetime.fromtimestamp(
            exp["modified"]
        ).strftime("%Y-%m-%d %H:%M:%S")

        print(f"  {exp['name']}  |  {date}")

    if args.list_all:
        return

    experiment = select_experiment(
        experiments,
        requested_name=args.experiment
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO SELECCIONADO")
    print("=" * 70)
    print(experiment["name"])
    print(f"Log: {experiment['log']}")

    history = parse_validation_history(experiment["log"])

    best = min(history, key=lambda item: item["val_loss"])
    best_epoch = best["epoch"]
    best_loss = best["val_loss"]

    # Ranking de las 10 mejores épocas.
    top_epochs = sorted(
        history,
        key=lambda item: item["val_loss"]
    )[:10]

    print("\n" + "=" * 70)
    print("RESULTADOS DE VALIDACIÓN")
    print("=" * 70)

    print(f"Primera época registrada: {history[0]['epoch']}")
    print(f"Última época registrada:  {history[-1]['epoch']}")
    print(f"Épocas evaluadas:         {len(history)}")
    print(f"Mejor época:              {best_epoch}")
    print(f"Mejor val_loss:           {best_loss:.8f}")

    print("\n" + "=" * 70)
    print("TOP 10 MEJORES ÉPOCAS")
    print("=" * 70)

    print(f"{'Ranking':<10}{'Época':<12}{'Val Loss':<15}")
    print("-" * 37)

    for rank, item in enumerate(top_epochs, start=1):
        print(
            f"{rank:<10}"
            f"{item['epoch']:<12}"
            f"{item['val_loss']:.8f}"
        )

    checkpoints = find_checkpoints(experiment["directory"])

    # Ranking de validaciones que sí tienen pesos guardados.
    saved_checkpoint_ranking = rank_saved_checkpoints(
        history,
        checkpoints
    )

    best_saved_checkpoint = (
        saved_checkpoint_ranking[0]
        if saved_checkpoint_ranking
        else None
    )

    print("\n" + "=" * 70)
    print("TOP 10 - MEJORES ÉPOCAS CON CHECKPOINT")
    print("=" * 70)

    print(
        f"{'Ranking':<10}"
        f"{'Época':<12}"
        f"{'Val Loss':<15}"
        f"{'Checkpoint'}"
    )
    print("-" * 90)

    for rank, item in enumerate(
        saved_checkpoint_ranking[:10],
        start=1
    ):
        print(
            f"{rank:<10}"
            f"{item['epoch']:<12}"
            f"{item['val_loss']:<15.8f}"
            f"{Path(item['checkpoint']).name}"
        )

    if best_saved_checkpoint:
        print("\nMEJOR CHECKPOINT DISPONIBLE:")
        print(f"Época:     {best_saved_checkpoint['epoch']}")
        print(f"Val loss:  {best_saved_checkpoint['val_loss']:.8f}")
        print(f"Archivo:   {best_saved_checkpoint['checkpoint']}")

    print("\n" + "=" * 70)
    print("CHECKPOINTS")
    print("=" * 70)

    print(f"Checkpoints encontrados: {len(checkpoints)}")

    for cp in checkpoints:
        epoch_text = (
            str(cp["epoch"])
            if cp["epoch"] is not None
            else "sin época identificada"
        )

        print(
            f"  {cp['name']} | época: {epoch_text} "
            f"| {cp['size_mb']:.2f} MB"
        )

    status, selected = analyze_checkpoint_availability(
        best_epoch,
        checkpoints
    )

    if status == "exact":
        print("\n[OK] Existe checkpoint con el número de la mejor época.")
    elif status == "nearest":
        print(
            "\n[AVISO] No hay checkpoint identificado para la mejor época."
        )
        print("Checkpoint(s) más cercano(s):")
    else:
        print("\n[AVISO] No se pudo asociar checkpoint y época.")

    for cp in selected:
        print(f"  {cp['path']}")

    output_dir = OUTPUT_ROOT / experiment["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_history_csv(history, output_dir)
    top_epochs_path = save_top_epochs_csv(top_epochs, output_dir)
    saved_ranking_path = save_saved_checkpoint_ranking(
        saved_checkpoint_ranking,
        output_dir
    )
    plot_path = save_plot(history, best_epoch, output_dir)

    summary_path = save_summary(
        experiment,
        history,
        best_epoch,
        best_loss,
        top_epochs,
        checkpoints,
        status,
        selected,
        output_dir,
    )

    print("\n" + "=" * 70)
    print("ARCHIVOS GENERADOS")
    print("=" * 70)
    print(f"CSV:     {csv_path}")
    print(f"Resumen: {summary_path}")
    print(f"Top 10:  {top_epochs_path}")   
    print(f"Ranking checkpoints: {saved_ranking_path}") 

    if plot_path:
        print(f"Gráfica: {plot_path}")

    print("\nAnálisis terminado.")


if __name__ == "__main__":
    main()
