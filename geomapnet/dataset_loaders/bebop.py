
#!/usr/bin/env python3
"""
dataset_loaders/bebop.py

Dataset loader para Parrot Bebop preparado para GeoMapNet/MapNet.

Formato:
    data_path/scene/
        TrainSplit.txt
        TestSplit.txt
        pose_stats.txt
        stats.txt
        sequence_name/
            poses.txt
            imágenes listadas en poses.txt

Cada línea de poses.txt:
    filename r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

CONVENCIÓN:
    Las matrices de poses.txt son Camera-to-World (C2W).

    T_c2w = [R_c2w | C_w]

    R_c2w: orientación de la cámara expresada en el mundo.
    C_w: posición del centro de cámara en coordenadas mundo FLU.

Las imágenes listadas en poses.txt son la única fuente de muestras.
No se incorporan imágenes adicionales.

pose_stats.txt:
    Primera fila: media de las posiciones C_w (x, y, z).
    Segunda fila: desviación estándar de las posiciones C_w.
    Ambas calculadas exclusivamente con TrainSplit.
"""

import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset_loaders.utils import load_image
from common.pose_utils import process_poses


def _read_split(base_dir, train):
    split_file = osp.join(
        base_dir, "TrainSplit.txt" if train else "TestSplit.txt"
    )

    if not osp.isfile(split_file):
        raise FileNotFoundError("No se encontró: {}".format(split_file))

    with open(split_file, "r") as f:
        sequences = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not sequences:
        raise RuntimeError("El split está vacío: {}".format(split_file))

    if any(seq.lower() == "ref" for seq in sequences):
        raise ValueError(
            "'ref' aparece en {}. Excluye las referencias de los splits."
            .format(split_file)
        )

    return sequences


def _read_poses_file(poses_file):
    """Lee filename + 12 valores de una matriz C2W por línea."""
    filenames, poses = [], []

    with open(poses_file, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) != 13:
                raise ValueError(
                    "{}:{}: se esperaban 13 campos; se encontraron {}."
                    .format(poses_file, line_number, len(fields))
                )

            try:
                pose = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    "{}:{}: pose no numérica."
                    .format(poses_file, line_number)
                ) from exc

            if not np.all(np.isfinite(pose)):
                raise ValueError(
                    "{}:{}: pose contiene valores no finitos."
                    .format(poses_file, line_number)
                )

            filenames.append(fields[0])
            poses.append(pose)

    return filenames, poses


def _load_pose_stats(base_dir):
    """
    Carga estadísticas C2W generadas por prepare_bebop.py.

    No recalcula estadísticas con el conjunto de test.
    """
    stats_file = osp.join(base_dir, "pose_stats.txt")

    if not osp.isfile(stats_file):
        raise FileNotFoundError(
            "No existe {}. Ejecuta primero prepare_bebop.py."
            .format(stats_file)
        )

    stats = np.loadtxt(stats_file, dtype=np.float64)

    if stats.shape != (2, 3):
        raise ValueError(
            "{} debe tener 2 filas y 3 columnas; forma encontrada: {}."
            .format(stats_file, stats.shape)
        )

    mean_t = stats[0]
    std_t = stats[1]

    if not np.all(np.isfinite(stats)):
        raise ValueError(
            "{} contiene valores no finitos.".format(stats_file)
        )

    if np.any(std_t <= 0):
        raise ValueError(
            "{} debe contener desviaciones estándar positivas."
            .format(stats_file)
        )

    return mean_t, std_t


def compute_rgb_stats(data_path, scene, output_file=None):
    """
    Calcula media y varianza RGB usando solo las imágenes del TrainSplit.

    Las imágenes se leen desde las listas de poses.txt.

    Guarda dos filas:
        mean_R mean_G mean_B
        var_R  var_G  var_B

    Los valores están en [0, 1].
    """
    base_dir = osp.join(data_path, scene)
    sequences = _read_split(base_dir, train=True)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    image_count = 0

    for seq in sequences:
        seq_dir = osp.join(base_dir, seq)
        poses_file = osp.join(seq_dir, "poses.txt")

        filenames, _ = _read_poses_file(poses_file)

        for filename in filenames:
            image_path = osp.join(seq_dir, filename)

            if not osp.isfile(image_path):
                raise FileNotFoundError(
                    "Imagen listada en poses.txt no encontrada: {}"
                    .format(image_path)
                )

            with Image.open(image_path) as im:
                arr = np.asarray(
                    im.convert("RGB"), dtype=np.float64
                ) / 255.0

            pixels = arr.reshape(-1, 3)

            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += np.square(pixels).sum(axis=0)

            pixel_count += pixels.shape[0]
            image_count += 1

        print(
            "[{:<18}] imágenes para RGB stats: {}"
            .format(seq, len(filenames))
        )

    if pixel_count == 0:
        raise RuntimeError(
            "No se encontraron píxeles para calcular stats RGB."
        )

    mean = channel_sum / pixel_count
    variance = channel_sq_sum / pixel_count - np.square(mean)
    variance = np.maximum(variance, 0.0)

    if output_file is None:
        output_file = osp.join(base_dir, "stats.txt")

    np.savetxt(
        output_file,
        np.vstack((mean, variance)),
        fmt="%.10f",
    )

    print(
        "RGB stats calculadas con {} imágenes / {:,} píxeles."
        .format(image_count, pixel_count)
    )
    print("Mean RGB    :", mean)
    print("Variance RGB:", variance)
    print("Guardado en :", output_file)

    return mean, variance


class Bebop(Dataset):

    def __init__(
        self,
        scene,
        data_path,
        train=True,
        transform=None,
        target_transform=None,
        real=False,
        skip_images=False,
        seed=7,
        align_R=None,
        align_t=None,
        align_s=1.0,
    ):
        super().__init__()

        self.scene = scene
        self.data_path = data_path
        self.base_dir = osp.join(data_path, scene)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.real = real
        self.skip_images = skip_images
        self.seed = seed

        self.align_R = (
            np.eye(3, dtype=np.float64)
            if align_R is None
            else np.asarray(align_R, dtype=np.float64)
        )

        self.align_t = (
            np.zeros(3, dtype=np.float64)
            if align_t is None
            else np.asarray(align_t, dtype=np.float64)
        )

        self.align_s = float(align_s)

        if self.align_R.shape != (3, 3):
            raise ValueError("align_R debe tener forma (3, 3).")

        if self.align_t.shape != (3,):
            raise ValueError("align_t debe tener forma (3,).")

        if not np.isfinite(self.align_s) or self.align_s <= 0:
            raise ValueError("align_s debe ser positivo y finito.")

        sequences = _read_split(self.base_dir, train)

        self.img_paths = []
        self.raw_poses = []
        self.sequence_ranges = []
        self.sequence_names = []

        for seq in sequences:
            seq_dir = osp.join(self.base_dir, seq)
            poses_file = osp.join(seq_dir, "poses.txt")

            if not osp.isfile(poses_file):
                raise FileNotFoundError(
                    "No se encontró poses.txt: {}".format(poses_file)
                )

            names, poses = _read_poses_file(poses_file)

            if not names:
                raise RuntimeError(
                    "No hay poses en {}".format(poses_file)
                )

            start = len(self.img_paths)

            for filename in names:
                image_path = osp.join(seq_dir, filename)

                if not osp.isfile(image_path):
                    raise FileNotFoundError(
                        "Imagen listada en poses.txt no encontrada: {}"
                        .format(image_path)
                    )

                self.img_paths.append(image_path)

            self.raw_poses.extend(poses)

            end = len(self.img_paths)

            self.sequence_ranges.append((start, end))
            self.sequence_names.append(seq)

            print(
                "[{:<18}] {:>4} imágenes"
                .format(seq, len(names))
            )

        self.raw_poses = np.asarray(
            self.raw_poses, dtype=np.float64
        )

        if self.raw_poses.ndim != 2 or self.raw_poses.shape[1] != 12:
            raise RuntimeError(
                "Las poses C2W deben tener forma (N, 12); forma encontrada: {}"
                .format(self.raw_poses.shape)
            )

        if not np.all(np.isfinite(self.raw_poses)):
            raise ValueError("Las poses contienen valores no finitos.")

        # En C2W, las columnas [3, 7, 11] representan C_w.
        # Las estadísticas se generan previamente con TrainSplit.
        self.mean_t, self.std_t = _load_pose_stats(self.base_dir)

        print("Estadísticas C2W cargadas desde pose_stats.txt.")

        # process_poses recibe directamente matrices C2W.
        # No se invierten las matrices ni se cambia su convención.
        self.poses = process_poses(
            self.raw_poses,
            self.mean_t,
            self.std_t,
            self.align_R,
            self.align_t,
            self.align_s,
        )

        if len(self.poses) != len(self.img_paths):
            raise RuntimeError(
                "Cantidad de imágenes y poses inconsistente."
            )

        print(
            "Bebop {}: {} secuencias, {} imágenes".format(
                "train" if train else "test",
                len(self.sequence_names),
                len(self.img_paths),
            )
        )

        print("  Pose convention: C2W")
        print("  mean_Cw:", self.mean_t)
        print("  std_Cw :", self.std_t)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        if self.skip_images:
            image = None
        else:
            image = load_image(self.img_paths[index])

            if image is None:
                return None

            if self.transform is not None:
                image = self.transform(image)

        pose = self.poses[index]

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        return image, pose
