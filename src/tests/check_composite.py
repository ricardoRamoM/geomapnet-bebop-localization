import torch
from torchvision import transforms

from dataset_loaders.composite import MF

# Transformaciones para reproducir la entrada esperada por MapNet.
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Convierte las poses NumPy a tensores PyTorch.
pose_transform = lambda pose: torch.as_tensor(
    pose, dtype=torch.float32
)

kwargs = {
    "scene": "bebop",
    "data_path": "../data/deepslam_data/Bebop",
    "train": True,
    "steps": 3,
    "skip": 10,
    "variable_skip": False,
    "transform": image_transform,
    "target_transform": pose_transform,
}

print("=== PRUEBA DATASET BEBOP / MAPNET ===")

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

train_set = MF(dataset="Bebop", **kwargs)

print("\n--- Entrenamiento ---")
print("Imágenes:", len(train_set.dset))
print("Clips:", len(train_set))

imgs, poses = train_set[0]

print("Forma imágenes:", imgs.shape)
print("Forma poses:", poses.shape)
print("Tipo imágenes:", imgs.dtype)
print("Tipo poses:", poses.dtype)
print("Índices del primer clip:", train_set.get_indices(0))

assert imgs.shape == (3, 3, 224, 224)
assert poses.shape == (3, 6)
assert imgs.dtype == torch.float32
assert poses.dtype == torch.float32

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

kwargs["train"] = False

val_set = MF(dataset="Bebop", **kwargs)

print("\n--- Validación ---")
print("Imágenes:", len(val_set.dset))
print("Clips:", len(val_set))

imgs_val, poses_val = val_set[0]

print("Forma imágenes:", imgs_val.shape)
print("Forma poses:", poses_val.shape)
print("Índices del primer clip:", val_set.get_indices(0))

assert imgs_val.shape == (3, 3, 224, 224)
assert poses_val.shape == (3, 6)

print("\n================================")
print("TODAS LAS PRUEBAS SUPERADAS")
print("================================")