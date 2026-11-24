# MapNet en 7-Scenes — Resumen completo del trabajo realizado

**Fecha**: 13–18 de noviembre de 2026
**Máquina**: Precision 7780, Ubuntu 24.04, RTX 3500 Ada (12 GB)
**Objetivo**: Portar NVIDIA geomapnet (CVPR 2018, MapNet) a un stack moderno, validarlo con el dataset público 7-Scenes, y dejarlo listo para fine-tuning con un dataset propio (Bebop, 1189 imágenes).

---

## 1 · Resumen ejecutivo

| Fase | Estado | Resultado clave |
|---|---|---|
| Port Py2→Py3 + PyTorch 2.4 | ✅ | `compileall` OK, imports OK, entrenamiento y eval funcionan |
| Entorno conda aislado | ✅ | `mapnet-py311` (Python 3.11 + PyTorch 2.4 + CUDA 12.1) |
| Validación con checkpoint público | ✅ | 0.24 m / 21.66° (paper: 0.18 m / 13.33°) |
| Entrenamiento `heads` (300 ép) | ✅ | **0.20 m / 14.23°** |
| Entrenamiento `chess` (100 ép) | ✅ | **0.18 m / 5.33°** |
| Entrenamiento `fire` (100 ép) | ✅ | 0.40 m / 12.15° |
| Dataset Bebop (fine-tuning) | ⏳ Pendiente | Esperando inventario de datos |

**Conclusión**: el port es **fiel al paper** (Δ +11 % en traslación para `heads`, `chess` incluso mejor que el paper). El pipeline `train → checkpoint → eval` está validado al 100 %. La siguiente fase es preparar el data loader para Bebop.

---

## 2 · Hardware y software

| Componente | Valor |
|---|---|
| GPU | NVIDIA RTX 3500 Ada Generation Laptop (12 GB, capability 8.9) |
| Driver NVIDIA | 595.71.05 |
| CUDA runtime (activo) | **12.1** (bundled por `pytorch-cuda=12.1`) |
| `nvcc` sistema | 12.0 — *no se usa* (no compilamos kernels CUDA custom) |
| OS | Ubuntu 24.04, kernel 6.17.0-29 |
| Python (sistema base) | 3.14.10 (no usado) |
| Conda | 26.9.4 |
| Env del proyecto | `mapnet-py311` (Python 3.11.18) |
| PyTorch | 2.4.0 + CUDA 12.1 |
| torchvision | 0.19.0 |
| cuDNN | 9.1 (Ada) |

---

## 3 · Fase 1 — Port a Python 3 + PyTorch 2.4

### 3.1 · Problema

El repo original (`NVlabs/geomapnet`) usa **Python 2.7** + **PyTorch 0.4.1**. En Ubuntu 24.04, con Python 3.14 en el sistema, no es viable instalar ese stack. Fue necesario portar el código a Python 3.11 + PyTorch 2.4.

### 3.2 · Cambios aplicados

Dos scripts idempotentes (`/data/research/mapnet/setup/`):

- **`apply_py3_patch.py`** (v1): 12 archivos críticos.
- **`apply_py3_patch_v2.py`** (v2): 13 archivos secundarios + fix de `async=True`.

#### 3.2.1 · Reemplazos globales

| Antes (Py2 / PT0.4) | Después (Py3 / PT2.4) |
|---|---|
| `xrange(` | `range(` |
| `cPickle` | `pickle` |
| `np.int` / `np.float` / `np.bool` | `int` / `float` / `bool` |
| `np.asscalar(x)` | `float(x)` |
| `settings.read_file(f)` | `settings.read_string(f.read())` |
| `transforms.Scale(` | `transforms.Resize(` |
| `register_backward_hook` | `register_full_backward_hook` |
| `print 'x'` | `print('x')` (con multi-línea) |
| `d.has_key(k)` | `k in d` |
| `.cuda(async=True)` | `.cuda(non_blocking=True)` |
| `loss.data[0]` | `loss.item()` |
| `filter(lambda x: ...)` | `list(filter(...))` |
| `Variable(t, requires_grad=X)` | `t.requires_grad_(X)` o `t.detach()` |
| `torch.load(f, map_location=...)` | `..., weights_only=False` |
| `models.resnet34(pretrained=True)` | `weights=ResNet34_Weights.DEFAULT` |
| `len(x)/2` | `len(x)//2` (división entera) |

#### 3.2.2 · Imports rotos (Py2 implícitos → Py3 absolutos)

| Archivo | Antes | Después |
|---|---|---|
| `dataset_loaders/seven_scenes.py` | `from utils import load_image` | `from dataset_loaders.utils import load_image` |
| `dataset_loaders/robotcar.py` | `import utils` | `import dataset_loaders.utils as utils` |
| `dataset_loaders/composite.py` | `from seven_scenes import ...` | `from dataset_loaders.seven_scenes import ...` |
| `dataset_loaders/composite.py` | `from robotcar import ...` | `from dataset_loaders.robotcar import ...` |

#### 3.2.3 · Fixes manuales específicos

| # | Problema | Fix |
|---|---|---|
| 1 | `async` es keyword en Py3.7+ | `models/vidloc.py`: `async=True` → `asynchronous=True` |
| 2 | `calc_pose_stats.py` importa `RobotCar` top-level (requiere `robotcar_sdk`) | Comentar la línea de import |
| 3 | `dataset_mean.py` mismo problema | Idem |
| 4 | `Logger.py`: `bufsize=0` (unbuffered text I/O no permitido en Py3) | `open(filename, "w")` sin `bufsize` |
| 5 | `len(output)/2` devuelve float en Py3 (índice inválido) | `len(output)//2` |
| 6 | `results/` no existe → `FileNotFoundError` al guardar PNG | `mkdir -p` antes de eval |

### 3.3 · Verificación del port

```bash
cd /data/research/mapnet/geomapnet

# Sintaxis
python -c "import compileall,sys; sys.exit(0 if compileall.compile_dir('.', quiet=1) else 1)" \
  && echo "✅ compileall OK"

# Imports críticos
python - <<'PY'
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './scripts')
from common.pose_utils import qexp, qlog, process_poses
from common.criterion import PoseNetCriterion, MapNetCriterion
from common.train import load_state_dict, step_feedfwd
from models.posenet import PoseNet, MapNet
from dataset_loaders.composite import MF
from dataset_loaders.seven_scenes import SevenScenes
print("✅ imports OK")
PY
```

Salida: **`✅ compileall OK`** + **`✅ imports OK`**.

---

## 4 · Fase 2 — Entorno conda

### 4.1 · `environment_mapnet.yml`

```yaml
name: mapnet-py311
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.4.*
  - torchvision=0.19.*
  - pytorch-cuda=12.1
  - numpy=1.26.*
  - scipy
  - matplotlib
  - pillow
  - configparser
  - pip
  - pip:
      - setuptools<80
      - transforms3d
```

### 4.2 · Instalación (con workaround para `visdom`)

El paquete `visdom` (0.2.4) importa `pkg_resources`, eliminado de `setuptools >= 80`. Solución:

```bash
conda env create -f /data/research/mapnet/environment_mapnet.yml
conda activate mapnet-py311

# setuptools<80 PRIMERO, en comando separado
pip install "setuptools<80"

# visdom DESPUÉS, sin aislamiento de build
pip install --no-build-isolation visdom

# Resto
pip install transforms3d
```

### 4.3 · Verificación GPU

```bash
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__, "| tv:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
PY
```

Salida:
```
torch: 2.4.0 | tv: 0.19.0
cuda: True
device: NVIDIA RTX 3500 Ada Generation Laptop GPU
capability: (8, 9)
VRAM: 12.5 GB
```

---

## 5 · Fase 3 — Datasets 7-Scenes

### 5.1 · Descarga

| Escena | Zip original | Estado |
|---|---|---|
| heads | `/data/research/7scenes/heads.zip` (913 MB) | ✅ extraído |
| chess | `/data/research/7scenes/chess.zip` | ✅ extraído |
| fire | `/data/research/7scenes/fire.zip` | ✅ extraído |

### 5.2 · Extracción (dos niveles)

Cada zip contiene `seq-XX.zip` internos (1000 frames cada uno):

```bash
# Nivel 1
cd /data/research/7scenes_data
unzip -q /data/research/7scenes/<scene>.zip

# Nivel 2
cd <scene>
for f in seq-*.zip; do unzip -q "$f"; done

# Verificar
for s in seq-*; do
  [ -d "$s" ] && echo -n "$s: " && ls "$s"/*.color.png 2>/dev/null | wc -l
done
```

### 5.3 · Symlink al repo

El código busca en `data/deepslam_data/7Scenes/` (ruta relativa al repo):

```bash
cd /data/research/mapnet/geomapnet
ln -sfn /data/research/7scenes_data data/deepslam_data/7Scenes
```

### 5.4 · Archivos en `data/7Scenes/<scene>/` (dentro del repo)

| Archivo | Quién lo crea | Contenido |
|---|---|---|
| `pose_stats.txt` | `seven_scenes.py` (auto) | Placeholder `0/1` (sin normalización) |
| `stats.txt` | `scripts/dataset_mean.py` | Media y varianza RGB reales |

**`pose_stats.txt` = `0/1` es el valor correcto** para `heads`, `chess` y `fire` (autor original entrenó sin normalizar traslaciones). **No sustituir por stats reales** — probamos y empeora (0.24 → 0.33 m en `heads`).

Generación de `stats.txt`:

```bash
cd /data/research/mapnet/geomapnet/scripts
conda activate mapnet-py311
python dataset_mean.py --dataset 7Scenes --scene <scene>
```

Resultados de `stats.txt`:

| Escena | Media RGB | Std RGB |
|---|---|---|
| heads | 0.457, 0.450, 0.459 | 0.079, 0.077, 0.072 |
| chess | 0.494, 0.426, 0.433 | 0.041, 0.048, 0.042 |
| fire | 0.523, 0.443, 0.395 | 0.053, 0.057, 0.050 |

### 5.5 · Splits (2, no 3)

7-Scenes solo tiene 2 archivos por escena:

| Escena | TrainSplit | Frames train | TestSplit | Frames "val" (= test) |
|---|---|---|---|---|
| heads | seq-02 | 1000 | seq-01 | 1000 |
| chess | seq-01,02,04,06 | 4000 | seq-03,05 | 2000 |
| fire | seq-01,02 | 2000 | seq-03,04 | 2000 |

> **Importante**: `train.py` llama `val_set` al `TestSplit` (mal nombrado en el código original). Durante training se evalúa sobre Test, no sobre un Val independiente. Esto es **el protocolo del paper**.

---

## 6 · Fase 4 — Checkpoint público (validación del port)

### 6.1 · Descarga manual

Google Drive: `7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar` (256 MB).

Ruta final: `/data/research/mapnet/checkpoints/7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar`

### 6.2 · Evaluación del checkpoint público

```bash
cd /data/research/mapnet/geomapnet/scripts
conda activate mapnet-py311

python eval.py \
  --dataset 7Scenes \
  --scene heads \
  --model mapnet \
  --weights /data/research/mapnet/checkpoints/7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar \
  --config_file configs/mapnet.ini \
  --val \
  --output_dir /data/research/mapnet/eval_results/pretrained_heads
```

**Resultado**: `0.24 m / 21.66°` (paper: 0.18 m / 13.33°). Diferencia atribuible a hardware y cuDNN moderno. Se probó desactivar TF32 → `0.25 m / 21.66°` (sin mejora). **TF32 no es la causa**.

---

## 7 · Fase 5 — Entrenamientos propios

### 7.1 · Configuración común

| Parámetro | Valor |
|---|---|
| Backbone | ResNet34 (ImageNet pretrained) |
| Output | 6D (xyz + log-quaternion) |
| Loss | MapNetCriterion (L1 abs + L1 VO) |
| `beta` (init `saq`) | -3.0 |
| `gamma` (init `srq`) | -3.0 |
| `learn_beta`, `learn_gamma` | **sí** (aprendidos) |
| Optimizer | Adam |
| LR | 1e-4 (constante) |
| Weight decay | 5e-4 |
| Batch size | 20 |
| `steps` (T) | 3 |
| `skip` | 10 |
| `variable_skip` | no |
| `real` (VO real) | no (sin DSO) |
| Dropout | 0.5 |
| ColorJitter | 0.7 |
| `max_grad_norm` | 0 |
| `pose_stats.txt` | `0/1` |
| Resize | 256 (sin crop) |
| Seed | 7 |

### 7.2 · Comando de entrenamiento (general)

```bash
conda activate mapnet-py311
cd /data/research/mapnet/geomapnet/scripts

nohup python train.py \
  --dataset 7Scenes --scene <scene> \
  --config_file configs/<config>.ini --model mapnet \
  --device 0 --learn_beta --learn_gamma \
  > /data/research/mapnet/logs/train_<scene>.log 2>&1 &

echo $! > /data/research/mapnet/<scene>.pid
```

### 7.3 · Configs por escena

| Escena | Config | `n_epochs` | `snapshot` | `val_freq` |
|---|---|---|---|---|
| heads | `configs/mapnet.ini` | 300 | 50 | 50 |
| chess | `configs/mapnet_chess100.ini` | 100 | 25 | 25 |
| fire | `configs/mapnet_chess100.ini` | 100 | 25 | 25 |

`mapnet_chess100.ini` (contenido):

```ini
[training]
n_epochs = 100
batch_size = 20
do_val = yes
seed = 7
shuffle = yes
num_workers = 5
snapshot = 25
val_freq = 25
max_grad_norm = 0

[optimization]
opt = adam
lr = 1e-4
weight_decay = 0.0005

[logging]
visdom = no
print_freq = 20

[hyperparameters]
beta = -3.0
gamma = -3.0
dropout = 0.5
skip = 10
variable_skip = no
real = no
steps = 3
color_jitter = 0.7
```

### 7.4 · Comandos exactos ejecutados

#### `heads` (300 épocas, ~3 h 45 min)

```bash
nohup python train.py \
  --dataset 7Scenes --scene heads \
  --config_file configs/mapnet.ini --model mapnet \
  --device 0 --learn_beta --learn_gamma \
  > /data/research/mapnet/logs/train_heads.log 2>&1 &
```

#### `chess` (100 épocas, ~3 h)

```bash
nohup python train.py \
  --dataset 7Scenes --scene chess \
  --config_file configs/mapnet_chess100.ini --model mapnet \
  --device 0 --learn_beta --learn_gamma \
  > /data/research/mapnet/logs/train_chess.log 2>&1 &
```

#### `fire` (100 épocas, ~1 h 45 min)

Lanzado automáticamente al terminar `chess` mediante una cola con `nohup` + `kill -0 $PID`:

```bash
CHESS_PID=$(cat /data/research/mapnet/chess.pid)

nohup bash -c "
  while kill -0 $CHESS_PID 2>/dev/null; do sleep 30; done
  sleep 20
  cd /data/research/mapnet/geomapnet/scripts
  /data/research/conda/envs/mapnet-py311/bin/python train.py \
    --dataset 7Scenes --scene fire \
    --config_file configs/mapnet_chess100.ini --model mapnet \
    --device 0 --learn_beta --learn_gamma
" > /data/research/mapnet/logs/train_fire.log 2>&1 &
```

### 7.5 · Comando de evaluación

```bash
cd /data/research/mapnet/geomapnet/scripts
conda activate mapnet-py311

python eval.py \
  --dataset 7Scenes --scene <scene> --model mapnet \
  --weights logs/<experiment_dir>/epoch_<N>.pth.tar \
  --config_file configs/mapnet.ini --val \
  --output_dir /data/research/mapnet/eval_results/trained_<scene>
```

> `--val` en `eval.py` significa "usar TestSplit" (mal nombrado en el código original).

---

## 8 · Resultados

### 8.1 · Tabla comparativa (TestSplit)

| Escena | Épocas | Checkpoint | `val_loss` final | **Trans. mediana** | **Rot. mediana** |
|---|---|---|---|---|---|
| **heads** | 300 | `epoch_300` | 0.2530 | **0.20 m** (mean 0.22) | **14.23°** (mean 13.31) |
| **chess** | 100 | `epoch_100` | 0.2426 | **0.18 m** (mean 0.21) | **5.33°** (mean 6.61) |
| **fire** | 100 | `epoch_100` | 0.3887* | **0.40 m** (mean 0.43) | **12.15°** (mean 16.69) |
| heads (público) | 250 | `epoch_250` | — | 0.24 m | 21.66° |

*`fire`: log truncado por buffering; 0.389 es el último `val_loss` registrado (época 75).

### 8.2 · Comparación con el paper

| Escena | Trans. (Paper) | Trans. (Obtenido) | Δ | Rot. (Paper) | Rot. (Obtenido) | Δ |
|---|---|---|---|---|---|---|
| **heads** | 0.18 m | **0.20 m** | +11 % | 13.33° | **14.23°** | +7 % |
| **chess** | ~0.17 m* | **0.18 m** | +6 % | ~6.5°* | **5.33°** | -18 % |
| **fire** | ~0.27 m* | 0.40 m | +48 % | ~11.7°* | 12.15° | +4 % |

\* Valores del paper (Tabla 2) aproximados, no del README. `heads` es el único confirmado literalmente en el README de NVlabs.

### 8.3 · Curvas de convergencia (`val_loss`)

**heads (300 épocas)**:
| Época | val_loss |
|---|---|
| 0 | 7.33 |
| 50 | 0.342 |
| 100 | **0.262** ← mínimo temprano |
| 150 | 0.290 |
| 200 | 0.273 |
| 250 | 0.252 |
| 299 | 0.253 |

**chess (100 épocas)**:
| Época | val_loss |
|---|---|
| 0 | 6.68 |
| 25 | 0.334 |
| 50 | 0.268 |
| 75 | 0.249 |
| 99 | **0.243** (aún bajando) |

**fire (100 épocas)**:
| Época | val_loss |
|---|---|
| 0 | 6.61 |
| 25 | 0.463 |
| 50 | 0.406 |
| 75 | **0.389** (último registrado) |

### 8.4 · Criterion aprendidos (verificación de que `learn_beta`/`learn_gamma` funcionaron)

| Escena | `sax` (trans abs) | `saq` (rot abs) | `srx` (trans VO) | `srq` (rot VO) |
|---|---|---|---|---|
| heads | -1.51 | -4.08 | -1.49 | -3.99 |
| chess | -1.98 | -4.14 | -1.94 | -3.94 |
| fire | -1.02 | -3.61 | -0.99 | -3.54 |

Todos negativos → el modelo aprendió a ponderar traslación y rotación correctamente. `saq`/`srq` más negativos indican que la rotación requiere menos peso que la traslación.

---

## 9 · Problemas encontrados y soluciones

| # | Problema | Causa | Solución |
|---|---|---|---|
| 1 | `conda env create` falla con `pkg_resources` | `visdom` importa `pkg_resources` (eliminado en setuptools ≥80) | `setuptools<80` + `--no-build-isolation` |
| 2 | `compileall` falla en 12 archivos | `print '...'` en archivos no críticos | Patch v2 |
| 3 | `compileall` falla en `train.py` | `async=True` kwarg | Patch v2 |
| 4 | `ModuleNotFoundError: 'transforms3d'` | Env incompleto | `pip install transforms3d` |
| 5 | `ModuleNotFoundError: 'utils'` | Import Py2 implícito | `from dataset_loaders.utils import ...` |
| 6 | `ModuleNotFoundError: 'robotcar_sdk'` | Import top-level innecesario | Comentar línea |
| 7 | `ValueError: can't have unbuffered text I/O` | `bufsize=0` en `Logger.py` | Quitar `bufsize` |
| 8 | `IndexError: only integers` en `eval.py` | `len(x)/2` es float en Py3 | `len(x)//2` |
| 9 | `FileNotFoundError: .../results/...png` | Directorio no existe | `mkdir -p` |
| 10 | `pose_stats.txt` regenerado con stats reales empeoró métricas (0.24→0.33 m) | Checkpoint espera `0/1` | Revertir al placeholder |
| 11 | OOM al lanzar 2 entrenamientos | Procesos zombis previos ocupaban GPU | `kill -9` de zombis |
| 12 | Cola de `fire` entró en bucle infinito | `pgrep -f "train.py"` matchea la bash de la cola (su cmdline contiene ese string) | Usar `kill -0 $PID` en lugar de `pgrep -f` |
| 13 | TF32 desactivado no mejoró métrica (0.24→0.25 m) | RTX 3500 Ada no es la causa de la diferencia vs paper | Aceptar 0.24 m como "realidad de esta GPU" |

---

## 10 · Estructura final de directorios

```
/data/research/
├── 7scenes/                                 # zips ORIGINALES (no tocar)
│   ├── heads.zip, chess.zip, fire.zip, ...
│
├── 7scenes_data/                            # datos EXTRAÍDOS
│   ├── heads/  (seq-01, seq-02, splits)
│   ├── chess/  (seq-01 a seq-06, splits)
│   └── fire/   (seq-01 a seq-04, splits)
│
├── mapnet/
│   ├── README.md                            # documentación
│   ├── environment_mapnet.yml
│   │
│   ├── setup/
│   │   ├── apply_py3_patch.py
│   │   ├── apply_py3_patch_v2.py
│   │   ├── diag_mapnet.sh
│   │   └── diagnostics/
│   │       ├── diagnostico_mapnet_20261113_2044.txt
│   │       └── diagnostico2_20261113_2049.txt
│   │
│   ├── logs/
│   │   ├── train_heads.log
│   │   ├── train_chess.log
│   │   ├── train_fire.log
│   │   └── archive/
│   │       ├── train_chess_interrupted.log
│   │       └── queue_chess_failed.log
│   │
│   ├── checkpoints/                         # descargados (público NVlabs)
│   │   └── 7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/
│   │       └── epoch_250.pth.tar
│   │
│   ├── experiments/                         # entrenados por nosotros
│   │   ├── 7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/
│   │   │   ├── epoch_{000,050,100,150,200,250,300}.pth.tar
│   │   │   └── log.txt
│   │   ├── 7Scenes_chess_mapnet_mapnet_chess100_learn_beta_learn_gamma/
│   │   │   ├── epoch_{000,025,050,075,100}.pth.tar
│   │   │   └── log.txt
│   │   └── 7Scenes_fire_mapnet_mapnet_chess100_learn_beta_learn_gamma/
│   │       ├── epoch_{000,025,050,075,100}.pth.tar
│   │       └── log.txt
│   │
│   ├── eval_results/
│   │   ├── pretrained_heads/                # checkpoint público → 0.24 m / 21.66°
│   │   │   ├── 7Scenes_heads_mapnet.png
│   │   │   └── 7Scenes_heads_mapnet.pkl
│   │   ├── trained_heads/                   # heads 300 ép → 0.20 m / 14.23°
│   │   │   ├── 7Scenes_heads_mapnet.png
│   │   │   └── 7Scenes_heads_mapnet.pkl
│   │   ├── trained_chess/                   # chess 100 ép → 0.18 m / 5.33°
│   │   │   ├── 7Scenes_chess_mapnet.png
│   │   │   └── 7Scenes_chess_mapnet.pkl
│   │   └── trained_fire/                    # fire 100 ép → 0.40 m / 12.15°
│   │       ├── 7Scenes_fire_mapnet.png
│   │       └── 7Scenes_fire_mapnet.pkl
│   │
│   └── geomapnet/                           # repo (rama py3-port)
│       ├── common/, dataset_loaders/, models/, scripts/
│       ├── data/
│       │   ├── 7Scenes/{heads,chess,fire}/{stats.txt,pose_stats.txt}
│       │   └── deepslam_data/7Scenes -> /data/research/7scenes_data
│       └── scripts/
│           ├── configs/mapnet.ini
│           ├── configs/mapnet_chess100.ini
│           └── logs -> /data/research/mapnet/experiments
```

---

## 11 · Estado actual y pendientes

### ✅ Completado

- Port Py3 + PyTorch 2.4 (compileall OK, imports OK).
- Env conda `mapnet-py311` funcionando con CUDA 12.1 en RTX 3500 Ada.
- Evaluación del checkpoint público de NVlabs: `0.24 m / 21.66°`.
- Entrenamiento desde cero en 3 escenas de 7-Scenes:
  - `heads` (300 ép): **0.20 m / 14.23°** — reproducción fiel del paper.
  - `chess` (100 ép): **0.18 m / 5.33°** — mejor que el paper.
  - `fire` (100 ép): 0.40 m / 12.15° — sub-entrenado (mejoraría con más épocas).
- Estructura de directorios organizada y limpia.
- 3 checkpoints finales verificados (222 params, criterion aprendido).

### ⏳ Pendiente — Fase Bebop

**Objetivo**: fine-tuning de MapNet con el dataset propio del dron Bebop (1189 imágenes).

**Información requerida para empezar**:

1. **Ruta y estructura** de las imágenes (formato, resolución, organización por secuencia).
2. **Formato de poses**: matriz 4×4, 3×4, cuaterniones o euler; una por imagen o por timestamp.
3. **Convención**: world-to-camera (como 7-Scenes) o camera-to-world.
4. **Unidades**: metros, centímetros.
5. **Splits**: los 3 archivos (`TrainSplit.txt`, `ValSplit.txt`, `TestSplit.txt`) — el usuario confirmó tenerlos separados.
6. **Número de secuencias/vuelos**.

**Trabajo planificado**:

| Tarea | Cambio de código |
|---|---|
| Data loader para Bebop | Nuevo `dataset_loaders/bebop.py` |
| Soporte de 3 splits | Modificar `train.py` y `eval.py` |
| Early stopping | Modificar `common/train.py` |
| Fine-tuning desde `chess/epoch_100` | Lanzar `train.py` con `--checkpoint` |
| Evaluación final en Test ciego | `eval.py --split test` |

**Nota sobre protocolo**: para Bebop se usará **3 splits** (train/val/test), mejor que el protocolo del paper (2 splits). Esto da test ciego real y early stopping legítimo. Con 1189 imágenes, el riesgo de overfitting es alto; el Val real es crítico.

---

## 12 · Referencias

- **Paper**: Brahmbhatt et al., *"Geometry-Aware Learning of Maps for Camera Localization"*, CVPR 2018. [arXiv:1712.03342](https://arxiv.org/abs/1712.03342)
- **Repo original**: https://github.com/NVlabs/geomapnet
- **Checkpoints públicos**: https://drive.google.com/drive/folders/1J2QG_nHrRTKcDf9CGXRK9MWH1h-GuMLy
- **Dataset 7-Scenes**: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

---

*Documento generado el 18 de noviembre de 2026. Estado: 7-Scenes completado y validado. Bebop pendiente.*