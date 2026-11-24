#!/usr/bin/env bash
set -u
REPO=/data/research/mapnet/geomapnet
OUT=/data/research/mapnet/diagnostico_mapnet_$(date +%Y%m%d_%H%M).txt

mkdir -p /data/research/mapnet
exec > >(tee "$OUT") 2>&1

echo "===================== DIAGNÓSTICO MAPNET ====================="
echo "Fecha        : $(date)"
echo "Host         : $(hostname)"
echo "Usuario      : $(whoami)"
echo "Salida en    : $OUT"
echo

echo "########## 0. ENTORNO ##########"
echo "--- uname ---"; uname -a
echo "--- python (sistema) ---"; python3 --version 2>&1
echo "--- conda ---"; conda --version 2>&1
echo "--- nvidia-smi (resumen) ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
echo "--- conda envs ---"; conda env list 2>&1
echo

echo "########## 1. ESTRUCTURA DEL REPO ##########"
if [ ! -d "$REPO" ]; then
  echo "ERROR: no existe $REPO"
  echo "Buscando posibles repos:"
  find /data/research -maxdepth 4 -type d -name '*apnet*' 2>/dev/null
  exit 1
fi
cd "$REPO" || exit 1
pwd
echo
echo "--- git (si aplica) ---"
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "(no es repo git)"
git log -1 --oneline 2>/dev/null || true
git status --short 2>/dev/null | head -50 || true
echo
echo "--- Árbol (2 niveles, sin .git) ---"
find . -maxdepth 2 -not -path './.git*' | sort | head -300
echo

echo "########## 2. CHECKPOINTS / PESOS PREENTRENADOS ##########"
find . -maxdepth 4 \( -name '*.pth' -o -name '*.pth.tar' -o -name '*.ckpt' -o -name '*.tar' \) \
     -not -path './.git/*' -exec ls -lh {} \; 2>/dev/null
echo

echo "########## 3. A. dataset_loaders/seven_scenes.py ##########"
if [ -f dataset_loaders/seven_scenes.py ]; then
  sed -n '1,280p' dataset_loaders/seven_scenes.py
else
  echo "(no existe dataset_loaders/seven_scenes.py)"; ls -la dataset_loaders 2>/dev/null
fi
echo

echo "########## 4. B. scripts/set_paths.py ##########"
if [ -f scripts/set_paths.py ]; then
  sed -n '1,220p' scripts/set_paths.py
else
  echo "(no existe scripts/set_paths.py)"
fi
echo

echo "########## 5. B. dataset_loaders/README.md ##########"
if [ -f dataset_loaders/README.md ]; then
  cat dataset_loaders/README.md
else
  echo "(no existe dataset_loaders/README.md)"
fi
echo

echo "########## 6. B. scripts/configs/mapnet.ini ##########"
if [ -f scripts/configs/mapnet.ini ]; then
  cat scripts/configs/mapnet.ini
else
  echo "(no existe scripts/configs/mapnet.ini)"; ls -la scripts/configs 2>/dev/null
fi
echo

echo "########## 7. C. models/posenet.py ##########"
if [ -f models/posenet.py ]; then
  sed -n '1,300p' models/posenet.py
else
  echo "(no existe models/posenet.py)"
fi
echo

echo "########## 8. D. common/criterion.py ##########"
if [ -f common/criterion.py ]; then
  sed -n '1,280p' common/criterion.py
else
  echo "(no existe common/criterion.py)"
fi
echo

echo "########## 9. D. common/train.py ##########"
if [ -f common/train.py ]; then
  sed -n '1,280p' common/train.py
else
  echo "(no existe common/train.py)"
fi
echo

echo "########## 10. E. SINTAXIS INCOMPATIBLE CON PYTHON 3 (grep original) ##########"
grep -RniE 'cPickle|xrange|iteritems|raw_input|print [^(]' \
    common models dataset_loaders scripts \
    --include='*.py' 2>/dev/null || echo "(sin coincidencias)"
echo

echo "########## 11. E-bis. OTROS PROBLEMAS TÍPICOS Py2 / PyTorch 0.4 ##########"
grep -RniE 'Variable\(|volatile|\.data\[0\]|scipy\.misc|imread|np\.float\b|np\.int\b|np\.bool\b|torch\.utils\.serialization|async\b' \
    common models dataset_loaders scripts \
    --include='*.py' 2>/dev/null || echo "(sin coincidencias)"
echo
echo "--- usos de .cuda() ---"
grep -RniE '\.cuda\(' common models dataset_loaders scripts --include='*.py' 2>/dev/null | wc -l
echo
echo "--- imports de torch / torchvision ---"
grep -RniE '^(import torch|from torch|import torchvision|from torchvision)' \
    common models dataset_loaders scripts --include='*.py' 2>/dev/null | sort -u
echo

echo "########## 12. FICHEROS .py LISTADOS ##########"
find common models dataset_loaders scripts -name '*.py' 2>/dev/null | sort
echo

echo "===================== FIN ====================="
