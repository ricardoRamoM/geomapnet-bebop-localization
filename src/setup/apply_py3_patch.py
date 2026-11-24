#!/usr/bin/env python3
"""
Port mínimo de geomapnet (Py2 / PyTorch 0.4) a Python 3.11 / PyTorch 2.4.
Idempotente: se puede ejecutar varias veces sin romper.
Solo cambia sintaxis/APIs obsoletas, no lógica.
"""
import re
from pathlib import Path

ROOT = Path("/data/research/mapnet/geomapnet")

FILES = [
    "dataset_loaders/seven_scenes.py",
    "dataset_loaders/utils.py",
    "dataset_loaders/composite.py",
    "common/pose_utils.py",
    "common/criterion.py",
    "common/train.py",
    "common/optimizer.py",
    "common/Logger.py",
    "models/posenet.py",
    "scripts/eval.py",
    "scripts/train.py",
    "scripts/calc_pose_stats.py",
    "scripts/dataset_mean.py",
]


def _count_parens(s: str) -> int:
    """Cuenta paréntesis balanceados ignorando strings (aproximado)."""
    depth = 0
    in_str = None
    esc = False
    for c in s:
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if in_str:
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return depth


def convert_prints(src: str) -> str:
    """Convierte statements `print x` (Py2) a `print(x)` (Py3), multi-línea."""
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)print\s+(.+)$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent, rest = m.group(1), m.group(2)
        # ¿ya está en formato print(...)?
        if rest.startswith("(") and rest.endswith(")") and _count_parens(rest) == 0:
            out.append(line)
            i += 1
            continue
        # acumular líneas hasta paréntesis balanceados
        buf = rest
        depth = _count_parens(rest)
        while (depth > 0 or buf.rstrip().endswith("\\")) and i + 1 < len(lines):
            buf = buf.rstrip().rstrip("\\").rstrip() + " "
            i += 1
            buf += lines[i]
            depth = _count_parens(buf)
        out.append(f"{indent}print({buf})")
        i += 1
    return "\n".join(out)


def patch(path: Path) -> bool:
    src = orig = path.read_text()

    # ----- Reemplazos token a token (seguros en todo el repo) -----
    src = src.replace("xrange(", "range(")
    src = src.replace("cPickle", "pickle")
    src = re.sub(r"\bnp\.int\b", "int", src)
    src = re.sub(r"\bnp\.float\b", "float", src)
    src = re.sub(r"\bnp\.bool\b", "bool", src)
    src = re.sub(r"np\.asscalar\(([^)]+)\)", r"float(\1)", src)
    src = src.replace("settings.read_file(f)", "settings.read_string(f.read())")
    src = src.replace("transforms.Scale(", "transforms.Resize(")
    src = src.replace("register_backward_hook", "register_full_backward_hook")

    # ----- print statements Py2 -> print() Py3 (multi-línea) -----
    src = convert_prints(src)

    # ----- has_key -----
    src = re.sub(r"(\w+)\.has_key\(([^)]+)\)", r"\2 in \1", src)

    # ----- .cuda(async=True) -----
    src = src.replace(".cuda(async=True)", ".cuda(non_blocking=True)")

    # ----- loss.data[0] -> loss.item() -----
    src = re.sub(r"(\w+)\.data\[0\]", r"\1.item()", src)

    # ----- filter() de safe_collate (en Py3 es iterador) -----
    src = src.replace(
        "batch = filter(lambda x: x is not None, batch)",
        "batch = list(filter(lambda x: x is not None, batch))",
    )

    # ----- Variable(...) -----
    src = re.sub(
        r"Variable\((\w+),\s*volatile=\(not train\),\s*requires_grad=train\)",
        r"\1", src,
    )
    src = re.sub(
        r"Variable\((\w+),\s*requires_grad=True\)",
        r"\1.requires_grad_(True)", src,
    )
    src = re.sub(
        r"Variable\((\w+),\s*requires_grad=False\)",
        r"\1.detach()", src,
    )
    src = re.sub(r"Variable\((\w+_idx)\)", r"\1", src)

    # ----- torch.load con weights_only=False (PT >= 2.6 lo pone a True por defecto) -----
    if "weights_only" not in src:
        src = re.sub(
            r"torch\.load\(([^,]+),\s*map_location=([^)]+)\)",
            r"torch.load(\1, map_location=\2, weights_only=False)", src,
        )

    # ----- torchvision: pretrained -> weights -----
    if "pretrained=False" in src:
        src = src.replace(
            "models.resnet34(pretrained=False)",
            "models.resnet34(weights=None)",
        )
    if "pretrained=True" in src:
        src = src.replace(
            "models.resnet34(pretrained=True)",
            "models.resnet34(weights=ResNet34_Weights.DEFAULT)",
        )
        if "from torchvision.models import ResNet34_Weights" not in src:
            src = src.replace(
                "from torchvision import transforms, models",
                "from torchvision import transforms, models\n"
                "from torchvision.models import ResNet34_Weights",
            )

    # ----- divisiones enteras (Py2 / ya era floor en ints) -----
    replacements = [
        ("len(idx) / 2", "len(idx) // 2"),
        ("len(output) / 2", "len(output) // 2"),
        ("len(target) / 2", "len(target) // 2"),
        ("len(offsets) / 2", "len(offsets) // 2"),
        ("self.steps/2", "self.steps//2"),
        ("self.steps / 2", "self.steps // 2"),
        ("N/B", "N//B"),
        ("T/G", "T//G"),
    ]
    for old, new in replacements:
        src = src.replace(old, new)

    # ----- imports rotos dentro de paquetes -----
    src = src.replace(
        "from seven_scenes import SevenScenes",
        "from dataset_loaders.seven_scenes import SevenScenes",
    )
    src = src.replace(
        "from robotcar import RobotCar",
        "from dataset_loaders.robotcar import RobotCar",
    )

    if src != orig:
        path.write_text(src)
        print(f"[PATCH] {path}")
        return True
    print(f"[SKIP ] {path}")
    return False


if __name__ == "__main__":
    any_patched = False
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            any_patched |= patch(p)
        else:
            print(f"[MISS ] {p}")
    if not any_patched:
        print("\nNada que cambiar: el repo ya estaba parcheado.")
