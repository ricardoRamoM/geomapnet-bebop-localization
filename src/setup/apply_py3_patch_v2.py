#!/usr/bin/env python3
"""v2: cubre los .py que v1 no tocó + fix async= en KwArg. Idempotente."""
import re
from pathlib import Path

ROOT = Path("/data/research/mapnet/geomapnet")
FILES = [
    "common/train.py",          # re-toca: fix async=True como kwarg
    "common/visdom_utils.py",
    "common/vis_utils.py",
    "dataset_loaders/robotcar.py",
    "dataset_loaders/time_imload.py",
    "models/vidloc.py",
    "scripts/align_vo_poses.py",
    "scripts/plot_activations.py",
    "scripts/plot_vo_poses.py",
    "scripts/process_robotcar_gps.py",
    "scripts/process_robotcar_images.py",
    "scripts/reverse_vo_poses.py",
    "scripts/test_vo.py",
]


def convert_prints(src):
    lines = src.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)print\s+(.+)$", line)
        if not m:
            out.append(line); i += 1; continue
        indent, rest = m.group(1), m.group(2)
        if rest.startswith("(") and rest.endswith(")"):
            out.append(line); i += 1; continue
        buf = rest
        while buf.rstrip().endswith("\\") and i + 1 < len(lines):
            buf = buf.rstrip().rstrip("\\").rstrip() + " "
            i += 1
            buf += lines[i]
        out.append(f"{indent}print({buf})")
        i += 1
    return "\n".join(out)


def patch(path):
    src = orig = path.read_text()

    # Reemplazos token a token
    src = src.replace("xrange(", "range(")
    src = re.sub(r"\bnp\.int\b", "int", src)
    src = re.sub(r"\bnp\.float\b", "float", src)
    src = re.sub(r"\bnp\.bool\b", "bool", src)
    src = re.sub(r"np\.asscalar\(([^)]+)\)", r"float(\1)", src)

    # print statement -> print()
    src = convert_prints(src)

    # async=True como kwarg en llamadas (SyntaxError en Py3.7+)
    src = src.replace(", async=True)", ")")
    src = src.replace(", async=True,", ",")

    # vidloc.py: async como nombre de parámetro
    if "vidloc" in path.name:
        src = src.replace("def forward(self, x, cuda=False, async=True):",
                          "def forward(self, x, cuda=False, asynchronous=True):")
        src = src.replace("def forward(self, x, cuda=False, async=False):",
                          "def forward(self, x, cuda=False, asynchronous=False):")
        src = src.replace("async=async", "non_blocking=asynchronous")
        src = src.replace("Variable(None)", "None")

    if src != orig:
        path.write_text(src)
        print(f"[PATCH] {path}")
    else:
        print(f"[SKIP ] {path}")


if __name__ == "__main__":
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            patch(p)
        else:
            print(f"[MISS ] {p}")
