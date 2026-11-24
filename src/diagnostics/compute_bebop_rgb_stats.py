#!/usr/bin/env python3
"""Calcula stats.txt RGB para Bebop usando solo TrainSplit/poses.txt.

Ejecutar desde la raíz de geomapnet:
    python compute_bebop_rgb_stats.py
"""
from dataset_loaders.bebop import compute_rgb_stats

if __name__ == "__main__":
    compute_rgb_stats(
        data_path="../data/deepslam_data/Bebop",
        scene="bebop",
    )
