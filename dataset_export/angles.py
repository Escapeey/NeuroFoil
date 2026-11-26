"""视角采样与 GT npz 存取（与需求说明书 §3.3 ±45° 视角约定对齐）。

本模块只负责「角度 / 文件」层面的数据工程，不涉及 torch 逐像素采样
（后者见 training/pixel_dataset.py）。
"""

from __future__ import annotations

import numpy as np

from .synthetic_gt import MAX_ANGLE_DEG

ANGLE_RANGE = MAX_ANGLE_DEG  # ±45°


def angles_to_norm(angles_deg: np.ndarray) -> np.ndarray:
    """角度（度）→ 归一化 [-1,1]。"""
    return np.asarray(angles_deg, dtype=np.float32) / ANGLE_RANGE


def sample_train_angles(n: int, seed: int = 0) -> np.ndarray:
    """训练视角：均匀随机采样 (θ, φ) ∈ ±45°。返回 (n, 2) 度。"""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 2)) * 2.0 - 1.0) * ANGLE_RANGE


def sample_val_angles(k: int = 5) -> np.ndarray:
    """验证视角：规则网格 k×k（检验未见角度的插值泛化）。返回 (k*k, 2) 度。"""
    lin = np.linspace(-ANGLE_RANGE, ANGLE_RANGE, k)
    th, ph = np.meshgrid(lin, lin)
    return np.stack([th.ravel(), ph.ravel()], axis=-1).astype(np.float32)


def save_dataset(path, angles_deg: np.ndarray, images: np.ndarray) -> None:
    """保存 .npz：angles (N,2) float32 度 + images (N,res,res,4) uint8。"""
    np.savez_compressed(
        str(path),
        angles=angles_deg.astype(np.float32),
        images=images.astype(np.uint8),
    )
