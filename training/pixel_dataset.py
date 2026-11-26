"""逐像素训练数据集：把 GT 帧摊平为「帧 × 像素」矩阵供随机批采样。

角度归一化复用 dataset_export.angles，保证训练与导出端口径一致。
"""

from __future__ import annotations

import numpy as np
import torch

from dataset_export.angles import angles_to_norm


class PixelDataset:
    """images_flat: (N, res*res, 4) float ∈ [0,1]
    angles_norm: (N, 2) float ∈ [-1,1]
    uv_grid:     (res*res, 2) float ∈ [0,1]
    """

    def __init__(self, angles_deg: np.ndarray, images: np.ndarray, device: torch.device):
        self.res = images.shape[1]
        self.angles_norm = torch.as_tensor(
            angles_to_norm(angles_deg), device=device)
        imgs = images.astype(np.float32) / 255.0
        n = imgs.shape[0]
        self.images_flat = torch.as_tensor(
            imgs.reshape(n, -1, 4), device=device)          # (N, res², 4)
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / self.res, 1.0 - 0.5 / self.res, self.res),
            torch.linspace(0.5 / self.res, 1.0 - 0.5 / self.res, self.res),
            indexing="ij",
        )
        self.uv_grid = torch.stack(
            [xx.reshape(-1), yy.reshape(-1)], dim=-1).to(device)  # (res², 2)
        self.n_frames = n
        self.n_pixels = self.res * self.res

    def sample_batch(self, batch: int, rng: torch.Generator):
        """随机采样 (batch,) 个 (帧, 像素) 对，返回 uv、归一化视角、GT RGBA。"""
        frame = torch.randint(0, self.n_frames, (batch,), generator=rng)
        pix = torch.randint(0, self.n_pixels, (batch,), generator=rng)
        uv = self.uv_grid[pix]
        angle = self.angles_norm[frame]
        gt = self.images_flat[frame, pix]
        return uv, angle, gt


def load_dataset(path, device: torch.device) -> PixelDataset:
    data = np.load(str(path))
    return PixelDataset(data["angles"], data["images"], device)
