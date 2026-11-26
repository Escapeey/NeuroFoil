"""画质指标：PSNR / SSIM（需求说明书 §5 验收标准）。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """RGB 通道 PSNR（dB）。输入 (H, W, 3) 或 (H, W, 4)，取 RGB，∈[0,1]。"""
    gt = np.asarray(gt, dtype=np.float32)[..., :3]
    pred = np.asarray(pred, dtype=np.float32)[..., :3]
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _gaussian_window(kernel_size: int = 11, sigma: float = 1.5,
                     channels: int = 3) -> torch.Tensor:
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """RGB 三通道平均 SSIM，返回标量 ∈ [0,1]。输入 (H, W, 3/4) ∈ [0,1]。"""
    gt = torch.as_tensor(np.asarray(gt, dtype=np.float32)[..., :3]).permute(2, 0, 1)[None]
    pred = torch.as_tensor(np.asarray(pred, dtype=np.float32)[..., :3]).permute(2, 0, 1)[None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = _gaussian_window().to(gt.dtype)

    def _ssim(a, b):
        mu_a = F.conv2d(a, win, padding=0, groups=a.shape[1])
        mu_b = F.conv2d(b, win, padding=0, groups=b.shape[1])
        mu_a2, mu_b2 = mu_a ** 2, mu_b ** 2
        sig_a = F.conv2d(a * a, win, padding=0, groups=a.shape[1]) - mu_a2
        sig_b = F.conv2d(b * b, win, padding=0, groups=b.shape[1]) - mu_b2
        sig_ab = F.conv2d(a * b, win, padding=0, groups=a.shape[1]) - mu_a * mu_b
        return ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sig_a + sig_b + C2))

    # 统一裁剪到同一尺寸（卷积 padding=0 会缩小 10px）
    s = _ssim(gt, pred).mean().item()
    return float(np.clip(s, 0.0, 1.0))
