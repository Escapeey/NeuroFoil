"""神经纹理 + MLP 模型（需求说明书 §3.2 冻结架构）。

- 一张可学习的 4 通道潜特征纹理 F ∈ R^{4×H×W}（神经纹理，隐式编码外观与深度）。
- 一个小型 MLP：输入 8 维（4 特征 + 2 uv + 2 视角），输出逐像素 RGBA。
- 特征纹理与 MLP 权重在同一反向传播中联合优化。

输入约定（与 gt_holo.py 对齐）：
  - uv ∈ [0,1]²（v=1 为卡片顶部）
  - 视角 (theta_n, phi_n) ∈ [-1,1]²，由角度（度）除以 45° 归一化得到
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCard(nn.Module):
    """可学习特征纹理 + 逐像素 MLP。

    fourier_octaves=0 时保持需求说明书冻结的 8 入参（4 特征 + 2 uv + 2 视角）；
    >0 时对 uv 额外拼接触发傅里叶位置编码（每八度 4 维 sin/cos），
    用于对抗光滑 MLP 的谱偏置、拟合文字/线稿等高频细节（需求 §5.1 遗留事项）。
    """

    def __init__(self, feat_res: int = 128, mlp_width: int = 32,
                 n_hidden: int = 2, act: str = "relu",
                 fourier_octaves: int = 0):
        super().__init__()
        self.feat_res = feat_res
        self.mlp_width = mlp_width
        self.fourier_octaves = fourier_octaves
        # 4 通道潜特征纹理（体积主导项）
        self.feature = nn.Parameter(torch.randn(4, feat_res, feat_res) * 0.1)

        uv_dim = 2 + 4 * fourier_octaves   # 原始 uv + 每八度 4 维 sin/cos
        in_dim = 4 + uv_dim + 2            # 4 特征 + uv(编码后) + 2 视角
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, mlp_width))
            layers.append(nn.ReLU() if act == "relu" else nn.SiLU())
            in_dim = mlp_width
        layers.append(nn.Linear(in_dim, 4))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def _sample_feature(self, uv: torch.Tensor) -> torch.Tensor:
        """在特征纹理上对 uv 点做双线性采样。

        uv: (N, 2) ∈ [0,1]² → (N, 4) 特征。
        """
        # grid_sample 网格坐标 (x, y) ∈ [-1, 1]，align_corners=True 时 -1↔u=0, +1↔u=1
        grid = (uv * 2.0 - 1.0).reshape(1, -1, 1, 2)  # (1, N, 1, 2)
        feat = self.feature.unsqueeze(0)              # (1, 4, H, W)
        out = F.grid_sample(feat, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)  # (1, 4, N, 1)
        return out[0, :, :, 0].t()                    # (N, 4)

    def _encode_uv(self, uv: torch.Tensor) -> torch.Tensor:
        """uv 位置编码：原始 uv + 可选傅里叶八度（sin/cos）。

        fourier_octaves=0 时返回原始 uv（等价于冻结的 8 入参架构）。
        """
        if self.fourier_octaves <= 0:
            return uv
        parts = [uv]
        for k in range(self.fourier_octaves):
            freq = (2.0 ** k) * math.pi
            parts.append(torch.sin(freq * uv))
            parts.append(torch.cos(freq * uv))
        return torch.cat(parts, dim=-1)

    def forward(self, uv: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """逐像素前向。

        uv:    (N, 2) ∈ [0,1]²
        angle: (2,) 或 (N, 2)，归一化视角 [theta_n, phi_n] ∈ [-1,1]²
        返回:  (N, 4) RGBA ∈ [0,1]（sigmoid 输出）
        """
        feat = self._sample_feature(uv)               # (N, 4)
        if angle.dim() == 1:
            angle = angle.unsqueeze(0).expand(uv.shape[0], -1)
        uv_enc = self._encode_uv(uv)
        x = torch.cat([feat, uv_enc, angle], dim=-1)  # (N, in_dim)
        return torch.sigmoid(self.mlp(x))             # (N, 4)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def render(self, theta_n: float, phi_n: float,
               res: int = 512) -> torch.Tensor:
        """渲染整帧 (res, res, 4) 浮点 RGBA，用于评估 / 查看器。"""
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / res, 1.0 - 0.5 / res, res),
            torch.linspace(0.5 / res, 1.0 - 0.5 / res, res),
            indexing="ij",
        )
        uv = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (res², 2)
        angle = torch.tensor([theta_n, phi_n], dtype=uv.dtype)
        rgba = self.forward(uv, angle)                              # (res², 4)
        return rgba.reshape(res, res, 4)

    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_bytes(self) -> int:
        """state_dict 序列化体积（fp32），近似模型文件大小。"""
        return sum(p.numel() * 4 for p in self.parameters())


def make_uv_grid(res: int) -> torch.Tensor:
    """生成 (res*res, 2) 的 uv 网格，u=x、v=y，∈[0,1]。"""
    yy, xx = torch.meshgrid(
        torch.linspace(0.5 / res, 1.0 - 0.5 / res, res),
        torch.linspace(0.5 / res, 1.0 - 0.5 / res, res),
        indexing="ij",
    )
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
