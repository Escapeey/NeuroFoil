"""推理延迟与算力度量（需求说明书 §4 / §6 风险）。

PC 穿刺阶段在 CPU 上量真实墙钟 + 解析 FLOPs：
  - flops_per_pixel：单像素 MLP 的乘加数（忽略采样与激活）
  - cpu_ms_per_512：CPU 上整帧 512×512 前向的真实耗时（多次取中位数）
  - est_mobile_ms：按移动端 GPU 吞吐（默认 ~400 GFLOPs）折算的帧耗时，
    用于回答「逐像素 MLP 能否在目标帧率下不掉电」。
"""

from __future__ import annotations

import time

import torch

from .model import NeuralCard, make_uv_grid


def flops_per_pixel(width: int, n_hidden: int = 2, in_dim: int = 8) -> int:
    """单像素 MLP 乘加数（MACs）：in_dim→W→…→W→4。"""
    macs = in_dim * width + (n_hidden - 1) * width * width + width * 4
    return 2 * macs  # 乘 + 加各算一次


def measure_cpu_frame_ms(model: NeuralCard, res: int = 512,
                         device: torch.device = None, warmup: int = 3,
                         repeat: int = 5) -> float:
    """CPU 上整帧 (res×res) 前向耗时（ms），取 repeat 次中位数。"""
    device = device or torch.device("cpu")
    model.eval().to(device)
    uv = make_uv_grid(res).to(device)
    angle = torch.tensor([0.0, 0.0], dtype=uv.dtype, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(uv, angle)
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            model(uv, angle)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def estimate_mobile_frame_ms(width: int, n_hidden: int = 2,
                             res: int = 512, gflops: float = 400.0,
                             in_dim: int = 8) -> float:
    """按移动端 GPU 吞吐折算整帧耗时（ms）。"""
    total_flops = flops_per_pixel(width, n_hidden, in_dim) * res * res
    return total_flops / (gflops * 1e9) * 1e3


def measure(model: NeuralCard, res: int = 512, device: torch.device = None) -> dict:
    """汇总延迟与算力指标。"""
    width = model.mlp_width
    in_dim = model.mlp[0].in_features
    n_hidden = len([m for m in model.mlp if isinstance(m, torch.nn.Linear)]) - 1
    return {
        "flops_per_pixel": flops_per_pixel(width, n_hidden, in_dim),
        "cpu_ms_per_512": measure_cpu_frame_ms(model, res, device),
        "est_mobile_ms_400gflops": estimate_mobile_frame_ms(width, n_hidden, res, in_dim=in_dim),
        "fps_cpu": 1000.0 / max(measure_cpu_frame_ms(model, res, device), 1e-9),
    }
