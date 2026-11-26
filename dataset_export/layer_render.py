"""基于 holo-card-studio 分层素材的真实 GT 渲染器。

读取 assets/<card>/source/ 下的四层透明 PNG（background / subject / lineart / text）
与 card-config.json 的深度参数，用与合成渲染器相同的透视相机 + 逐层视差模型，
自后向前 alpha 合成，并叠加随视角变化的镭射 foil 与星光，产出训练用 GT 帧。

这是「从 skill 演示结果导出训练集」的核心工程：skill 给出真实分层美术 + 深度，
本模块把它渲染成多视角 GT（替代 spike 阶段的纯合成占位卡）。

近似说明：skill 的 web 查看器渲染的是 Blender/GLB 场景（foil、glitter 已烘焙）。
本模块用 2D 分层 + 解析 foil/sparkle 近似该外观，足以驱动神经渲染穿刺；
若需像素级对齐 GLB，可后续改为直接对 GLB 做多视角离屏渲染。

角度 / 相机约定与 synthetic_gt 完全一致（θ 俯仰、φ 旋转，±45°，相机轨道注视原点）。
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .layout import LAYER_NAMES, CardLayout
from .synthetic_gt import (
    MAX_ANGLE_DEG,
    _camera,
    _foil_background,
    _ray_grid,
    _view_specular,
    _voronoi_sparkle,
)

# 各层默认深度（card 半高 = 1.0，正方向朝相机）；config 缺省时使用
DEFAULT_DEPTHS = {
    "background": -0.25,
    "subject": 0.00,
    "lineart": 0.03,
    "text": 0.06,
}


def load_config(layout: CardLayout) -> dict:
    """读取 card-config.json；缺失则返回默认参数。"""
    if layout.config_path.exists():
        with open(layout.config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"parameters": {}}


def _layer_depths(cfg: dict) -> dict:
    """从 config.parameters 推断各层深度（subjectDepth / backgroundDepth 优先）。"""
    p = cfg.get("parameters", {}) or {}
    subj = float(p.get("subjectDepth", DEFAULT_DEPTHS["subject"]))
    bg = float(p.get("backgroundDepth", DEFAULT_DEPTHS["background"]))
    return {
        "background": bg,
        "subject": subj,
        "lineart": float(p.get("lineartDepth", subj + 0.03)),
        "text": float(p.get("textDepth", subj + 0.06)),
    }


def load_layers(layout: CardLayout, res: int) -> dict:
    """加载四层 PNG 并缩放到 (res, res, 4) float ∈ [0,1]。缺失层用全透明占位。"""
    layers = {}
    for name in LAYER_NAMES:
        path = layout.layer_png(name)
        if path.exists():
            img = Image.open(path).convert("RGBA").resize((res, res), Image.LANCZOS)
            layers[name] = np.asarray(img).astype(np.float32) / 255.0
        else:
            layers[name] = np.zeros((res, res, 4), dtype=np.float32)
    return layers


def _sample_tex(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """双线性采样纹理 tex (H,W,4)，u/v ∈ [0,1]（v=1 为卡片顶部 → 图像第 0 行）。

    返回 (..., 4)。越界坐标被 clamp 到边缘。
    """
    H, W = tex.shape[:2]
    x = np.clip(u, 0.0, 1.0) * (W - 1)
    y = np.clip(1.0 - v, 0.0, 1.0) * (H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = tex[y0, x0] * (1 - wx) + tex[y0, x1] * wx
    bot = tex[y1, x0] * (1 - wx) + tex[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def render(layout: CardLayout, theta_deg: float, phi_deg: float,
           res: int = 512, cfg: dict | None = None,
           return_float: bool = False) -> np.ndarray:
    """渲染单帧真实分层卡：给定视角 (θ, φ) 度，返回 (res, res, 4)。"""
    cfg = cfg or load_config(layout)
    p = cfg.get("parameters", {}) or {}
    foil_w = float(p.get("foil", 1.0))
    part_w = float(p.get("particles", 1.0))
    depths = _layer_depths(cfg)
    layers = load_layers(layout, res)

    theta_n = theta_deg / MAX_ANGLE_DEG
    phi_n = phi_deg / MAX_ANGLE_DEG

    P, forward, right, up = _camera(theta_deg, phi_deg)
    ray = _ray_grid(res, forward, right, up)
    view_dir = -ray
    light = np.array([0.35, 0.55, 0.75])
    light = light / np.linalg.norm(light)
    normal = np.array([0.0, 0.0, 1.0])
    spec = _view_specular(view_dir, normal, light, shininess=14.0)

    out_rgb = np.zeros((res, res, 3), dtype=np.float32)
    out_a = np.zeros((res, res), dtype=np.float32)

    def intersect(dz):
        denom = ray[..., 2]
        denom = np.where(np.abs(denom) < 1e-6, -1e-6, denom)
        t = (dz - P[2]) / denom
        x = P[0] + t * ray[..., 0]
        y = P[1] + t * ray[..., 1]
        u = (x + 1.0) / 2.0
        v = (y + 1.0) / 2.0
        inside = ((x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)).astype(np.float32)
        return u, v, inside

    def composite(rgb, a):
        nonlocal out_rgb, out_a
        a = a[..., None] if a.ndim == 2 else a
        out_rgb = out_rgb * (1.0 - a) + rgb * a
        out_a = out_a * (1.0 - a[..., 0]) + a[..., 0]

    # 1) 背景层（含随视角的镭射 foil 叠加）
    u, v, inside = intersect(depths["background"])
    bg = _sample_tex(layers["background"], u, v)
    composite(bg[..., :3], bg[..., 3] * inside)
    if foil_w > 0:
        foil = _foil_background(u, v, theta_n, phi_n, spec)
        foil_a = np.clip(0.30 * foil_w * (0.35 + 0.65 * spec), 0.0, 1.0) * inside
        composite(foil, foil_a)

    # 2) 星光（衬在主体之后）
    if part_w > 0:
        u, v, inside = intersect(depths["background"])
        star_rgb, star_a = _voronoi_sparkle(u, v, theta_n, phi_n)
        composite(star_rgb, star_a * part_w * inside)

    # 3) 主体 / 4) 线稿 / 5) 文字
    for name in ("subject", "lineart", "text"):
        u, v, inside = intersect(depths[name])
        tex = _sample_tex(layers[name], u, v)
        composite(tex[..., :3], tex[..., 3] * inside)

    rgba = np.clip(np.concatenate([out_rgb, out_a[..., None]], axis=-1), 0.0, 1.0)
    if return_float:
        return rgba
    return (rgba * 255.0).astype(np.uint8)


def generate_dataset(layout: CardLayout, angles_deg: np.ndarray,
                     res: int = 512, cfg: dict | None = None) -> np.ndarray:
    """批量渲染：angles_deg (N,2)，返回 (N, res, res, 4) uint8。"""
    cfg = cfg or load_config(layout)
    out = np.empty((len(angles_deg), res, res, 4), dtype=np.uint8)
    for i, (th, ph) in enumerate(angles_deg):
        out[i] = render(layout, float(th), float(ph), res=res, cfg=cfg)
    return out
