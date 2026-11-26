"""GT 渲染统一门面：按卡片资产可用性在「skill 分层」与「合成占位」间调度。

export / compare / preview 都通过本模块取渲染器，避免各处重复判断逻辑。

source 取值：
  - "auto"      有 skill 分层素材则用之，否则回退合成（默认）
  - "skill"     强制用 assets/<card>/source 的真实分层素材（缺失则报错）
  - "synthetic" 强制用纯合成占位卡（无需任何资产，便于自检 / CI）
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .layout import CardLayout

Renderer = Callable[[float, float, int], np.ndarray]


def make_renderer(card: str | None = None, source: str = "auto",
                  root: str = "assets") -> tuple[Renderer, str]:
    """返回 (render(theta_deg, phi_deg, res)->uint8, 实际使用的 source)。"""
    layout = CardLayout(card, root) if card else None

    use_skill = False
    if source == "skill":
        if layout is None or not layout.has_skill_assets():
            raise FileNotFoundError(
                f"skill 分层素材不齐备：需要 {layout.source if layout else '<card>'}"
                f"/{{background,subject,lineart,text}}.png 与 card-config.json")
        use_skill = True
    elif source == "auto":
        use_skill = layout is not None and layout.has_skill_assets()
    elif source != "synthetic":
        raise ValueError(f"未知 source={source!r}（auto|skill|synthetic）")

    if use_skill:
        from .layer_render import load_config
        from .layer_render import render as lay_render
        cfg = load_config(layout)
        return (lambda th, ph, res: lay_render(layout, th, ph, res=res, cfg=cfg)), "skill"

    from .synthetic_gt import render as syn_render
    return (lambda th, ph, res: syn_render(th, ph, res=res)), "synthetic"


def generate_dataset(angles_deg: np.ndarray, card: str | None = None,
                     source: str = "auto", res: int = 512,
                     root: str = "assets") -> tuple[np.ndarray, str]:
    """批量渲染 GT，返回 ((N,res,res,4) uint8, 实际 source)。"""
    render, used = make_renderer(card, source, root)
    out = np.empty((len(angles_deg), res, res, 4), dtype=np.uint8)
    for i, (th, ph) in enumerate(angles_deg):
        out[i] = render(float(th), float(ph), res)
    return out, used
