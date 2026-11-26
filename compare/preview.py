"""渲染 GT 帧到 PNG 拼图，用于目检（PC 实时查看器的简化前身）。

渲染源由 dataset_export.api 统一调度：有 skill 分层素材则渲染真实卡，否则合成占位卡。
默认输出到 assets/<card>/outputs/。

用法：
  python -m compare.preview --card MegaDiancie --angles "0,0;-30,20;45,-45" --res 256
"""

from __future__ import annotations

import argparse

from PIL import Image

from dataset_export.api import make_renderer
from dataset_export.layout import CardLayout


def preview_gt(angles: list[tuple[float, float]], res: int, out: str,
               card: str | None = None, source: str = "auto",
               root: str = "assets") -> str:
    render, used = make_renderer(card, source, root)
    frames = [render(th, ph, res) for th, ph in angles]
    cols = min(len(frames), 4)
    rows = (len(frames) + cols - 1) // cols
    canvas = Image.new("RGBA", (res * cols, res * rows), (20, 20, 24, 255))
    for i, fr in enumerate(frames):
        canvas.paste(Image.fromarray(fr), ((i % cols) * res, (i // cols) * res))
    canvas.convert("RGB").save(out)
    print(f"已保存 {out}（渲染源={used}，{len(frames)} 帧，{cols}×{rows} 拼图）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie")
    ap.add_argument("--root", default="assets")
    ap.add_argument("--source", default="auto", choices=["auto", "skill", "synthetic"])
    ap.add_argument("--angles", default="0,0;-40,-30;0,0;40,30",
                    help="分号分隔的视角列表，如 \"0,0;-40,-30;40,30\"")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default=None, help="默认 assets/<card>/outputs/gt_preview.png")
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root).ensure()
    out = args.out or str(layout.outputs / "gt_preview.png")
    pairs = [s.split(",") for s in args.angles.split(";") if s.strip()]
    preview_gt([(float(th), float(ph)) for th, ph in pairs], args.res, out,
               card=args.card, source=args.source, root=args.root)


if __name__ == "__main__":
    main()
