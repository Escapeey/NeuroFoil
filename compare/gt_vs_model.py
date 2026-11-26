"""效果对比演示：GT 多视角拼图 + 「GT vs 模型重建」并排对比图。

读取 assets/<card>/outputs/<tag>/model.pt（架构参数从同目录 metrics.json 恢复），
用与训练集相同的渲染源重绘 GT，逐视角与模型输出并排，产出 PNG 供目检。

用法：
  python -m compare.gt_vs_model --card MegaDiancie --tag demo --source synthetic
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset_export.api import make_renderer
from dataset_export.layout import CardLayout
from training.model import NeuralCard

# GT 拼图 3×3 视角（θ 行 × φ 列）
MONTAGE_ANGLES = [
    (-45, -45), (-45, 0), (-45, 45),
    (0, -45), (0, 0), (0, 45),
    (45, -45), (45, 0), (45, 45),
]
# 对比行视角
COMPARE_ANGLES = [(-40, -40), (-40, 0), (0, 0), (40, 0), (0, 40), (40, 40)]


def _load_model(layout: CardLayout, tag: str) -> NeuralCard:
    mdir = layout.model_dir(tag)
    with open(mdir / "metrics.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    model = NeuralCard(feat_res=m.get("feat_res", 64),
                       mlp_width=m.get("width", 32),
                       fourier_octaves=m.get("fourier_octaves", 0))
    model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
    model.eval()
    return model


def build(layout: CardLayout, tag: str, res: int, source: str, root: str) -> None:
    render, used = make_renderer(layout.card, source, root)
    model = _load_model(layout, tag)

    # 1) GT 多视角拼图 3×3
    canvas = Image.new("RGBA", (res * 3, res * 3), (20, 20, 24, 255))
    for i, (t, p) in enumerate(MONTAGE_ANGLES):
        canvas.paste(Image.fromarray(render(t, p, res)),
                     ((i % 3) * res, (i // 3) * res))
    montage_path = layout.outputs / f"{tag}_gt_montage.png"
    canvas.convert("RGB").save(montage_path)

    # 2) GT vs 模型重建（左 GT，右 模型）
    H, W = res, res * 2
    canvas2 = Image.new("RGBA", (W, H * len(COMPARE_ANGLES)), (20, 20, 24, 255))
    dr = ImageDraw.Draw(canvas2)
    with torch.no_grad():
        for r, (t, p) in enumerate(COMPARE_ANGLES):
            gt = render(t, p, res)
            pr = model.render(t / 45.0, p / 45.0, res=res).numpy()
            pr = (np.clip(pr, 0, 1) * 255).astype(np.uint8)
            canvas2.paste(Image.fromarray(gt), (0, r * H))
            canvas2.paste(Image.fromarray(pr), (res, r * H))
            dr.text((6, r * H + 4), "GT", fill=(255, 80, 80))
            dr.text((res + 6, r * H + 4), "MODEL", fill=(80, 255, 120))
    cmp_path = layout.outputs / f"{tag}_gt_vs_model.png"
    canvas2.convert("RGB").save(cmp_path)

    print(f"渲染源={used}  模型={tag}")
    print(f"已保存 {montage_path}")
    print(f"已保存 {cmp_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie")
    ap.add_argument("--root", default="assets")
    ap.add_argument("--tag", default="model", help="模型子目录（outputs/<tag>/model.pt）")
    ap.add_argument("--source", default="auto", choices=["auto", "skill", "synthetic"],
                    help="需与训练集渲染源一致")
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root).ensure()
    build(layout, args.tag, args.res, args.source, args.root)


if __name__ == "__main__":
    main()
