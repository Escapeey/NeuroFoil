"""导出训练集：渲染多视角 GT → assets/<card>/dataset/{train,val}.npz。

这是「从 skill 演示结果导出训练集」的入口（需求说明书 §6 交付物 1 的数据端）。
默认 auto：若 assets/<card>/source 有 skill 分层素材则渲染真实卡，否则回退合成占位卡。

用法：
  # 真实卡（先在 card_studio 跑完 skill，素材落到 assets/MegaDiancie/source）
  python -m dataset_export.export --card MegaDiancie --res 256 --train 60 --val-grid 5

  # 合成占位卡（无需任何资产，链路自检）
  python -m dataset_export.export --card MegaDiancie --source synthetic --res 128
"""

from __future__ import annotations

import argparse

from .angles import sample_train_angles, sample_val_angles, save_dataset
from .api import generate_dataset
from .layout import CardLayout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie", help="卡片名（assets/<card>/）")
    ap.add_argument("--root", default="assets", help="资产根目录")
    ap.add_argument("--source", default="auto", choices=["auto", "skill", "synthetic"])
    ap.add_argument("--train", type=int, default=60, help="训练帧数（随机视角）")
    ap.add_argument("--val-grid", type=int, default=5, help="验证网格 k（k×k 帧）")
    ap.add_argument("--res", type=int, default=256, help="渲染分辨率")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root).ensure()
    train_angles = sample_train_angles(args.train, seed=args.seed)
    val_angles = sample_val_angles(args.val_grid)

    print(f"渲染训练集 {args.train} 帧 @{args.res}×{args.res} …", flush=True)
    train_imgs, used = generate_dataset(
        train_angles, card=args.card, source=args.source,
        res=args.res, root=args.root)
    save_dataset(layout.train_npz, train_angles, train_imgs)

    print(f"渲染验证集 {len(val_angles)} 帧 @{args.res}×{args.res} …", flush=True)
    val_imgs, _ = generate_dataset(
        val_angles, card=args.card, source=args.source,
        res=args.res, root=args.root)
    save_dataset(layout.val_npz, val_angles, val_imgs)

    print(f"完成（渲染源={used}）：")
    print(f"  {layout.train_npz}  {train_imgs.shape}")
    print(f"  {layout.val_npz}  {val_imgs.shape}")


if __name__ == "__main__":
    main()
