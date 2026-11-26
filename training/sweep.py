"""超参扫描：产出 size ↔ quality ↔ latency 曲线（需求说明书 §6 交付物 3）。

对 (特征贴图分辨率 × MLP 宽度) 网格逐点：训练 → 评估 PSNR/SSIM → 测延迟
→ 记模型体积，结果写入 CSV。默认在 CPU 上用小训练预算做「快扫」，
完整 512 分辨率扫描留待后续。

用法：
  python -m training.sweep --card MegaDiancie --tag sweep
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

from dataset_export.layout import CardLayout

from .latency import measure
from .pixel_dataset import load_dataset
from .train import run_training

DEFAULT_CONFIGS = [
    (64, 16), (64, 32), (64, 64),
    (128, 16), (128, 32), (128, 64),
    (256, 16), (256, 32), (256, 64),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie", help="卡片名（assets/<card>/）")
    ap.add_argument("--root", default="assets", help="资产根目录")
    ap.add_argument("--tag", default="sweep", help="曲线 CSV 名（outputs/<tag>.csv）")
    ap.add_argument("--train-npz", default=None, help="覆盖默认训练集路径")
    ap.add_argument("--val-npz", default=None, help="覆盖默认验证集路径")
    ap.add_argument("--out", default=None, help="覆盖默认曲线 CSV 路径")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--feat-res", type=int, action="append",
                    help="覆盖默认特征贴图分辨率（可多次）")
    ap.add_argument("--width", type=int, action="append",
                    help="覆盖默认 MLP 宽度（可多次）")
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root)
    train_npz = args.train_npz or layout.train_npz
    val_npz = args.val_npz or layout.val_npz
    out_csv = args.out or str(layout.outputs / f"{args.tag}.csv")

    feat_res_list = args.feat_res or sorted({c[0] for c in DEFAULT_CONFIGS})
    width_list = args.width or sorted({c[1] for c in DEFAULT_CONFIGS})
    configs = [(f, w) for f in feat_res_list for w in width_list]

    device = torch.device("cpu")
    train_ds = load_dataset(train_npz, device)
    val_ds = load_dataset(val_npz, device)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["feat_res", "width", "n_params", "model_kb",
                  "psnr", "ssim", "train_loss",
                  "flops_per_pixel", "cpu_ms_512", "est_mobile_ms_400gflops"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for feat_res, width in configs:
            print(f"\n===== 配置 feat={feat_res} width={width} =====", flush=True)
            r = run_training(
                train_ds, val_ds, feat_res=feat_res, width=width,
                steps=args.steps, lr=args.lr, batch=args.batch,
                eval_every=args.eval_every, device=device, quiet=True,
            )
            lat = measure(r["model"], res=512, device=device)
            row = {
                "feat_res": feat_res,
                "width": width,
                "n_params": r["n_params"],
                "model_kb": round(r["model_bytes"] / 1024.0, 1),
                "psnr": round(r["final_psnr"], 3),
                "ssim": round(r["final_ssim"], 4),
                "train_loss": round(r["final_loss"], 5),
                "flops_per_pixel": lat["flops_per_pixel"],
                "cpu_ms_512": round(lat["cpu_ms_per_512"], 2),
                "est_mobile_ms_400gflops": round(lat["est_mobile_ms_400gflops"], 3),
            }
            writer.writerow(row)
            f.flush()
            print(f"  -> PSNR={row['psnr']}dB SSIM={row['ssim']} "
                  f"体积={row['model_kb']}KB CPU={row['cpu_ms_512']}ms/512 "
                  f"估算移动端={row['est_mobile_ms_400gflops']}ms", flush=True)

    print(f"\n完成：结果写入 {out_csv}")


if __name__ == "__main__":
    main()