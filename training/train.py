"""训练入口：联合优化特征纹理 + MLP，周期评估 PSNR / SSIM。

资源约定：默认从 assets/<card>/dataset 读取训练集、向 assets/<card>/outputs/<tag>
写产物；也可用 --train-npz / --val-npz / --out 显式覆盖。

用法（也可被 sweep 以函数形式调用）：
  python -m training.train --card MegaDiancie --feat-res 64 --width 32 \
      --steps 2000 --tag demo
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset_export.layout import CardLayout

from .metrics import psnr, ssim
from .model import NeuralCard, make_uv_grid
from .pixel_dataset import PixelDataset, load_dataset


def render_frame(model: NeuralCard, theta_n: float, phi_n: float,
                 res: int, device: torch.device) -> np.ndarray:
    """渲染整帧为 numpy float (res, res, 4)。"""
    model.eval()
    uv = make_uv_grid(res).to(device)
    angle = torch.tensor([theta_n, phi_n], dtype=uv.dtype, device=device)
    with torch.no_grad():
        rgba = model(uv, angle).reshape(res, res, 4).cpu().numpy()
    return rgba


def evaluate(model: NeuralCard, ds: PixelDataset, device: torch.device,
             max_frames: int | None = None) -> dict:
    """在验证集整帧渲染并计算 PSNR / SSIM（RGB 通道）。"""
    model.eval()
    angles_deg = ds.angles_norm.cpu().numpy() * 45.0
    images = (ds.images_flat.reshape(ds.n_frames, ds.res, ds.res, 4)
              .cpu().numpy())  # float [0,1]
    n = ds.n_frames if max_frames is None else min(ds.n_frames, max_frames)
    ps, ss = [], []
    with torch.no_grad():
        for i in range(n):
            th_n, ph_n = ds.angles_norm[i].tolist()
            pred = render_frame(model, th_n, ph_n, ds.res, device)
            gt = images[i]
            ps.append(psnr(gt, pred))
            ss.append(ssim(gt, pred))
    return {
        "psnr_mean": float(np.mean(ps)),
        "ssim_mean": float(np.mean(ss)),
        "psnr_list": [float(p) for p in ps],
        "ssim_list": [float(s) for s in ss],
    }


def run_training(train_ds: PixelDataset, val_ds: PixelDataset,
                 feat_res: int, width: int, n_hidden: int = 2, act: str = "relu",
                 fourier_octaves: int = 0, steps: int = 2000, lr: float = 1e-3,
                 batch: int = 8192, eval_every: int = 500, seed: int = 0,
                 device: torch.device = None, quiet: bool = False) -> dict:
    """训练并返回最终指标。"""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = NeuralCard(feat_res=feat_res, mlp_width=width,
                       n_hidden=n_hidden, act=act,
                       fourier_octaves=fourier_octaves).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator(device=device).manual_seed(seed)

    loss_hist: list[float] = []
    evals: list[dict] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        uv, angle, gt = train_ds.sample_batch(batch, rng)
        pred = model(uv, angle)
        loss = F.mse_loss(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())

        if step % eval_every == 0 or step == steps:
            m = evaluate(model, val_ds, device)
            m["step"] = step
            m["train_loss"] = float(np.mean(loss_hist[-eval_every:]))
            evals.append(m)
            if not quiet:
                print(f"[step {step:6d}] loss={m['train_loss']:.4f} "
                      f"val PSNR={m['psnr_mean']:.2f}dB SSIM={m['ssim_mean']:.4f} "
                      f"({time.time()-t0:.1f}s)", flush=True)

    final = evals[-1]
    return {
        "model": model,
        "feat_res": feat_res,
        "width": width,
        "n_params": model.n_params(),
        "model_bytes": model.model_bytes(),
        "final_psnr": final["psnr_mean"],
        "final_ssim": final["ssim_mean"],
        "final_loss": final["train_loss"],
        "loss_hist": loss_hist,
        "evals": evals,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie", help="卡片名（assets/<card>/）")
    ap.add_argument("--root", default="assets", help="资产根目录")
    ap.add_argument("--tag", default="model",
                    help="产物子目录名（assets/<card>/outputs/<tag>）")
    ap.add_argument("--train-npz", default=None, help="覆盖默认训练集路径")
    ap.add_argument("--val-npz", default=None, help="覆盖默认验证集路径")
    ap.add_argument("--out", default=None, help="覆盖默认产物目录")
    ap.add_argument("--feat-res", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--fourier-octaves", type=int, default=0,
                    help="uv 傅里叶位置编码八度数（0=冻结 8 入参）")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root)
    train_npz = args.train_npz or layout.train_npz
    val_npz = args.val_npz or layout.val_npz
    out_dir = args.out or str(layout.model_dir(args.tag))

    device = torch.device("cpu")
    train_ds = load_dataset(train_npz, device)
    val_ds = load_dataset(val_npz, device)

    result = run_training(
        train_ds, val_ds,
        feat_res=args.feat_res, width=args.width, n_hidden=args.n_hidden,
        fourier_octaves=args.fourier_octaves,
        steps=args.steps, lr=args.lr, batch=args.batch,
        eval_every=args.eval_every, seed=args.seed, device=device,
    )

    print(f"\n=== 结果 ===")
    print(f"模型参数: {result['n_params']}  体积: {result['model_bytes']/1024:.1f} KB")
    print(f"val PSNR: {result['final_psnr']:.2f} dB   SSIM: {result['final_ssim']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(result["model"].state_dict(), os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "card": args.card,
                "feat_res": result["feat_res"],
                "width": result["width"],
                "fourier_octaves": args.fourier_octaves,
                "n_params": result["n_params"],
                "model_bytes": result["model_bytes"],
                "final_psnr": result["final_psnr"],
                "final_ssim": result["final_ssim"],
                "final_loss": result["final_loss"],
            }, f, indent=2)
        print(f"已保存模型与指标到 {out_dir}")


if __name__ == "__main__":
    main()