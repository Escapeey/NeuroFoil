"""卡片资产目录布局（assets/<CardName>/…）。

所有训练 / 推理 / 对比代码统一通过 CardLayout 解析资源路径，
保证「资产集中在 assets/ 下、代码与数据解耦」。

目录约定：
  assets/<card>/
    ├── source/    原始图片（skill 的分层 PNG：background/subject/lineart/text）+ card-config.json
    ├── html/      skill 产出的演示 HTML（web 查看器）
    ├── dataset/   导出的训练集 / 验证集（.npz）
    └── outputs/   训练产物（model.pt、metrics.json、对比图、曲线 CSV）
"""

from __future__ import annotations

from pathlib import Path

# skill 分层素材文件名（holographic 路线）
LAYER_NAMES = ("background", "subject", "lineart", "text")


class CardLayout:
    """单张卡片的资产路径解析器。

    用法：
        lay = CardLayout("MegaDiancie")          # 默认 root=assets
        lay.source / lay.html / lay.dataset / lay.outputs
        lay.layer_png("subject")                 # source/subject.png
        lay.config_path                          # source/card-config.json
        lay.train_npz / lay.val_npz              # dataset/{train,val}.npz
    """

    def __init__(self, card: str, root: str | Path = "assets"):
        self.card = card
        self.root = Path(root) / card
        self.source = self.root / "source"
        self.html = self.root / "html"
        self.dataset = self.root / "dataset"
        self.outputs = self.root / "outputs"

    # ------------------------------------------------------------------
    def ensure(self) -> "CardLayout":
        """创建全部子目录（幂等）。"""
        for d in (self.source, self.html, self.dataset, self.outputs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # --- source 素材 ---
    def layer_png(self, name: str) -> Path:
        return self.source / f"{name}.png"

    @property
    def config_path(self) -> Path:
        return self.source / "card-config.json"

    def has_skill_assets(self) -> bool:
        """skill 分层素材是否齐备（用于 auto 选择渲染源）。"""
        return self.config_path.exists() and all(
            self.layer_png(n).exists() for n in LAYER_NAMES)

    # --- dataset 产物 ---
    @property
    def train_npz(self) -> Path:
        return self.dataset / "train.npz"

    @property
    def val_npz(self) -> Path:
        return self.dataset / "val.npz"

    # --- outputs 产物 ---
    def model_dir(self, tag: str) -> Path:
        return self.outputs / tag

    def __repr__(self) -> str:  # pragma: no cover
        return f"CardLayout(card={self.card!r}, root={self.root})"
