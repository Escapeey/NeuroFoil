"""驱动 holo-card-studio skill 产出可用演示 HTML（web 查看器）。

工作流（需求说明书 §6 交付物的「数据源头」）：
  1. 准备 assets/<card>/source/ 下四层透明 PNG（background/subject/lineart/text）
     与 card-config.json（可从 card_studio/card-config.template.json 复制修改）。
  2. 本脚本把素材暂存为 skill 工程目录 assets/<card>/studio_project/
     （skill 约定：<project>/assets/*.png + <project>/card-config.json）。
  3. 调用 skill 的 scripts/run_pipeline.py 构建 Blender 场景、渲染、导出 GLB、
     生成白色 web 查看器。
  4. 把产出的 web 查看器收集到 assets/<card>/html/（即「可用的演示 HTML」），
     渲染预览图收集到 assets/<card>/outputs/。

skill 本体装在用户级目录（默认 ~/.claude/skills/holo-card-studio），不在本工程内；
本脚本只做「素材编排 + 调用 + 产物收集」，重活由 skill 完成。

用法：
  python -m card_studio.run_card_studio --card MegaDiancie
  python -m card_studio.run_card_studio --card MegaDiancie --stage-only   # 只暂存不运行
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from dataset_export.layout import LAYER_NAMES, CardLayout

DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "holo-card-studio"


def _stage_project(layout: CardLayout, mode: str) -> Path:
    """把 source/ 素材编排成 skill 工程目录 studio_project/。"""
    stage = layout.root / "studio_project"
    stage_assets = stage / "assets"
    stage_assets.mkdir(parents=True, exist_ok=True)

    # 分层 PNG（lenticular 路线为 image_a/image_b/text，按需扩展）
    for name in LAYER_NAMES:
        src = layout.layer_png(name)
        if src.exists():
            shutil.copy2(src, stage_assets / f"{name}.png")

    # card-config.json（确保含 mode 字段）
    cfg_src = layout.config_path
    if cfg_src.exists():
        import json
        cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
        cfg.setdefault("mode", mode)
        (stage / "card-config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return stage


def _collect(layout: CardLayout, stage: Path) -> None:
    """把 skill 产物收集回 assets/<card>/{html,outputs}。"""
    web = stage / "web"
    if web.exists():
        if layout.html.exists():
            shutil.rmtree(layout.html)
        shutil.copytree(web, layout.html)
        print(f"[收集] web 查看器 → {layout.html}")
    else:
        print(f"[警告] 未找到 {web}，skill 可能未生成 web 查看器")

    # 渲染预览（skill 通常输出 renders/ 或 previews/）
    for sub in ("renders", "previews"):
        d = stage / sub
        if d.exists():
            dst = layout.outputs / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(d, dst)
            print(f"[收集] {sub} → {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="MegaDiancie")
    ap.add_argument("--root", default="assets")
    ap.add_argument("--mode", default="holographic",
                    choices=["holographic", "lenticular"])
    ap.add_argument("--skill-dir", default=str(DEFAULT_SKILL_DIR))
    ap.add_argument("--python", default=sys.executable,
                    help="运行 skill 管线的解释器（需已装 Blender 等 skill 依赖）")
    ap.add_argument("--skip-npm", action="store_true")
    ap.add_argument("--stage-only", action="store_true",
                    help="只暂存 skill 工程目录，不运行管线（便于手动 / 用 Skill 工具跑）")
    args = ap.parse_args()

    layout = CardLayout(args.card, args.root).ensure()
    skill_dir = Path(args.skill_dir)
    pipeline = skill_dir / "scripts" / "run_pipeline.py"

    # 素材校验
    missing = [n for n in LAYER_NAMES if not layout.layer_png(n).exists()]
    if not layout.config_path.exists():
        missing.append("card-config.json")
    if missing:
        print(f"[警告] assets/{args.card}/source 缺少：{missing}")
        print("       请先用 holo-card-studio 的图像生成产出四层 PNG，并")
        print("       复制 card_studio/card-config.template.json 为 source/card-config.json。")
        if not args.stage_only:
            sys.exit(1)

    stage = _stage_project(layout, args.mode)
    print(f"[暂存] skill 工程目录 → {stage}")

    if args.stage_only:
        print("[stage-only] 已暂存，未运行管线。可手动执行：")
        print(f'  {args.python} "{pipeline}" --project "{stage}" --mode {args.mode}')
        return

    if not pipeline.exists():
        sys.exit(f"找不到 skill 管线：{pipeline}\n请用 --skill-dir 指定（默认 {DEFAULT_SKILL_DIR}）")

    cmd = [args.python, str(pipeline), "--project", str(stage), "--mode", args.mode]
    if args.skip_npm:
        cmd.append("--skip-npm")
    print(f"[运行] {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        sys.exit(f"skill 管线失败（exit={ret.returncode}）")

    _collect(layout, stage)
    print(f"\n完成：演示 HTML 位于 {layout.html}/index.html（用浏览器打开）")


if __name__ == "__main__":
    main()
