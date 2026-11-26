# card_studio —— 用 skill 产出可用演示 HTML

本文件夹是 NeuroFoil 数据管线的**源头**：通过用户级安装的
[`holo-card-studio`](~/.claude/skills/holo-card-studio) skill，把一张卡的分层美术
做成带视差 / 镭射 / 星光的 **web 查看器（演示 HTML）**，并落到
`assets/<card>/html/`。后续 `dataset_export` 再从同一批分层素材导出训练集。

## 为什么需要它

skill 的 holographic 路线产出四层透明 PNG（`background / subject / lineart / text`）
+ `card-config.json`（含 `subjectDepth / backgroundDepth / foil / particles` 等），
并用 Blender 把它们摆在**不同深度**上、导出 GLB、生成白色 web 查看器。
这正是需求说明书里「多层视差」的真实来源 —— 神经渲染要拟合的就是这个效果。

## 资产落点（统一在 assets/ 下）

```
assets/<card>/
├── source/          ← 你准备：四层 PNG + card-config.json（skill 的输入）
├── studio_project/  ← 本脚本暂存的 skill 工程（card.blend / renders / web，重产物，已 gitignore）
├── html/            ← 产出：可用的演示 HTML（web 查看器）
├── dataset/         ← dataset_export 产出：train.npz / val.npz
└── outputs/         ← training / compare 产物：model.pt、metrics、对比图、曲线
```

## 用法

1. **准备分层素材**（二选一）
   - 用 skill 的图像生成能力产出四层 PNG（推荐，风格统一）；或
   - 手动放四张对齐的透明 PNG 到 `assets/<card>/source/`。
2. **写配置**：复制 `card_studio/card-config.template.json` 为
   `assets/<card>/source/card-config.json`，改标题 / 深度 / foil 等。
3. **跑管线**：
   ```bash
   python -m card_studio.run_card_studio --card MegaDiancie
   ```
   脚本会暂存 skill 工程、调用 `run_pipeline.py`、把 web 查看器收集到 `assets/<card>/html/`。
   - 只想暂存、随后手动 / 用 Skill 工具跑：加 `--stage-only`。
   - skill 装在别处：`--skill-dir <path>`。
   - 跳过 web 依赖安装：`--skip-npm`。
4. **看演示**：浏览器打开 `assets/<card>/html/index.html`，鼠标 / 触摸倾斜即可看到
   视差 + 镭射 + 星光。

## 依赖

skill 管线需要 Blender 及其自身依赖（见 skill 文档）。本脚本只负责素材编排、
调用与产物收集，不重复实现 skill 逻辑。

## 没有真实素材时

`dataset_export` / `compare` 支持 `--source synthetic`，用纯合成占位卡跑通
「导出 → 训练 → 对比」全链路（无需 Blender / 素材），便于自检与 CI。
