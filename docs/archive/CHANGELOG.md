# 开发者日志（CHANGELOG）

> 按版本号记录：改动时间点、回溯点文件路径、git commit 信息。
> 约定：AI 只提供 commit 信息，代码由人工提交。

---

## v0.0.1 — 2026-11-25

- **变更文档**：`docs/archive/version/v0-0-1.md`
- **改动主题**：初始化工程 + 定稿需求说明书 + 建立可回溯归档体系
- **改动结果**：需求说明书定稿（8 入参神经纹理架构 / 512×512@60fps / 视角 ±45° / 验收标准与待定项）；归档体系就位，开发原则写入 README。
- **提交信息**（人工提交）：

  ```
  init: 初始化 NeuroFoil 工程与需求说明书（v0.0.1）

  - 建立可回溯归档体系：docs/archive/CHANGELOG.md、version/v0-0-1.md、_TEMPLATE.md
  - 定稿需求说明书：8入参神经纹理架构、512×512@60fps、视角±45°、验收标准与待定项
  - README 写入开发原则（版本号 + 变更文档 + 开发者日志 + git 提交）

  Co-Authored-By: Claude Code <noreply@anthropic.com>
  ```

---

## v0.0.2 — 2026-11-25

- **变更文档**：`docs/archive/version/v0-0-2.md`
- **改动主题**：PC 穿刺端到端链路打通（合成 GT → 训练 → 评估 → 延迟）
- **改动结果**：新增 `spike/` 包（合成 GT 渲染、神经纹理 MLP、训练 / 评估 / 延迟 / 超参扫描 / 预览脚本）与 `requirements.txt`；链路全程 CPU 可复现。Smoke test（feat=64,width=32,2000 步）：PSNR 19.15 dB / SSIM 0.606；模型约 71 KB；512×512 前向 CPU 33.68 ms（不达 60fps），移动端 GPU 400 GFLOPs 折算 1.85 ms（可达成）。
- **提交信息**（人工提交）：

  ```
  feat: PC 穿刺端到端链路（合成 GT → 训练 → 评估 → 延迟）

  - 新增 spike 包：gt_holo（合成四层镭射卡 GT）、model（8入参神经纹理 MLP）、
    train/eval/metrics/latency/make_dataset/sweep/preview
  - 新增 requirements.txt（torch/numpy/Pillow）
  - 链路 CPU 跑通：feat=64 width=32 训练2000步 PSNR 19.15dB SSIM 0.606
  - 延迟初测：512×512 CPU 33.68ms；移动端GPU(400GF) 折算 1.85ms
  - 变更文档 docs/archive/version/v0-0-2.md + CHANGELOG

  Co-Authored-By: Claude Code <noreply@anthropic.com>
  ```

---

## v0.0.3 — 2026-11-26

- **变更文档**：`docs/archive/version/v0-0-3.md`
- **改动主题**：项目结构重组（四段管线 card_studio → dataset_export → training → compare + assets 资产目录）
- **改动结果**：原平铺的 `spike/` 按数据流拆为四个独立包，新增 `assets/<card>/{source,studio_project,html,dataset,outputs}` 资产目录，代码经 `CardLayout` 按卡片名解析路径、不写死路径；`dataset_export` 新增真实分层 GT 渲染（`layer_render`，读 skill 四层 PNG + 深度 + 视差）与 `auto|skill|synthetic` 调度门面（`api`）；`card_studio` 编排 skill 产出演示 HTML；`compare/gt_vs_model` 由临时脚本升级。纯结构调整，未改模型架构与训练逻辑。跨包依赖单向无环。
- **提交信息**（人工提交）：

  ```
  refactor: 项目结构重组为四段管线 + assets 资产目录（v0.0.3）

  - 按数据流拆分代码：card_studio（skill 演示HTML）→ dataset_export（导出训练集）
    → training（训练/评估）→ compare（效果对比），跨包依赖单向无环
  - 新增 assets/<card>/{source,studio_project,html,dataset,outputs} 资产目录，
    代码经 dataset_export.layout.CardLayout 按卡片名解析路径，不写死路径
  - dataset_export 新增 layer_render（skill 四层 PNG 真实视差 GT）+ api 门面
    （auto|skill|synthetic 调度）；synthetic_gt 由原 spike/gt_holo 迁移
  - 原 spike/ 代码迁移到 training/（model/metrics/latency/train/sweep）与
    compare/（preview/gt_vs_model）；train/sweep 改为 --card 驱动；删除 spike/ 与 _show.py
  - 更新 .gitignore（*.npz、assets/*/ 生成物、torchinductor_*）、README、assets/README
  - 变更文档 docs/archive/version/v0-0-3.md + CHANGELOG

  Co-Authored-By: Claude Code <noreply@anthropic.com>
  ```

---