# NeuroFoil

> 宝可梦光栅卡 / 闪卡 · 神经渲染 技术穿刺工程
> 最终落地：OpenHarmony + 自研渲染引擎　|　当前阶段：PC 穿刺

用一个小型网络拟合「视角 → 卡片外观」的映射（神经纹理 / 2D 神经辐射场范式），
在手机上以低功耗实时复现光栅卡的镭射反光、多层视差、星光等**组合效果**。

## 需求与目标

- 需求说明书：[`需求说明书.md`](需求说明书.md)
- 参考效果 / 项目：见需求说明书 §2

## 管线与目录结构

四个代码文件夹按数据流串联，所有卡片资产集中在 `assets/<CardName>/` 下，
代码只认卡片名、不写死路径（由 `dataset_export.layout.CardLayout` 统一解析）：

```
card_studio  →  dataset_export  →  training  →  compare
(skill 演示HTML)  (导出训练集)      (训练模型)   (效果对比)
        └──────────────┴──────────────┴─────────────┘
                    都读写 assets/<card>/
```

```
NeuroFoil/
├── 需求说明书.md                 # 穿刺需求（目标效果、架构、指标、验收标准）
├── README.md                     # 本文件：工程说明 + 开发原则
├── requirements.txt              # Python 依赖（torch / numpy / Pillow）
├── docs/
│   └── archive/
│       ├── CHANGELOG.md          # 开发者日志（按版本号记录）
│       └── version/              # 各版本变更文档（点号 → 横线）
│           ├── _TEMPLATE.md
│           ├── v0-0-1.md
│           ├── v0-0-2.md
│           └── v0-0-3.md
├── card_studio/                  # ① 用 holo-card-studio skill 产出可用演示 HTML
│   ├── run_card_studio.py        # 编排素材 → 调 skill 管线 → 收集 web 查看器到 assets/<card>/html
│   ├── card-config.template.json # 卡片配置模板（深度 / foil / particles）
│   └── README.md
├── dataset_export/               # ② 从演示结果导出训练集
│   ├── layout.py                 # assets/<card>/ 路径解析（CardLayout）
│   ├── angles.py                 # 视角采样 + npz 存取
│   ├── layer_render.py           # 真实分层 GT 渲染（skill 的四层 PNG + 深度 + 视差）
│   ├── synthetic_gt.py           # 合成占位卡渲染（无资产时自检 / CI）
│   ├── api.py                    # 渲染门面：auto|skill|synthetic 调度
│   └── export.py                 # CLI：--card → assets/<card>/dataset/{train,val}.npz
├── training/                     # ③ 训练相关
│   ├── model.py                  # 8 入参神经纹理 MLP（冻结架构）
│   ├── pixel_dataset.py          # 逐像素训练数据集
│   ├── metrics.py                # PSNR / SSIM
│   ├── latency.py                # FLOPs / CPU 墙钟 / 移动端折算
│   ├── train.py                  # 训练 + 评估（--card，读写 assets/<card>/）
│   └── sweep.py                  # 超参扫描（size↔quality↔latency 曲线）
├── compare/                      # ④ 效果对比演示
│   ├── preview.py                # GT 多视角拼图目检
│   └── gt_vs_model.py            # GT vs 模型重建并排对比图
└── assets/                       # 卡片资产（详见 assets/README.md）
    └── MegaDiancie/
        ├── source/               # 原始图片：四层 PNG + card-config.json（入库）
        ├── studio_project/       # skill 工程暂存（card.blend / renders / web，不入库）
        ├── html/                 # 演示 HTML（skill web 查看器，不入库）
        ├── dataset/              # 训练集 npz（不入库）
        └── outputs/              # 产物 model.pt / metrics / 对比图 / 曲线（不入库）
```

### 常用命令（在仓库根目录执行）

```bash
# ① 产出演示 HTML（需先在 assets/<card>/source 备好四层 PNG + card-config.json）
python -m card_studio.run_card_studio --card MegaDiancie

# ② 导出训练集（auto：有 skill 素材用真实卡，否则回退合成占位卡）
python -m dataset_export.export --card MegaDiancie --res 256
python -m dataset_export.export --card MegaDiancie --source synthetic   # 无资产自检

# ③ 训练
python -m training.train --card MegaDiancie --feat-res 64 --width 32 --steps 2000 --tag demo

# ④ 效果对比（source 需与训练集一致）
python -m compare.gt_vs_model --card MegaDiancie --tag demo --source synthetic
```

## 开发原则：每次改动可回溯

**方案总纲**：版本号 + git commit 信息 + 开发者日志 + 变更文档

### 回溯点（每个版本的变更文档必须记录）

1. 改动原因
2. 改动方案
3. 最终改动方案
4. 改动结果
5. 遗留事项
6. 风险点

### 回溯产物

| 产物 | 规则 |
|---|---|
| a. 版本号 | `v0.0.1`，区分大版本（重大/不兼容）· 中版本（功能新增）· bugfix（修复） |
| b. 变更文档 | 每个版本一份，内容 = 上述 6 个回溯点 |
| c. 开发者日志 | `docs/archive/CHANGELOG.md`，按版本号记录：改动时间点、回溯点文件路径、git commit 信息 |
| d. git commit | 每更新一个版本 commit 一次，并总结该版本对应的 commit 位置；**AI 只提供 commit 信息，由人工提交代码** |

### 产物存储位置

- 变更文档：`docs/archive/version/v0-0-1.md`（版本号点号换成横线）
- 开发者日志：`docs/archive/CHANGELOG.md`

## 版本与状态

- 当前版本：`v0.0.3`（项目结构重组：四段管线 + assets 资产目录）
- 开发者日志：[`docs/archive/CHANGELOG.md`](docs/archive/CHANGELOG.md)