# assets —— 卡片资产目录

所有训练 / 推理 / 对比代码统一从这里读取资源，按卡片名分子目录：

```
assets/<CardName>/
├── source/          原始图片：skill 的四层透明 PNG（background/subject/lineart/text）
│                    + card-config.json（深度 / foil / particles 等参数）
├── studio_project/  card_studio 暂存的 skill 工程（card.blend / renders / web，重产物，不入库）
├── html/            演示 HTML：skill 产出的 web 查看器（不入库）
├── dataset/         训练集：dataset_export 产出的 train.npz / val.npz（不入库）
└── outputs/         产物：training / compare 的 model.pt、metrics.json、对比图、曲线 CSV（不入库）
```

- **入库**：`source/`（策划好的分层素材与配置）。
- **不入库**：`studio_project/ html/ dataset/ outputs/`（由代码运行时生成，见 `.gitignore`）。
- 目录由 `dataset_export.layout.CardLayout(...).ensure()` 在运行时自动创建，
  代码只认 `<CardName>`，不写死路径。

示例卡片：`assets/MegaDiancie/`。
