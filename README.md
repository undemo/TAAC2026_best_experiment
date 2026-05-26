# TAAC2026

0.82617 方案开源分享

## Strongest Baseline

- 目录：`experiments/hyformer_pm_head_semantic_feature_v1`
- 线上分数：`0.826173`
- 主要改动：PM head、TimeToken/时间特征、语义 NS group、MissingAware 特征处理、user dense 分组编码。

运行入口：

```bash
cd experiments/hyformer_pm_head_semantic_feature_v1
bash run.sh
```

训练脚本依赖环境变量或命令行参数传入数据、checkpoint 和日志路径，详见 `train.py`。
