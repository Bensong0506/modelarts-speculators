# DSpine：Qwen3.8-Flash-Next 离线特性对照

分支 `codex/dspine-qwen38-offline` 从 `ff412a0` dense 业务分支独立创建。用户选择在既有草稿和 ALLaVA 20k 五层缓存上移植特性。本分支训练草稿，目标 Qwen 权重、图片/回答和共享 head 保持冻结。

DSpine 特性包括每层 gated adjacent injection、目标输出 embedding 的 PCA/白化 token codes、逐层 cosine 对齐、CE+L1 分布训练、gold predecessor 条件的候选 refinement，以及验证时的相邻候选 transition cache。验证阶段不读取 gold successor；前缀上下文严格 `<a`。

本移植保留原 DFlash2 卷积、五层 target cache 和七候选，将草稿内部 attention 改为 causal。新增写回投影以零初始化，transfer rank=1024、message width=512；对齐 beta=0.5 并随 LR warmup 上升，前三层 gold replacement 概率从 0.5 在训练前 1/6 后逐渐下降，到 1/3 时归零。共享 embedding/head 和原 selector 冻结；特性路径用自己的条件解码，原 selector 不参与。

官方 SpecForge 2dadea10 的最终 loss 按序列数缩放。本移植沿用现有训练器的 weighted-token mean，并分别按 covered/eligible 权重归一 refinement CE/L1；alignment 在草稿层间取均值，再乘 detached backbone CE 与 beta。论文成绩不可直接移用。

`feature-dspine-causal-control.yaml` 保留相同 causal backbone 和 CE+L1 目标，但关闭注入、refinement、alignment，用 unary 候选。它用于区分新的因果 mask/基础目标与 DSpine 模块的贡献。默认未开特性时仍为原 dense KL。

## 输入与公平比较

沿用 [缓存契约](TRAINING.md) 和 [数据检查](BUSINESS_DATA.md)。所有组使用同一初始 dense checkpoint、shared.pt、train/val 划分和 batch/anchor 计划。epoch 0 是本轮起点；初始权重已有训练历史，不称从零训练。真实输入、模型权重和个人服务器路径不进入 Git。

配方为五轮、lr=3e-5、seed280、batch上限32、context budget48000、每图64训练anchors、8验证anchors。正式步数由实际 manifest 决定，历史 17998 个有效train对应2815步。验证报告同时记录 teacher/transcript 的 unary 与 selected 连续前缀，不用逐位置准确率冒充接受率。AR 分母为7×blocks，AL包含1个bonus。均是 offline cached-prefix proxy，不代表在线生成接受率、吞吐或输出等价。

## 执行

```bash
export DFLASH2_INITIAL=/personal/models/initial
export DFLASH2_SHARED=/personal/cache/shared.pt
export ALLAVA_CACHE_MANIFEST=/personal/cache/manifest.json
export EXPERIMENT_ROOT=/personal/runs/dspine

speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dspine.yaml --check-data
speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dspine.yaml \
  --output "$EXPERIMENT_ROOT/pilot" --pilot-steps 3 --pilot-val-images 8
speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dspine.yaml
# 仅在同源/同配方的最后完整epoch上恢复：
speculators train-cached-dflash2 --config "$EXPERIMENT_ROOT/dspine/run.yaml" --resume

python scripts/evaluate_cached_dspine.py \
  --run "$EXPERIMENT_ROOT/dspine" \
  --checkpoint "$EXPERIMENT_ROOT/dspine/epochs/epoch-05" \
  --output "$EXPERIMENT_ROOT/reloaded-epoch05.json"
```

GPU/显存先按实时分配检查；本配方的160GiB allocator上限可在新实验中调整并记录。CPU仅允许显式 synthetic smoke。两分支各用独立环境或源码路径，不向同一 output 同时写入。

## 逐图计数与配对复算

`evaluate_cached_prefix_counts.py` 使用相同的固定验证 anchor，并保存每张图片的连续前缀计数和评估 provenance。`compare_cached_prefix_counts.py` 对同图的差值做图片级 bootstrap；不同训练 seed 的方差不包含在该区间中。

```bash
python scripts/evaluate_cached_prefix_counts.py --kind dspine \
  --run "$EXPERIMENT_ROOT/dspine" \
  --checkpoint "$EXPERIMENT_ROOT/dspine/epochs/epoch-05" \
  --output "$EXPERIMENT_ROOT/epoch05-counts.json"
python scripts/compare_cached_prefix_counts.py \
  --input dense /personal/results/dense-counts.json \
  --input dspine "$EXPERIMENT_ROOT/epoch05-counts.json" \
  --reference dense --output "$EXPERIMENT_ROOT/paired-comparison.json"
```

缓存只搬迁目录时可传 `--manifest`；验证集身份、边界及缓存摘要必须与原 manifest 一致。此入口不会重新训练。

## 导出与验证边界

导出使用独立 architecture 标识，支持本分支 `load_export` 和上述离线评估脚本，且 `serving_verified=false`。继承的 vLLM wrapper 会拒绝这个新结构；本轮不实现 serving。重新加载校验不能当作在线推理通过。纯 dense、未开启特性的旧流程继续适用旧文档。

使用 `python scripts/test_cached_dflash2_cpu.py` 检查组件、FP32/BF16 loss梯度、未来信息边界、特性训练、断点恢复和导出。完整包导入和真实GPU小步训练另外记录。`--check-data` 校验缓存的形状、可选SHA、有限值和split；仍需核对目标权重/精度、tokenizer及采集 provenance。

实现来源与许可见 [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md)。真实实验结果完成后另附，未完成前不宣称提升。
