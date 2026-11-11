# DCORE：Qwen3.8-Flash-Next 离线特性对照

分支 `codex/dcore-qwen38-offline` 从 `ff412a0` dense 业务分支独立创建。用户选择在既有草稿和 ALLaVA 20k 五层缓存上移植特性。本分支训练草稿，目标 Qwen 权重、图片/回答和共享 head 保持冻结。

DCore 特性包括 transcript CE 上的 detached D-PACE 后缀置信度权重（alpha=0.5），以及逐草稿层的目标特征混合和独立 context K/V。为保留起点，目标特征仍经过预训练 fc：对各目标层贡献施加 softmax 权重，乘以层数；均匀初值等于原 fc 输入，context K/V 从原 K/V 复制。原卷积、滑窗、selector 保留，selector 冻结。

这不是原论文的 DFlare 九层特征架构。本实验固定五层 [5,15,25,35,45]、原预训练投影和 ALLaVA 数据；D-PACE 使用蒸馏 transcript 标签，原 dense CE 则使用 teacher argmax。不同 loss 的绝对值不能横向排名。

消融可以使用 `--no-dcore-context --objective dpace`（仅目标）或 `--dcore-context --objective kl`（仅 context 特性）。默认未开特性时仍为原 dense KL。

## 输入与公平比较

沿用 [缓存契约](TRAINING.md) 和 [数据检查](BUSINESS_DATA.md)。所有组使用同一初始 dense checkpoint、shared.pt、train/val 划分和 batch/anchor 计划。epoch 0 是本轮起点；初始权重已有训练历史，不称从零训练。真实输入、模型权重和个人服务器路径不进入 Git。

配方为五轮、lr=3e-5、seed280、batch上限32、context budget48000、每图64训练anchors、8验证anchors。正式步数由实际 manifest 决定，历史 17998 个有效train对应2815步。验证报告同时记录 teacher/transcript 的 unary 与 selected 连续前缀，不用逐位置准确率冒充接受率。AR 分母为7×blocks，AL包含1个bonus。均是 offline cached-prefix proxy，不代表在线生成接受率、吞吐或输出等价。

## 执行

```bash
export DFLASH2_INITIAL=/personal/models/initial
export DFLASH2_SHARED=/personal/cache/shared.pt
export ALLAVA_CACHE_MANIFEST=/personal/cache/manifest.json
export EXPERIMENT_ROOT=/personal/runs/dcore

speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dcore.yaml --check-data
speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dcore.yaml \
  --output "$EXPERIMENT_ROOT/pilot" --pilot-steps 3 --pilot-val-images 8
speculators train-cached-dflash2 --config examples/cached_dflash2/feature-dcore.yaml
# 仅在同源/同配方的最后完整epoch上恢复：
speculators train-cached-dflash2 --config "$EXPERIMENT_ROOT/dcore/run.yaml" --resume

python scripts/evaluate_cached_dcore.py \
  --run "$EXPERIMENT_ROOT/dcore" \
  --checkpoint "$EXPERIMENT_ROOT/dcore/epochs/epoch-05" \
  --output "$EXPERIMENT_ROOT/reloaded-epoch05.json"
```

GPU/显存先按实时分配检查；本配方的160GiB allocator上限可在新实验中调整并记录。CPU仅允许显式 synthetic smoke。两分支各用独立环境或源码路径，不向同一 output 同时写入。

## 逐图计数与配对复算

`evaluate_cached_prefix_counts.py` 使用相同的固定验证 anchor，并保存每张图片的连续前缀计数和评估 provenance。`compare_cached_prefix_counts.py` 对同图的差值做图片级 bootstrap；不同训练 seed 的方差不包含在该区间中。

```bash
python scripts/evaluate_cached_prefix_counts.py --kind dcore \
  --run "$EXPERIMENT_ROOT/dcore" \
  --checkpoint "$EXPERIMENT_ROOT/dcore/epochs/epoch-05" \
  --output "$EXPERIMENT_ROOT/epoch05-counts.json"
python scripts/compare_cached_prefix_counts.py \
  --input dense /personal/results/dense-counts.json \
  --input dcore "$EXPERIMENT_ROOT/epoch05-counts.json" \
  --reference dense --output "$EXPERIMENT_ROOT/paired-comparison.json"
```

缓存只搬迁目录时可传 `--manifest`；验证集身份、边界及缓存摘要必须与原 manifest 一致。此入口不会重新训练。

## 导出与验证边界

导出使用独立 architecture 标识，支持本分支 `load_export` 和上述离线评估脚本，且 `serving_verified=false`。继承的 vLLM wrapper 会拒绝这个新结构；本轮不实现 serving。重新加载校验不能当作在线推理通过。纯 dense、未开启特性的旧流程继续适用旧文档。

使用 `python scripts/test_cached_dflash2_cpu.py` 检查组件、FP32/BF16 loss梯度、未来信息边界、特性训练、断点恢复和导出。完整包导入和真实GPU小步训练另外记录。`--check-data` 校验缓存的形状、可选SHA、有限值和split；仍需核对目标权重/精度、tokenizer及采集 provenance。

实现来源与许可见 [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md)。真实实验结果完成后另附，未完成前不宣称提升。
