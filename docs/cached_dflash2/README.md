> 当前分支新增 [DSpine 离线特性对照](FEATURE_DSPINE.md)。下文原 dense 约定适用于关闭特性的路径；新结构、loss 和导出边界以特性指南为准。

# Dense DFlash2：ALLaVA 与内网业务增训

分支 `codex/dflash2-dense-business-training` 专用于普通 dense DFlash2 continuation。`speculators train-cached-dflash2` 读取 target 特征缓存，训练 backbone 预测七位草稿 token；target Qwen、共享 embedding/head 和 selector 保持冻结。它不对 Qwen 本体做业务 SFT。

- [同事 AI 交接入口](AI_HANDOFF.md)
- [业务数据格式、检查、试跑与正式训练](BUSINESS_DATA.md)
- [训练契约](TRAINING.md)
- [Dense vLLM 推理与在线验证](VLLM_INFERENCE.md)
- [验证记录与已知限制](VALIDATION.md)
- [Dense 配方](../../examples/cached_dflash2)

D-MTP 后续研发使用单独的 [codex/dflash2-allava-dmtp-training 分支](https://github.com/Bensong0506/modelarts-speculators/tree/codex/dflash2-allava-dmtp-training)。本分支的训练器拒绝 `token`/`fusion` 配方；在线 wrapper 只接受 `dense`/`target`。

## Dense 结构与产物

输入为 pending token A 加七个 mask，并使用严格 `<a` 的 target 特征，预测 T1…T7。初始化须是兼容的 dense DFlash2；导出为 BF16 `DFlash2DraftModel`，不含共享 embedding/head。目标权重的 FP8/BF16 条件独立记录，不能从草稿 dtype 推断目标精度。

最新已交付的 ALLaVA epoch 5 是 dense 权重。可以作为匹配 target 的业务 continuation 起点，但必须记录已有训练历史，并用本轮 epoch 0 作为比较基线。当前分支并未额外启动 GPU 训练。

## 历史 dense loss

下面是 2026-11-02 的旧 dense 配方曲线，不是当前内网业务数据结果。CE 与 KL 不能按绝对 loss 跨目标比较。

![Dense ALLaVA historical losses](figures/dense_loss_curves.png)

新运行可用 `scripts/plot_cached_dflash2.py` 从实际 `train.jsonl` 生成 PNG/PDF。在线接受率和输出一致性需另测，缓存 AL 不直接证明在线收益。
