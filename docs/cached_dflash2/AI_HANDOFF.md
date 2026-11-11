> 当前分支新增 [DCORE 离线特性对照](FEATURE_DCORE.md)。下文原 dense 约定适用于关闭特性的路径；新结构、loss 和导出边界以特性指南为准。

# 同事 AI 接手入口：dense 业务增训

本轮离线结果与复现计数见 [DCORE 结果](results/qwen38_dcore_20261111/README.md)。

本分支：`codex/dflash2-dense-business-training`。当前任务是用内网业务数据继续训练普通 dense DFlash2，首位及后六位全部由 DFlash2 提出。训练入口为 `speculators train-cached-dflash2`，只接受 `arm: dense`；不需要原生 MTP sidecar。

D-MTP 的 token/fusion 实现与实验继续保留在 [codex/dflash2-allava-dmtp-training](https://github.com/Bensong0506/modelarts-speculators/tree/codex/dflash2-allava-dmtp-training)，该分支没有被本次拆分修改。不要在 dense 业务分支加入 D-MTP 配方；需要继续 D-MTP 时使用它自己的分支和工作目录。

## 按顺序执行

1. 读取根目录 `AGENTS.md`，确认分支、commit 和本地修改，完整阅读 [业务数据格式和训练步骤](BUSINESS_DATA.md)。实际数据、GPU、目录和任务授权由内网操作者提供。
2. 核对兼容的初始 dense checkpoint、匹配 target 的 `shared.pt`、manifest 和 target `.pt` 缓存。图片/prompt/answer 不等于 hidden 缓存，其他 Speculators 格式不能直接假定兼容。
3. 按业务实体划分 train/val/test，审核短于 16 tokens 的回答、多轮 assistant 边界和媒体身份。核对 target 权重、精度、tokenizer/template、视觉处理和特征层来源。
4. 配置业务 YAML，执行 `--check-data`。它不加载全部模型权重，也不证明采集语义和 GPU 环境正确。通过后在已授权 GPU 做独立 3 步 pilot，再从相同初始化开正式五轮训练。
5. 监控真实 PID、退出码、日志、每轮验证及导出；核对 `summary.completed`、实际步数和冻结 selector。日志和结果保留内网，恢复只用相同分支/代码/数据/配方的 `run.yaml --resume`。
6. 按 [dense vLLM 推理](VLLM_INFERENCE.md) 对比业务 epoch 0 与各 checkpoint，保持目标、数据与运行条件一致。缓存指标和在线指标分别报告；输出差异保留诊断证据，不宣称未经证明的无损收益。

## 代码地图

| 位置                                                                    | 用途                                       |
| ----------------------------------------------------------------------- | ------------------------------------------ |
| `src/speculators/train/cached_dflash2/config.py`                        | dense 配置与拒绝混合配方                   |
| `data.py`                                                               | target 缓存检查、batch 和七位监督下标      |
| `model.py`, `attention.py`, `sdpa.py`                                   | 保持原 dense checkpoint 计算约定           |
| `loss.py`                                                               | 冻结 LM head 的 full-vocab KL/CE           |
| `trainer.py`, `checkpoint.py`, `common.py`                              | 训练、缓存验证、BF16 导出和完整 epoch 恢复 |
| `examples/cached_dflash2/dense-*.yaml`                                  | dense 四种历史配方                         |
| `scripts/vllm_dflash2_runtime.py`                                       | 固定 private runtime 安装/摘要检查         |
| `scripts/serve_cached_dflash2.py`, `scripts/evaluate_cached_dflash2.py` | 仅 target/dense 的在线入口                 |
| `scripts/compare_cached_dflash2_outputs.py`                             | 逐请求输出比较                             |

## 分支与运行边界

- 训练数据、权重、凭据、SSH 配置、客户样本及个人运行路径不提交进公开 GitHub。
- 本次从已训练的 dense 路径拆分，历史五轮 GPU 证据见 [验证记录](VALIDATION.md)；拆分不等于已在内网重新跑通。CPU 组件测试也不等于完整 CLI/CUDA 验收。
- `integrations/vllm_dflash2` 的 22 个文件仍是原固定 runtime 的原始字节。内部保留未启用的 MTP/vendor 代码是为维持已记录的依赖版本；业务训练器已移除 D-MTP 实现，推理 wrapper 只允许 `target`/`dense`。不要擅自裁剪 vendor 或换成普通 pip vLLM。
- 此分支移除了 `fusion_init` 配置项。旧分支的 `run.yaml`/optimizer provenance 不可原样跨分支恢复；旧作业回原分支恢复，新业务任务可用兼容的 dense 导出权重在新目录开始。
- 继续保持 FP32 master/Adam、BF16 autocast/export、冻结 embedding/head/selector、target context 严格 `<a`、七候选和 AL 含 bonus 的约定。更换目标精度或 loss 属于新实验配方。

## 验证与错误处理

在受支持的 Linux 环境运行 `python -m pytest tests/unit/train/cached_dflash2` 和仓库 quality checks。Intel Mac 可用 `python scripts/test_cached_dflash2_cpu.py` 检查独立组件；它不验证完整安装包或 GPU。

缓存/来源不匹配时修复来源和转换；不要关闭校验。OOM 时调整私人配置并重新 pilot，不覆盖共享环境或停止别人的作业。输出比较非零时报告失败和首差位置；运行完成不代表输出等价通过。提交使用 DCO 签署，保持数据和密钥在内网。
