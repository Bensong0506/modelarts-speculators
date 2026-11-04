# 同事 AI 接手入口

## 接手目的与现状

本分支将 dense ALLaVA continuation 和 D-MTP token/fusion continuation 接入 Speculators 包，并提供固定私有 vLLM runtime 的推理工具。先读本目录 [README](README.md)、[训练契约](TRAINING.md)、[推理契约](VLLM_INFERENCE.md) 和 [实际验证记录](VALIDATION.md)。历史结果仅是快照，不能当成新分支已在 GPU 上重新训练的结果。

这里没有数据、模型权重、SSH 配置、密钥或运行中的训练任务。实际目录、GPU 分配及 job 生命周期由操作者在自己的环境提供。先核对当前状态，再执行其本轮明确授权的工作；不要自动启动旧任务、停止别人的进程或改共享环境。

## 代码地图

| 位置                                             | 管理内容                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `src/speculators/cli/train_cached_dflash2.py`    | `speculators train-cached-dflash2` 入口                      |
| `src/speculators/train/cached_dflash2/config.py` | 严格 YAML/CLI 配方；未知字段报错                             |
| `model.py`, `attention.py`, `sdpa.py`            | checkpoint 兼容 dense backbone / anchor 分块 attention       |
| `dmtp.py`                                        | native T1 + 后六位，query/context 时序分离及 residual fusion |
| `data.py`                                        | cache 形状/身份/sidecar 合同、anchor 与监督下标              |
| `loss.py`                                        | 冻结 head 的 full-vocab KL/CE 和 bounded-memory backward     |
| `trainer.py`, `common.py`, `checkpoint.py`       | 训练、验证、batch plan、指标、导出及 epoch 续训              |
| `examples/cached_dflash2/*.yaml`                 | dense 四组 + token/fusion 两组                               |
| `integrations/vllm_dmtp/overlay`                 | 原始 runtime 文件，逐文件校验；不批量格式化                  |
| `scripts/vllm_dmtp_runtime.py`                   | 固定 base 的安装/检查工具                                    |
| `scripts/serve_cached_dflash2.py`                | 安全默认监听地址的服务入口及 provenance                      |
| `scripts/evaluate_cached_dflash2.py`             | 同口径在线 AR/AL、生成结果与重复检查                         |
| `scripts/compare_cached_dflash2_outputs.py`      | 固定样本的完整输出比较                                       |
| `scripts/plot_cached_dflash2.py`                 | 实际日志曲线，CE/KL 分面                                     |

## 管理顺序

1. 查看 `git status`、当前 branch/commit 与本地修改，不覆盖别人的工作。阅读根目录 `AGENTS.md`。
2. 读取操作者提供的 target/init/shared/cache provenance，确认 FP8/BF16、target 层、图像 split 和 tokenizer/template 一致。D-MTP 必须有真实 native sidecars。
3. 设置 YAML 中四个路径环境变量；`--check-data` 扫描所有 cache。不要移动或复制密钥到工作目录，不把原始图像/回答、权重或个人路径提交进仓库。
4. 在已分配 GPU 上先单独 pilot，再正式训练。新配方用新 output；恢复只用同配方 `run.yaml --resume`。所有组使用同一初始 checkpoint、数据和更新预算。
5. 核对实际 PID/命令、退出码和日志；`progress.json` 只是快照。`summary.completed=true` 才表示训练循环结束，仍不代表在线验证通过。
6. 缓存选择最佳 epoch 后，检查 vLLM HEAD/overlay/import 路径，做小图加载、真实输出比较、容量检查，再做完整 held-out 在线验证。不要跳过输出异常。
7. 汇总时逐个注明样本集、数量、目标精度、K、并发、KV、graph、训练 seed、checkpoint epoch、GPU 身份、重复数；区分缓存、16图、容量和完整结果。

## 不可静默改变的实验约定

- 原生 MTP、共享 embedding/head、selector 冻结；不能为了“更容易提升”解冻后仍称同一对照。
- T1 训练用 gold transcript，验证/推理用 native candidate。这是 teacher forcing，不是 on-policy 训练。
- `a` 的 target context `<a`；MTP hidden 为消费 A 后、消费 T1 前的单流 hidden。禁止未来信息泄漏。
- 同前缀 teacher argmax CE 与 transcript CE 不同；prefix AR 与独立位置正确率不同。
- AL 含一个 bonus；缓存 `online_validated=false` 不能手工改 true。导出 `vllm_runtime_verified=false` 也不要因“能加载”就自动改成正确性通过。
- 原生 MTP T1 不一定高于 DFlash2。收益需看完整连续前缀及运行开销，不把“用了 MTP”当作理论保证。

## 错误处理

- cache 对齐、hash、prompt boundary 或 split 检查失败：修复数据来源/路径；不要关闭检查绕过。
- `Resume provenance changed`：先比较具体字段。确需改配方就新建实验，不编辑旧 provenance 假装相同。
- OOM：先核对实时空闲显存、模型常驻内存、KV 与 allocator 上限。别仅凭 GB300 总显存就增加全部参数；改容量配置需为所有对照重测。
- runtime base/hash/import 路径失败：在正确私有 checkout 重建，不对任意 vLLM 强行应用。
- 输出比较非零：保留逐条结果与首差位置，先做相同前缀诊断；不能以 AR 更高覆盖这个失败。

## 验证和提交

支持的 Linux 开发环境运行 `python -m pytest tests/unit/train/cached_dflash2` 和仓库 quality checks。Intel Mac 可运行 `python scripts/test_cached_dflash2_cpu.py`，但它隔离掉无关 Transformers 模型导入；不能宣称检查了完整安装包或 CUDA。

提交前检查源码、测试、配方、文档和生成物链接，按根目录规则签署 DCO。不要重新上传个人运行目录。修改 overlay 时应作为新 runtime 版本，记录 base/diff/hash，并重新做 GPU 验证；不能只更新 SHA 来消除失败。

目前仍待完成：受支持环境中的完整 CLI/CUDA 迁移回归、D-MTP 跨卡输出差异定位、BF16 target 对照、独立训练 seed 和更完整状态等价验证。它们没有在本次代码交接中被自动提交为作业。
