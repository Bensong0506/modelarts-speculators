# 迁移验证记录

本记录保留拆分前 dense 实跑证据；不代表 dense 独立分支在业务环境中又完成了一次 GPU 实验。运行数据、权重、连接信息和逐请求结果保留在操作者的私人实验记录中。

## 2026-11-10：dense 独立分支拆分检查

- 从 `847b8c89f5786793efe726fe1e79ef1aeb3d2f02` 建立 `codex/dflash2-dense-business-training`。原 `codex/dflash2-allava-dmtp-training` 保留，业务训练器与在线 wrapper 移除 token/fusion 入口，四份 dense 配方和数据交接文档保留。
- 21 项 CPU 组件检查通过，覆盖训练、恢复、导出、缓存与时序校验，以及混合配置/权重拒绝；17 个相关文件的定向 mypy 通过。全仓 Ruff、格式检查、文档格式和 provenance 检查通过；12 段文档 Bash 示例通过语法检查。
- 拆分前后使用同一合成数据、初始化、seed 和配置执行 CPU 两轮、四次更新。逐步 loss、梯度范数、学习率，以及初始和每轮缓存验证指标完全相同；epoch 0、1、2 各 36 个导出 tensor 逐元素完全相同。这仅证明该小型测试中 dense 计算保持一致，不代替真实业务数据或 GPU 回归。
- 更名后的 private vLLM overlay 22 个源文件与拆分前逐字节相同，原 provenance 摘要未改。本次没有新跑 GPU 训练或 vLLM 在线生成。
- 本机 `make quality` 在全仓 mypy 阶段失败：262 个文件中有 48 个文件、93 个错误，涉及缺少 `huggingface_hub`、`loguru`、`httpx` 等依赖和本机 PyTorch API 不兼容。不得将定向检查通过写成全仓 quality 通过。

## 2026-11-10：dense 路径实际训练与推理

- 在 ARM64 / GB300、PyTorch `2.14.0a0+b2c75dd062.nv26.09`、CUDA 13.4 的私有环境中，完整包入口完成了独立 3 步 pilot 和 64 图、1 epoch 的小训练。
- 完整 ALLaVA 缓存有 20,000 条，按图像分为 18,000 train / 1,000 val / 1,000 test；2 条训练响应过短被排除。正式 dense KL `3e-5` 训练完成 5 epoch、2815 更新，epoch 0–5 全部导出；loss/梯度有限，selector 校验未改变。
- 起始权重是此前 ALLaVA 实验选出的已增训 dense checkpoint；本轮衡量额外五轮 continuation，不是从未增训原始 DFlash2 开始。
- FP8 目标、C1/eager、16 图的 target-only / 起点 dense / 小训练 dense 三组均完成真实 vLLM 生成；C64 上限、KV32 GiB、graph 的 64 图容量检查通过。随后 epoch 0–5 各完成同一批 100 验证图、max1024 的在线评测，训练及完整评测控制器正常退出。逐请求计数、权重摘要与图像配对区间保存在个人实验记录中；没有把容量检查、缓存指标或历史大样本结果混入该对照。
- 各轮目标权重摘要一致，draft 摘要与对应 epoch 导出一致，100 张验证图与训练集无图像交集。选择结果只覆盖这一批验证图和一个训练 seed，未额外运行独立 test，也未通过输出等价验证。
- 起点 dense 的 16/16 条输出异于 target-only，不能称为无损验证通过。在两个选定差异样本上，三条路径共 1536 个实际输出 token 均符合各自当步 target argmax，且追踪输出重现未插桩的结果。Prefill hidden 相同，首次 decode 在拒绝发生前已出现 hidden 差异；1-token 与 8-token 路径的完整状态等价仍未证明。
- 私有 checkout 缺少 wheel 的 ignored 构建/vendored 文件，曾导致 FlashMLA 导入失败；恢复同一 wheel 对应文件后，22 个 overlay 摘要保持通过，实际生成完成。显式使用 Triton MoE；未把可选 DeepGEMM 的 ABI 警告解释为其 backend 已可用。
- 新增目标分片索引完整性检查，并行记录全部目标权重/索引 SHA-256。缺分片用例包含在本轮 27 项 CPU 组件测试中，全部通过。修改的 Python 文件通过定向 Ruff/format/mypy，文档格式通过；没有宣称全仓 Linux pytest 或所有 backend 已通过。

本轮实际 GPU 迁移验证覆盖拆分前的 dense 路径；独立训练 seed、BF16 目标和通用输出等价仍待验证。训练与在线接受计数完成也不能自动把导出配置中的正确性标记改为已验证。
