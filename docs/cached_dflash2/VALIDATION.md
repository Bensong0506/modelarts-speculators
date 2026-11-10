# 迁移验证记录

本记录区分新分支实跑与历史预检查。运行数据、权重、连接信息和逐请求结果保留在操作者的私人实验记录中。

## 2026-11-10：D-MTP 两组五轮与逐 checkpoint 在线对照

完整的 [结果、设置、曲线和复核入口](results/allava_dmtp_20261110/README.md) 已归档；公开部分保留数值计数和摘要，原始图像/回答及私有路径不随仓库发布。

- Token-only / residual-zero fusion 从与 dense 相同的初始化和 batch plan 出发，17998 有效训练图、各五轮 2815 更新、89990 图次，训练正常结束；selector 与原生 MTP 冻结。全部 12 份 D-MTP 导出检查为 BF16。
- 两组独立 3 步 pilot、16 图 runtime、64 图容量均完成。随后 dense/token/fusion 各 epoch 0–5，同一批 100 验证图、同 epoch 同 GPU 的 18 次完整在线评测结束；FP8 目标、BF16 草稿、C64 上限、KV32 GiB、B2048、graph、K7、max1024。目标摘要一致、checkpoint 及输入合同审计通过。
- 第五轮 dense AR 24.877125% / AL 2.741399；token AR 24.401003% / AL 2.708070；fusion AR 25.323670% / AL 2.772657。Fusion 相比 dense +0.446545 个百分点、+0.031258 AL；图像配对 AL 差值 95% 区间 `[0.0002103, 0.0628615]`，优势小且边缘。Token-only 下降。仅一个训练 seed 和 100 张验证图，最佳 checkpoint 选择为探索性，未校正多重比较。
- 初始三种草稿各 16/16 输出不同于 target-only；初始 token/zero-fusion 的 16 图及 100 图互相完全一致。额外 dense epoch 5 跨 GPU 100 图输出完全一致，但没有重新做 D-MTP 跨 GPU 重复；历史异常不能据此判定已解决。
- 私有 SSD 文件页缓存曾影响 GB300 启动的可用显存判断。实跑在权重 SHA 校验后只对本作业拥有的文件做 `POSIX_FADV_DONTNEED`，并统一设置启动门限 `gpu_memory_utilization=0.65`，显式 KV 仍为 32 GiB。提交评测参数修复，全部 22 个 runtime overlay 文件保持原字节。
- 训练与在线接受计数完成，**不等于目标输出等价或无损吞吐收益**。BF16 目标、独立 test/训练 seed、完整状态等价仍未验证。

## 2026-11-10：dense 路径实际训练与推理

- 在 ARM64 / GB300、PyTorch `2.14.0a0+b2c75dd062.nv26.09`、CUDA 13.4 的私有环境中，完整包入口完成了独立 3 步 pilot 和 64 图、1 epoch 的小训练。
- 完整 ALLaVA 缓存有 20,000 条，按图像分为 18,000 train / 1,000 val / 1,000 test；2 条训练响应过短被排除。正式 dense KL `3e-5` 训练完成 5 epoch、2815 更新，epoch 0–5 全部导出；loss/梯度有限，selector 校验未改变。
- 起始权重是此前 ALLaVA 实验选出的已增训 dense checkpoint；本轮衡量额外五轮 continuation，不是从未增训原始 DFlash2 开始。
- FP8 目标、C1/eager、16 图的 target-only / 起点 dense / 小训练 dense 三组均完成真实 vLLM 生成；C64 上限、KV32 GiB、graph 的 64 图容量检查通过。随后 epoch 0–5 各完成同一批 100 验证图、max1024 的在线评测，训练及完整评测控制器正常退出。逐请求计数、权重摘要与图像配对区间保存在个人实验记录中；没有把容量检查、缓存指标或历史大样本结果混入该对照。
- 各轮目标权重摘要一致，draft 摘要与对应 epoch 导出一致，100 张验证图与训练集无图像交集。选择结果只覆盖这一批验证图和一个训练 seed，未额外运行独立 test，也未通过输出等价验证。
- 起点 dense 的 16/16 条输出异于 target-only，不能称为无损验证通过。在两个选定差异样本上，三条路径共 1536 个实际输出 token 均符合各自当步 target argmax，且追踪输出重现未插桩的结果。Prefill hidden 相同，首次 decode 在拒绝发生前已出现 hidden 差异；1-token 与 8-token 路径的完整状态等价仍未证明。
- 私有 checkout 缺少 wheel 的 ignored 构建/vendored 文件，曾导致 FlashMLA 导入失败；恢复同一 wheel 对应文件后，22 个 overlay 摘要保持通过，实际生成完成。显式使用 Triton MoE；未把可选 DeepGEMM 的 ABI 警告解释为其 backend 已可用。
- 新增目标分片索引完整性检查，并行记录全部目标权重/索引 SHA-256。缺分片用例包含在本轮 27 项 CPU 组件测试中，全部通过。修改的 Python 文件通过定向 Ruff/format/mypy，文档格式通过；没有宣称全仓 Linux pytest 或所有 backend 已通过。

以上是当日较早的 dense 验证；其后两组 D-MTP 的实际迁移运行见本页顶部。独立训练 seed、BF16 目标和通用输出等价仍待验证。训练与在线接受计数完成也不能自动把导出配置中的正确性标记改为已验证。

## 2026-11-04：历史预检查快照

日期：2026-11-04。以下记录本地测试与服务器预检查；没有重新训练全量 ALLaVA，也没有完成新分支的在线推理验收。

### 已执行

- `python scripts/test_cached_dflash2_cpu.py`：26 项通过。包括 FP32/BF16 full-vocab loss 与梯度参考、teacher 停止梯度、prefix 分母、epoch 覆盖、未来 target 特征零影响/零梯度、teacher-forced/native T1 分离、零初始化融合及其梯度、三种模式各两轮微型合成训练、导出、冻结 selector、中断续训与不中断权重逐元素一致、配置/缓存错误拒绝，以及真实 vLLM int32 target IDs 与 int64 native-MTP sidecar 的兼容和浮点 ID 拒绝。
- 既有 `test_resolve_mask_token_id.py` 的 7 项测试在隔离导入下通过，确认延迟加载预处理依赖后保留原 helper 的 monkeypatch 接口。
- 同一测试集检查实际 runtime fusion/window 数学、全部 22 个 overlay 原始 hash、安装的 base/干净目录/篡改保护、dense/hybrid 启动配置、在线计数和完整输出差异定位。
- `mypy --check-untyped-defs src/speculators/train/cached_dflash2 src/speculators/cli/train_cached_dflash2.py tests/unit/train/cached_dflash2`：19 个文件通过。
- 仓库 Ruff 检查及格式检查通过；本次 Markdown 文档按仓库 mdformat 配置检查。
- 新绘图工具读取历史两组完整的 2815 步日志，生成 PNG/PDF，并人工检查图片布局与坐标。文档中的两套历史曲线保持原归档内容，不是合成测试曲线。

CPU 环境：Intel macOS、Python 3.12、PyTorch 2.2.2、NumPy 1.26.4。CPU attention 显式重复 KV heads；CUDA 分支仍用原生 `enable_gqa=True`。CPU runner 只加载本功能及轻量 provenance 模块，绕过无关的包级 Transformers 模型注册；不能把它称作完整安装包测试。

### 服务器预检查

在用户个人目录恢复了固定私有 vLLM 环境，独立 checkout 的 base commit 与全部 22 个 overlay 摘要检查通过。宿主机为 ARM64；只读复用基础 PyTorch 2.13.0+cu130 / CUDA 13.0，未修改共享基础环境。BF16 GPU 矩阵乘法通过，离线补齐依赖后的完整 `import speculators`、`import vllm` 通过；vLLM 实际导入路径指向已检查的私有 checkout。

实际读取已有缓存时发现 target `ids` 是 int32、MTP sidecar `ids` 是 int64，因此修复了原先过严的 dtype 校验：只允许这两种整数类型并统一为 int64，保持逐 token 身份核对；合成 fixture 也改为匹配真实数据格式。

服务器不通 PyPI，恢复时使用离线依赖，并修复了旧容器绝对链接和系统 site-packages 混用。当前宿主机 PyTorch/CUDA 不同于历史实验镜像，不能假定旧二进制扩展的所有 GPU 路径兼容。这里只完成导入、摘要与基础运算检查，未运行完整模型训练、GPU pytest 或在线生成；未启动常驻训练/推理任务。后续 GPU 验证需另行安排。

### 环境阻塞与未验证项目

`uv pip install -e '.[dev]'` 实际失败：仓库要求的 `torch>=2.9` 没有 Intel macOS wheel，因此无法在本机安装完整受支持环境。正常 pytest 入口及全仓 mypy 受未安装的上游依赖/不兼容本机 torch 影响，没有宣称通过。只对新增源代码与测试通过的定向 mypy 作出通过声明。

需要在配套 Linux/CUDA 环境继续：

```bash
uv sync --extra dev
python -m pytest tests/unit/train/cached_dflash2
make quality
```

然后按推理文档先做目标/dense/token/fusion 小图检查、输出比较及容量验证，再运行完整在线对照。服务启动器和新评测 wrapper 本次仅做 CPU 配置/静态检查；不能声称已经在新安装环境中服务成功。旧 runtime 的 GPU 实验是历史证据，不替代迁移后的 GPU 回归。

### 尚未解决的科学结论

旧实验的高并发 D-MTP 跨卡输出差异没有在这次迁移中修复。独立训练 seed、BF16 target、完整 native/hybrid 状态一致性及输出等价仍待验证。历史融合组接受指标的小幅变化不能当作已验证的无损收益，也不能保证扩大训练必然有效。

开发 hook 已安装；首次 pre-commit bootstrap 在获取 GitHub hook 仓库时网络超时。对应 Ruff、format、mdformat 和 provenance sync 已直接执行；全仓 mypy 的环境阻塞单独记录，不冒称 hook 全部通过。
