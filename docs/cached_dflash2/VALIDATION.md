# 迁移验证记录

日期：2026-11-04。以下为代码集成的本地验证记录；没有重新训练全量 ALLaVA。服务器验证记录在后续实际完成后补充。

## 已执行

- `python scripts/test_cached_dflash2_cpu.py`：25 项通过。包括 FP32/BF16 full-vocab loss 与梯度参考、teacher 停止梯度、prefix 分母、epoch 覆盖、未来 target 特征零影响/零梯度、teacher-forced/native T1 分离、零初始化融合及其梯度、三种模式各两轮微型合成训练、导出、冻结 selector、中断续训与不中断权重逐元素一致、配置/缓存错误拒绝。
- 既有 `test_resolve_mask_token_id.py` 的 7 项测试在隔离导入下通过，确认延迟加载预处理依赖后保留原 helper 的 monkeypatch 接口。
- 同一测试集检查实际 runtime fusion/window 数学、全部 22 个 overlay 原始 hash、安装的 base/干净目录/篡改保护、dense/hybrid 启动配置、在线计数和完整输出差异定位。
- `mypy --check-untyped-defs src/speculators/train/cached_dflash2 src/speculators/cli/train_cached_dflash2.py tests/unit/train/cached_dflash2`：19 个文件通过。
- 仓库 Ruff 检查及格式检查通过；本次 Markdown 文档按仓库 mdformat 配置检查。
- 新绘图工具读取历史两组完整的 2815 步日志，生成 PNG/PDF，并人工检查图片布局与坐标。文档中的两套历史曲线保持原归档内容，不是合成测试曲线。

CPU 环境：Intel macOS、Python 3.12、PyTorch 2.2.2、NumPy 1.26.4。CPU attention 显式重复 KV heads；CUDA 分支仍用原生 `enable_gqa=True`。CPU runner 只加载本功能及轻量 provenance 模块，绕过无关的包级 Transformers 模型注册；不能把它称作完整安装包测试。

## 环境阻塞与未验证项目

`uv pip install -e '.[dev]'` 实际失败：仓库要求的 `torch>=2.9` 没有 Intel macOS wheel，因此无法在本机安装完整受支持环境。正常 pytest 入口及全仓 mypy 受未安装的上游依赖/不兼容本机 torch 影响，没有宣称通过。只对新增源代码与测试通过的定向 mypy 作出通过声明。

需要在配套 Linux/CUDA 环境继续：

```bash
uv sync --extra dev
python -m pytest tests/unit/train/cached_dflash2
make quality
```

然后按推理文档先做目标/dense/token/fusion 小图检查、输出比较及容量验证，再运行完整在线对照。服务启动器和新评测 wrapper 本次仅做 CPU 配置/静态检查；不能声称已经在新安装环境中服务成功。旧 runtime 的 GPU 实验是历史证据，不替代迁移后的 GPU 回归。

## 尚未解决的科学结论

旧实验的高并发 D-MTP 跨卡输出差异没有在这次迁移中修复。独立训练 seed、BF16 target、完整 native/hybrid 状态一致性及输出等价仍待验证。历史融合组接受指标的小幅变化不能当作已验证的无损收益，也不能保证扩大训练必然有效。

开发 hook 已安装；首次 pre-commit bootstrap 在获取 GitHub hook 仓库时网络超时。对应 Ruff、format、mdformat 和 provenance sync 已直接执行；全仓 mypy 的环境阻塞单独记录，不冒称 hook 全部通过。
