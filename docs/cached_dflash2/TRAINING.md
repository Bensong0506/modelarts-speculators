> 当前分支新增 [DCORE 离线特性对照](FEATURE_DCORE.md)。下文原 dense 约定适用于关闭特性的路径；新结构、loss 和导出边界以特性指南为准。

# 训练与缓存契约

使用内网真实业务数据时，按 [业务数据训练交接](BUSINESS_DATA.md) 的完整顺序执行。它提供业务路径配置、缓存检查、pilot 和正式训练命令，并解释短回答、多轮与业务分组等当前限制；下文是共用的缓存与训练契约。

## 环境与输入

在自己的 Linux/CUDA 环境中安装仓库依赖。训练不依赖 vLLM；蒸馏、缓存采集及在线验证另外使用适配的 vLLM 环境。

```bash
uv sync --extra dev
source .venv/bin/activate
pre-commit install
speculators train-cached-dflash2 --help
```

也可用 `python -m speculators.train.cached_dflash2`。这是单 GPU 单进程训练，不使用 torchrun/DDP；多卡用于同时运行不同配方。不要用 torchrun 对同一输出目录启动多份训练。模型、数据、环境、日志和缓存放在自己的目录中。

输入包括：

1. 同一初始 dense checkpoint：`config.json` + `model.safetensors`。所有 dense 配方从明确记录的同一初始化出发。
2. `shared.pt`：字典 `embed_tokens.weight` / `lm_head.weight`，形状均为 `[vocab_size, hidden_size]`；从同一目标权重提取并冻结。
3. manifest JSON 数组，以及它指向的 target cache。支持相对 manifest 的缓存路径，便于迁移。

报告中应注明初始 checkpoint 的训练历史和权重摘要。`epoch-00` 表示本轮起点：如果输入已经经过 ALLaVA 增训，后续指标衡量的是继续增训的变化，不能把它称为相对未增训原始 DFlash2 的总收益。

这不是直接读取图片 JSONL 的 SFT。预处理顺序为：按图像身份划分 train/val/test → 目标 Qwen 读取图片和问题、蒸馏回答 → 同一 tokenizer/chat template/图像预处理下采集 target hidden → 合并对齐 manifest。既有采集程序在[历史归档 source](https://github.com/Bensong0506/speculators-wenxuan/tree/bb5a1b1a4dbf97d4cd968f011a97fb458f229dc7/research/allava20k_finetune_20261104/02_dflash2_dmtp/source/experiments)中；本命令消费其缓存，不会自动下载数据或调用其他服务。

## 文件格式

Manifest 示例（实际文件为 JSON 数组）：

```json
[
  {
    "id": "sample-1",
    "split": "train",
    "image_sha256": "sha256-of-the-original-image-bytes",
    "tokens": 1024,
    "prompt_length": 600,
    "path": "target/sample-1.pt"
  }
]
```

需要非空且图像身份互不相交的 train/val；test 不用于选 checkpoint。重复 sample ID 或跨 split 图像重复会报错。`tokens - prompt_length < 16` 的样本在训练/验证中排除并记录。可加入 `sha256`，让 `--check-data` 核对 cache 文件本身的摘要。

Target `.pt` 使用 `torch.load(weights_only=True)`，包含：

| key            | 形状/类型                           | 含义                                                             |
| -------------- | ----------------------------------- | ---------------------------------------------------------------- |
| ids            | `[L]`, int32 或 int64               | 同一图片、prompt 和蒸馏回答的完整 token 序列；加载后统一为 int64 |
| features       | `[L, target_layer_count * H]`, BF16 | 选定 target 层的 collapsed hidden，非 HC 多流拼接                |
| teacher_hidden | `[L, H]`, BF16                      | 目标 LM head 的输入；位置 i 用来预测 token i+1                   |
| prompt_length  | int                                 | assistant 区域起点，必须与 manifest 一致                         |
| image_sha256   | str                                 | 原图身份，必须与 manifest 一致                                   |

Anchor `a` 表示已发出但尚待 target 处理的 A：目标 context 严格截止 `a-1`。Dense 输入 A 加七个 mask，监督 hidden 下标为 `a…a+6`，预测后七个 token；不得引入未来 target 特征。

`cache_target_precision` 是操作者声明的采集条件，不能单凭 BF16 hidden dtype 推断目标权重是 BF16。历史缓存来自 FP8 目标；改标签不会把它变成 BF16 对照。需从采集 provenance 独立核实权重、量化、tokenizer、模板、图像预处理和 target layer IDs。

## 启动、恢复与查看

```bash
export DFLASH2_INITIAL=/personal/models/dflash2-initial
export DFLASH2_SHARED=/personal/cache/shared.pt
export ALLAVA_CACHE_MANIFEST=/personal/cache/manifest.json
export EXPERIMENT_ROOT=/personal/runs/allava-recipes

# CPU 完整扫描缓存，不启动训练、不创建 output。
speculators train-cached-dflash2 \
  --config examples/cached_dflash2/dense-kl-3e-5.yaml --check-data

# 先用独立目录做 3 步 pilot，不能把 pilot 当正式结果。
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config examples/cached_dflash2/dense-kl-3e-5.yaml \
  --output "$EXPERIMENT_ROOT/dense-pilot" --pilot-steps 3 --pilot-val-images 8

# 正式运行：选择 dense 的 KL/CE 与学习率配方。
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config examples/cached_dflash2/dense-kl-3e-5.yaml

# 从最后完整 epoch 恢复，输入/源码/配方必须与 provenance 一致。
CUDA_VISIBLE_DEVICES=0 speculators train-cached-dflash2 \
  --config "$EXPERIMENT_ROOT/dense-kl-3e-5/run.yaml" --resume

python scripts/plot_cached_dflash2.py \
  "$EXPERIMENT_ROOT/dense-kl-3e-5" \
  --output "$EXPERIMENT_ROOT/plots"
```

绘图需要 matplotlib（可在独立绘图环境安装）；不需要模型或 GPU。配置的环境变量必须已设置，CLI 覆盖 YAML。输出目录首次必须不存在；已有输出只能通过 `--resume` 进入。文件锁避免两个训练器写同一目录。

默认五轮、batch 上限 32、padded context budget 48000、每图 64 training anchors、验证 8 anchors、seed 280、20% linear warmup 后 linear decay。历史 17998 个有效 train 样本的每轮为 563 步，共 2815；换数据/长度后以 provenance 生成的 batch plan 为准，不硬编码这个步数。

使用 FP32 trainable master weights/Adam，BF16 autocast，冻结 embedding、LM head、candidate selector。KL 为同前缀 full-vocab teacher 分布；CE 标签是同前缀 teacher argmax，**不是 transcript ID**。位置权重为 `exp(-slot/4)` 后归一到均值 1；dense 七位。head 按 128 hidden rows 分块计算完整 vocabulary；不会只在 top-16 内算 KL。top-16 用于冻结 selector 的候选选择。

`memory_limit_gib` 是本进程 PyTorch allocator 上限，非目标显存占用；CUDA 启动要求另留 16 GiB。提高 batch/context/anchors 会改变实验配方，不能静默调大后与旧记录混用。

## 产物与指标

- `run.yaml`, `train_command.txt`, `speculators.patch`, `provenance.json`：配方、源码/输入/初始化摘要、版本及 batch plan。
- `train.jsonl`：每次真实 optimizer 更新；`progress.json` 是阶段快照，不足以证明进程存活。
- `initial.json`, `validation.json`, `summary.json`：缓存前缀代理指标，始终标记 `online_validated=false`。
- `epochs/epoch-00…05/{config.json,model.safetensors,export.json}`：BF16 推理导出，不包含冻结的共享 embedding/head。
- `resume.pt`：最后完整 epoch 的 FP32 权重、优化器与 RNG。只加载自己的可信续训文件。崩溃后未持久化的日志挪到 `rolled-back-train.jsonl`，再重跑该 epoch 的未保存部分，不重复累计步数。

默认按 `teacher_selected.accepted_length` 选缓存最佳 epoch，包含 epoch 0；再做同口径在线复核。缓存 AR = 接受的连续前缀 token 数 / `(7 × blocks)`；AL = `1 + 连续前缀数 / blocks`，包含 bonus。独立位置正确率不是接受率，transcript 指标也不是 teacher 指标。在线生成发生分歧后，缓存后续 teacher 标签仍沿旧 transcript，因此不能用缓存 AL 宣称在线提升。
