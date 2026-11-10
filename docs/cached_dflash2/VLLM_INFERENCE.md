# vLLM：dense 与 D-MTP 怎样推理

**本仓库的 D-MTP 需要自定义 vLLM runtime，不能只 `pip install vllm` 后传入训练权重。** 原生 `method=mtp` 本身不会自动组合 DFlash2。这里把实验实际使用的 22 个修改/新增文件、对应 base commit、逐文件 SHA-256 一并放在 [integrations/vllm_dmtp](../../integrations/vllm_dmtp)。这份说明针对该固定 fork，不声称任意版本的上游 vLLM 都支持它。

## 运行时要求与安装

- 私有基线 commit：`bbf5893213eff1e1dc7bbe76ce410b31eaca1507`。
- 历史环境标识：vLLM 0.31.0、PyTorch `2.14.0a0+b2c75dd062.nv26.09`、CUDA 13.4、棒卡。
- 版本字符串相同不代表私有模型代码/编译扩展 ABI 相同。需取得支持该 Qwen target 的原私有基线与配套 CUDA 环境；这个仓库的 overlay 不是完整 vLLM 分发包。
- 固定 Git checkout 还需要配套 wheel 的编译产物和 vendored 文件（例如 `vllm/third_party/flashmla/flash_mla_interface.py`）；只复制 `.so` 不够。使用同一构建的文件，并额外记录其摘要。普通 `import vllm` 通过并不代表量化模块及模型启动已通过。
- 本适配器仅支持该基线的 `Qwen4ExpMTP` target、V2 model runner、单 GPU（TP/PP/DP/DCP/PCP 均 1）、greedy draft、固定七候选。LoRA、adaptive verification、EPLB、watermark、n-gram 叠加不支持。

从已有的正确私有 vLLM 源码创建独立 checkout；不要把 overlay 覆盖到共享环境或任意上游版本。

```bash
export VLLM_SOURCE=/personal/src/original-private-vllm
export RUNTIME_ROOT=/personal/src/vllm-dmtp
git -C "$VLLM_SOURCE" worktree add --detach "$RUNTIME_ROOT" \
  bbf5893213eff1e1dc7bbe76ce410b31eaca1507

# 以下命令在 modelarts-speculators 仓库根目录运行。
python scripts/vllm_dmtp_runtime.py --runtime-root "$RUNTIME_ROOT"       # 只检查计划
python scripts/vllm_dmtp_runtime.py --runtime-root "$RUNTIME_ROOT" --apply
python scripts/vllm_dmtp_runtime.py --runtime-root "$RUNTIME_ROOT" --check
```

应用脚本要求干净的正确 base checkout，校验 bundled 文件摘要后复制；版本或本地改动不匹配直接报错。`--check` 验证 HEAD、22 个文件摘要及额外改动。overlay 保留原始文件字节，不用格式化器批量改写。

在**原配套私有环境**中按照该 fork 原有的 editable 安装/构建流程让 Python 指向这个 checkout；不要把通用训练 venv 当作推理环境。若切换 PyTorch/CUDA，必须重建相应扩展并重新验证。下面启动器会验证 `import vllm` 的实际路径，发现导入到别处就拒绝运行。提供的 `--dry-run` 只打印参数，不检查 GPU、二进制扩展或启动成功。

## 为什么这样接

```python
# dense：一轮 DFlash2 提出七位
speculative_config = {
    "method": "dflash",
    "num_speculative_tokens": 7,
    "model": "/personal/runs/dense/epochs/epoch-05",
}

# D-MTP：调度器七候选；自定义 proposer 内部拆成 native MTP 一位 + DFlash2 六位
speculative_config = {"method": "mtp", "num_speculative_tokens": 7}
additional_config = {
    "swx_dmtp": {"model": "/personal/runs/fusion/epochs/epoch-05"}
}
```

运行时负责：原生 MTP 生成 T1 和对应单流 hidden → token-only 或 residual fusion 构造后段 anchor → DFlash2 同时生成 T2…T7 → target 验证七位 → 按接受长度更新/回滚两套状态。`qwen3_dmtp.py` 加载 fusion 权重，`spec_decode/dmtp/config.py` 检查时序/配置；proposer 还处理 query RoPE 前移一位与 context KV 空洞的差异。不要通过改 `architectures` 字段把 hybrid 伪装成普通 dense。

## 启动 OpenAI 兼容服务

在上面的推理环境中、仓库根目录运行；先 C1/eager/KV4 小检查，正式容量另测。

```bash
export TARGET=/personal/models/qwen-target
export DENSE=/personal/runs/dense-kl-3e-5/epochs/epoch-05
export FUSION=/personal/runs/fusion-kl-3e-5/epochs/epoch-05
export INFERENCE_ROOT=/personal/inference

CUDA_VISIBLE_DEVICES=0 python scripts/serve_cached_dflash2.py \
  --mode dense --target "$TARGET" --draft "$DENSE" \
  --runtime-root "$RUNTIME_ROOT" \
  --provenance-dir "$INFERENCE_ROOT/dense-serve" --port 8000

# 在另一张已分配的空卡上启动另一个进程；两个服务各用一张卡。
CUDA_VISIBLE_DEVICES=1 python scripts/serve_cached_dflash2.py \
  --mode fusion --target "$TARGET" --draft "$FUSION" \
  --runtime-root "$RUNTIME_ROOT" \
  --provenance-dir "$INFERENCE_ROOT/fusion-serve" --port 8001
```

Token-only 换成 `--mode token --draft <token-export>`；原生 MTP 单位对照用 `--mode native-mtp`，不传 draft；无投机目标对照用 `--mode target`。默认监听 `127.0.0.1`。先加 `--dry-run` 可以核对完整实际命令。

客户端可用 OpenAI SDK（`image_data_url` 是自己的图像 data URL）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="unused")
response = client.chat.completions.create(
    model="allava-target",
    temperature=0,
    max_tokens=256,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

`--dtype bfloat16` 是计算 dtype，**不会把 FP8 目标权重转换成完整 BF16 目标**。权重精度以 target checkpoint 与 provenance 为准。普通 HTTP 返回文本并不能提供这次实验的完整接受计数；接受率对照用下面的离线 LLM 评测入口。

目标权重须完整复制后再启动。工具会核对 safetensors 分片索引，并使用四个读取线程计算全部权重及索引的 SHA-256；这些校验在模型初始化之前完成。

## 接受率与输出一致性

Prompt manifest 为 JSONL：每行至少 `id`, `question`, `image_sha256`，再提供 `image_path`，或者原实验的 `image_tar`, `image_offset`, `image_size`。路径可相对 manifest；程序读取原始图像 bytes 后核对 SHA-256。不能用训练集做最终测试，需另外审计与 train/val 的图像交集。

```bash
export TEST_PROMPTS=/personal/data/test-prompts.jsonl

# 先分别跑 target/dense/token/fusion 的16图加载与输出检查。
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cached_dflash2.py \
  --mode dense --target "$TARGET" --target-precision fp8 --draft "$DENSE" \
  --runtime-root "$RUNTIME_ROOT" --prompts "$TEST_PROMPTS" \
  --provenance-dir "$INFERENCE_ROOT/dense-probe" --limit 16

# 容量检查通过后，才切换相同条件跑1000图；不要把两者指标混在一起。
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_cached_dflash2.py \
  --mode fusion --target "$TARGET" --target-precision fp8 --draft "$FUSION" \
  --runtime-root "$RUNTIME_ROOT" --prompts "$TEST_PROMPTS" \
  --provenance-dir "$INFERENCE_ROOT/fusion-test" \
  --limit 1000 --max-tokens 1024 --concurrency 64 \
  --kv-cache-gib 32 --batched-tokens 2048 --graph --repeats 3
```

Dense/token/fusion 必须使用同一 prompts、target、精度、采样、并发上限、KV 容量、kernel、输出长度及 graph 模式。三次同卡重复不等于跨卡轮换，也不等于独立训练 seed。完整对照应再轮换硬件，并保存 GPU 身份。

长评测可加 `--progress`，在日志里显示已完成的请求数量。它只影响进度显示，不改变推理配置；最终指标仍以完成后的 `summary.json` 为准。

评测入口支持 `--gpu-memory-utilization`（默认 `0.92`，范围 `(0, 1]`）。2026-11-10 的三组同条件对照统一使用 `0.65`，同时显式固定 `--kv-cache-gib 32`；前者调整 vLLM 启动时的可用显存门限，不能把两者混为 KV 分配比例。完整命令、GB300 私有权重文件页缓存处理及逐轮结果见 [实跑归档](results/allava_dmtp_20261110/README.md)。

输出 `requests.jsonl` 保留 token IDs 和接受计数；`summary.json` 的在线 AR 是 accepted/proposed，AL 是 `1 + accepted/verification_steps`。缺少底层计数会报错，不会以独立 token accuracy 冒充接受率。吞吐包含 multimodal prefill 和 decode，排除模型加载、读图和 warmup；当前不按吞吐选配方。

```bash
python scripts/compare_cached_dflash2_outputs.py \
  "$INFERENCE_ROOT/target-probe/requests.jsonl" \
  "$INFERENCE_ROOT/fusion-probe/requests.jsonl" \
  --output "$INFERENCE_ROOT/target-vs-fusion.json"
```

该工具要求相同请求/图像身份，比较完整 token IDs 和 finish reason；发现差异返回非零，并记录首个差异位置。也可比较同一模式不同 GPU 或不同 repeat。有限样本相同不等于通用无损证明。

## 已知未解决项

历史 1000 图三卡重复中，dense 输出一致；token/fusion 各有一张 GPU 的 990/1000 条输出异于另两张，而另两张相互一致。此前独立复现了 BF16 HC 投影在不同矩阵形状下的数值分支差异，但这还不是跨卡异常的完整解释。**本分支迁移了实际 runtime，没有宣称修复该问题。**

下一步应保存同一固定前缀下的 native/hybrid T1、pending token 位置/内容、拒绝后状态、hidden/logit 误差，再区分精度/内核/缓存原因。不得用不同在线生成前缀上的首位率当作配对证据。BF16 目标、独立训练 seed、完整输出等价均仍待验证。
