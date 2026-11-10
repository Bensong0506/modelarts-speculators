# Dense DFlash2 的 vLLM 推理

此分支的 wrapper 只支持 `--mode target` 和 `--mode dense`。Dense 一轮提出七个草稿 token，配置为 `method=dflash`；不调用原生 MTP 生成首位，也不启用混合 proposer。D-MTP 实验继续使用其独立分支。

## 固定运行时

这里仍需要匹配 Qwen target 的私有 vLLM，不能仅安装任意上游 `pip vllm`。

- Base commit：`bbf5893213eff1e1dc7bbe76ce410b31eaca1507`。
- 历史环境：GB300、vLLM 0.31.0 private fork、PyTorch `2.14.0a0+b2c75dd062.nv26.09`、CUDA 13.4。版本字符串不能证明 ABI 或模型实现一致。
- 适配范围为该基线的 `Qwen4ExpMTP` target、V2 runner、单 GPU、greedy、七候选、单图输入。这里的 target 架构名称包含 MTP，不代表 dense 草稿流程会使用它。
- `integrations/vllm_dflash2` 保留原 22 文件 overlay 的完整字节和原始来源记录，其中包含未启用的 MTP/vendor 模块。这是依赖归档；本分支的业务训练器和 wrapper 不开放 hybrid 路径。裁剪 vendor 会产生另一版未经验证的运行时。
- 配套 wheel 的编译扩展和 ignored vendored 文件也必须完整，例如 FlashMLA Python 接口；仅复制 `.so` 或仅成功 `import vllm` 不等于模型能启动。

从正确的私有源码建立个人 checkout，在 Speculators 仓库根目录执行：

```bash
export VLLM_SOURCE=/personal/src/original-private-vllm
export RUNTIME_ROOT=/personal/src/vllm-dflash2
git -C "$VLLM_SOURCE" worktree add --detach "$RUNTIME_ROOT" \
  bbf5893213eff1e1dc7bbe76ce410b31eaca1507
python scripts/vllm_dflash2_runtime.py --runtime-root "$RUNTIME_ROOT"
python scripts/vllm_dflash2_runtime.py --runtime-root "$RUNTIME_ROOT" --apply
python scripts/vllm_dflash2_runtime.py --runtime-root "$RUNTIME_ROOT" --check
```

也可从已交付包的 `runtime/vllm-base.bundle` 克隆该基线。运行时必须按私有 fork 的配套安装/构建方式准备，并确保当前 Python 实际导入这个 checkout。脚本会核对 base、22 文件 hash、额外修改和导入路径，不能通过改 hash 绕过。这里没有打包可直接安装的 CUDA 环境。

## 小图检查与服务

准备完整目标权重和本轮 dense 导出；目标分片索引、全部权重 SHA-256、draft SHA-256 及运行参数会写入独立 provenance 目录。目标是否 FP8 以实际权重为准，`dtype=bfloat16` 不会转换目标权重精度。

```bash
export TARGET=/personal/models/qwen-target
export DENSE=/personal/runs/business-dense-001/dense-kl-3e-5/epochs/epoch-05
export INFERENCE_ROOT=/personal/results/business-dense
export TEST_PROMPTS=/personal/data/validation-prompts.jsonl

# 仅查看参数；dry-run 不证明 CUDA 或权重加载成功。
python scripts/serve_cached_dflash2.py \
  --mode dense --target "$TARGET" --draft "$DENSE" \
  --runtime-root "$RUNTIME_ROOT" \
  --provenance-dir "$INFERENCE_ROOT/dense-service" --dry-run
```

去掉 `--dry-run` 才会启动服务；默认绑定 `127.0.0.1:8000`，服务模型名为 `allava-target`（历史 API 名称，业务训练数据不受此名称限制）。服务启动与评测需单独安排显存，避免同一 GPU 同时重复加载模型。

在线 prompts 为 JSONL，每行至少 `id`、`question`、`image_sha256`、`image_path`，图片路径可相对 manifest。也兼容原实验 tar 的 `image_tar`、`image_offset`、`image_size`；读图后核对原始字节 SHA-256。当前请求模板是单图 user 问答且关闭 thinking，多轮/多图/工具业务需另行适配真实线上模板。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cached_dflash2.py \
  --mode target --target "$TARGET" --target-precision fp8 \
  --runtime-root "$RUNTIME_ROOT" --prompts "$TEST_PROMPTS" \
  --provenance-dir "$INFERENCE_ROOT/target-probe" --limit 16

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cached_dflash2.py \
  --mode dense --target "$TARGET" --target-precision fp8 --draft "$DENSE" \
  --runtime-root "$RUNTIME_ROOT" --prompts "$TEST_PROMPTS" \
  --provenance-dir "$INFERENCE_ROOT/dense-probe" --limit 16

python scripts/compare_cached_dflash2_outputs.py \
  "$INFERENCE_ROOT/target-probe/requests.jsonl" \
  "$INFERENCE_ROOT/dense-probe/requests.jsonl" \
  --output "$INFERENCE_ROOT/target-vs-dense.json"
```

默认小检查 C1/eager/KV4 GiB。比较工具要求同一请求/图像身份，完整 token IDs 或结束原因不同时返回非零，并保留首差位置。已知历史输出比较没有通过，不能忽略非零退出或宣称无损，详见 [VALIDATION.md](VALIDATION.md)。

## 同口径业务接受率

确认加载、容量和正确性状态后，才运行业务验证集。下面 C64/KV32/graph 是历史大显存配置，必须先按内网硬件做容量检查；`--limit 100` 要求准备相应验证样本。每个 checkpoint 使用新 provenance 目录：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_cached_dflash2.py \
  --progress --mode dense --target "$TARGET" --target-precision fp8 \
  --draft "$DENSE" --runtime-root "$RUNTIME_ROOT" --prompts "$TEST_PROMPTS" \
  --provenance-dir "$INFERENCE_ROOT/dense-epoch05-val100" \
  --limit 100 --max-tokens 1024 --concurrency 64 \
  --kv-cache-gib 32 --batched-tokens 2048 --graph --repeats 1
```

本轮 epoch 0 与各 epoch 保持相同 prompts、target、精度、采样、并发上限、KV、kernel、长度及 graph 条件。验证集选方后再用独立 test 复核；重复评测不等于独立训练 seed。

`requests.jsonl` 保存输出 token IDs 与底层接受计数；`summary.json` 的 AR 是 accepted/proposed，AL 是 `1 + accepted/verification_steps`，含 bonus。它统计整段真实生成，不是每图 8 个缓存 anchor。缺失接受计数会报错，不能用 token accuracy 代替。缓存验证、在线接受率和输出一致性分别报告。
