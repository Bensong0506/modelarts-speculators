# 内网业务数据训练交接

本指南对应分支 `codex/dflash2-allava-dmtp-training` 的 `speculators train-cached-dflash2`。训练内容可以是业务数据，不要求 ALLaVA；**能直接训练的输入是下面定义的目标模型特征缓存，不是图片/prompt/answer JSONL，也不能默认兼容其他 Speculators 缓存格式。** 本指南提供数据接入与执行步骤，没有声称已在接收方的内网数据或硬件上验证。

## 同事 AI 的执行顺序

1. 阅读仓库 `AGENTS.md` 和 [AI_HANDOFF.md](AI_HANDOFF.md)，确认拿到上述分支，记录 `git rev-parse HEAD` 和本地修改。
2. 核对数据准备到了哪一层。只有图片/消息/蒸馏回答时，先在内网采集同一 target 的 hidden；已有缓存时，按本指南核对字段、时间对齐、来源与 split，再执行 `--check-data`。
3. 在内网配置独立环境、初始草稿权重、共享 embedding/head 和缓存路径。先完成 dense 的缓存检查及独立 3 步 pilot，再从同一初始化启动正式训练。
4. 每轮记录 loss、缓存接受率/接受长度、导出路径和实际退出码。选出的 checkpoint 还要按 [VLLM_INFERENCE.md](VLLM_INFERENCE.md) 做业务数据在线验证。

业务数据、图片、生成回答、权重、缓存、连接配置和凭据留在内网个人目录。本 GitHub 仓只提交通用代码与说明；不要把真实业务样例或含个人路径的运行配置提交进来。

## 开训前的输入与兼容条件

| 输入         | 必须满足                                                                                                                                 |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 初始 DFlash2 | 本训练器兼容的 dense `config.json` 和单文件 `model.safetensors`；不是任意草稿架构，也不能把 fusion/MoE 权重当作 dense 初始化             |
| `shared.pt`  | 来自缓存所对应 target 的 `embed_tokens.weight`、`lm_head.weight`，均为浮点 Tensor，形状 `[V, H]`，与草稿配置的词表大小/hidden size 一致  |
| Target cache | 每样本一份下述 `.pt`，所有样本使用同一 target 权重、精度、tokenizer、chat template、图像预处理及特征层配置                               |
| Manifest     | UTF-8 JSON **数组**，有有效且互不泄漏的 train/val；test 如提供只用于最终测试，不参与选 checkpoint                                        |
| 采集来源记录 | 内网保存 target 权重摘要、精度、tokenizer/template/processor 版本、target layer IDs 及拼接顺序、采集代码版本、生成参数和初始草稿训练历史 |

`H`、`V` 和特征层数量 `N` 从初始草稿配置读取：`hidden_size`、`vocab_size`、`len(dflash_config.target_layer_ids)`。形状相同不代表内容来自同一目标模型；改变 target 权重或量化后，不能只改 manifest 标签继续使用旧缓存。

已交付的 ALLaVA epoch 5 可以作为兼容 target 下业务 continuation 的初始化，但它已经训练过 ALLaVA。此时本轮 `epoch-00` 就是该初始化，业务收益应相对这个起点比较；不是未增训的原始 DFlash2。`shared.pt` 未包含在草稿导出中，需要从匹配的 target 另行准备。

已有人工答案或其他模型生成的答案不等于已经完成本训练器需要的蒸馏/采集。要复现现有配方，应使用实际部署 target 生成回答并采集特征；改用其他回答前缀属于新的训练数据配方，需要单独记录和验证。

## Manifest 与缓存格式

推荐内网布局：

```text
/personal/business-cache/
  manifest.json
  capture-provenance.json
  shared.pt
  target/
    train-0001.pt
    val-0001.pt
```

`capture-provenance.json` 是需要保存的采集证据，当前训练器不会自动读取并核验它。Manifest 最小结构如下，示例字符串需替换为实际摘要；文件路径可相对 manifest，也可为内网绝对路径：

```json
[
  {
    "id": "train-0001",
    "split": "train",
    "image_sha256": "<该训练样本原图字节的SHA256>",
    "tokens": 1200,
    "prompt_length": 800,
    "path": "target/train-0001.pt",
    "sha256": "<target/train-0001.pt文件的SHA256>"
  },
  {
    "id": "val-0001",
    "split": "val",
    "image_sha256": "<另一张验证原图字节的SHA256>",
    "tokens": 1000,
    "prompt_length": 700,
    "path": "target/val-0001.pt",
    "sha256": "<target/val-0001.pt文件的SHA256>"
  }
]
```

`id` 必须全局唯一；`split` 只能是 `train`、`val`、`test`。`tokens=L`，`0 < prompt_length < L`。`sha256` 是可选字段，填写后 `--check-data` 会核验缓存文件字节；不要填占位符或用图像摘要代替缓存摘要。Dense 不需要 `mtp_path`。

Target `.pt` 应为 `torch.save` 保存的普通字典，可通过 `torch.load(..., weights_only=True)` 加载：

| 字段             | 形状 / 类型                  | 要求                                                                                                             |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `ids`            | `[L]`，int32 或 int64 Tensor | 图像、完整 prompt 和回答使用实际 target 的同一 token 序列；采集与训练的 token 下标不能改变，IDs 在有效词表范围内 |
| `features`       | `[L, N * H]`，BF16 Tensor    | 按草稿 `target_layer_ids` 顺序拼接各层 collapsed hidden；不能替换成任意最后一层或 HC 多流展平结果                |
| `teacher_hidden` | `[L, H]`，BF16 Tensor        | 实际 target LM head 的输入；位置 `i` 预测 `ids[i+1]`，不是 logits，也不是预先右移一位的 hidden                   |
| `prompt_length`  | Python int                   | 与 manifest 完全一致，按采集时的真实 assistant 回答边界确定，不能用字符数代替 token 下标                         |
| `image_sha256`   | str                          | 与 manifest 的原图身份完全一致                                                                                   |

两个 hidden Tensor 都必须全为有限数值。保留模型实际的多模态 token 展开、位置及上下文规则，不能删掉视觉 token 后继续沿用旧下标。这里的 cache hidden 是 BF16，不代表生成它的 target 权重是 BF16。

Dense anchor `a` 的可见 target context 严格为 `<a`，`ids[a]` 是已发出但等待 target 消费的 pending token；草稿预测后七位，监督 hidden 使用 `a…a+6`。不要为了让 loss 更低把未来 token 的 target 特征放入 context。

## 业务数据特别需要核对的边界

- **短回答**：当前代码排除 `tokens - prompt_length < 16` 的 train/val 样本，并在正式运行的 `provenance.json.excluded_short` 中记录。分类、短 JSON、是/否等业务可能大量被排除；先统计有效/排除数量。不要填充虚假回答绕过，也不要把 `16` 当作现有 YAML 可修改参数。确需短回答训练，应单独改窗口与 mask 逻辑并验证。
- **多轮与工具消息**：当前缓存只有一个 `prompt_length`，没有逐 token 的 assistant loss mask。可将每个要训练的 assistant 回答单独作为一条样本，之前的完整对话作为 prompt，该回答之后不再拼接 user/tool 消息；同一会话的样本留在同一 split。不能把所有轮次简单串起来让后续 user/tool token 进入监督区。
- **多图、纯文本、视频**：当前身份契约围绕单图 `image_sha256`；这些输入不能用伪造图像 hash 冒充已适配。需要明确媒体身份、采集方式和在线评测入口，并额外检查任一共享媒体不跨 split。现有在线 wrapper 是单图输入，不能直接当作通用业务 evaluator。
- **数据隔离**：自动检查只覆盖完全相同的 `image_sha256` 跨 split；文档不同页、同一客户/会话/订单、近重复图片等需在生成 manifest 前按业务分组划分，再保存分组审计结果。只增加 `group_id` 字段不会自动启用分组检查。
- **长度与显存**：`context_token_budget` 约束训练 batch 的最大长度乘 batch size；单条 train 序列超出会报错，不会自动截断。该参数也不等于模型支持的最大上下文。验证 batch 单独受 `eval_batch_sequences` 控制，长验证样本可能需要减小它。

## 内网环境与业务配置

这条路径是**单 GPU、单进程**训练，不要用 `torchrun`/DDP 对同一 output 启动多份。确认 GPU 实时分配和空闲容量；业务数据与缓存准备好不等于二进制环境已经匹配。训练使用 PyTorch/CUDA，不需要启动 vLLM；缓存采集及在线推理环境另行按推理文档准备。

在内网先备齐本分支源码及依赖，使用内网镜像或与 Python/CUDA/架构匹配的离线 wheelhouse。不要让安装器回落公网，也不要为安装依赖覆盖共享 PyTorch。已备齐依赖和构建工具时，可在个人虚拟环境、仓库根目录安装：

```bash
python -m pip install --no-index --find-links /personal/wheelhouse \
  --no-build-isolation -e ./hs_connectors -e '.[dev]'
speculators train-cached-dflash2 --help
```

下面命令均在仓库根目录的同一个 Bash 会话中运行。所有 `/personal/...` 都是需替换的内网路径；为本轮选择新的 `EXPERIMENT_ROOT`，不要覆盖已有配置。选择已授权 GPU 的本机编号或 UUID：

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export DFLASH2_INITIAL=/personal/models/dflash2-initial
export DFLASH2_SHARED=/personal/business-cache/shared.pt
export BUSINESS_CACHE_MANIFEST=/personal/business-cache/manifest.json
export EXPERIMENT_ROOT=/personal/runs/business-dense-001
export CONFIG="$EXPERIMENT_ROOT/business-dense.yaml"
mkdir -p "$EXPERIMENT_ROOT/logs"
test ! -e "$CONFIG"
cat > "$CONFIG" <<'YAML'
dense_checkpoint: ${DFLASH2_INITIAL}
shared: ${DFLASH2_SHARED}
manifest: ${BUSINESS_CACHE_MANIFEST}
output: ${EXPERIMENT_ROOT}/dense-kl-3e-5
arm: dense
objective: kl
lr: 3e-5
epochs: 5
warmup_fraction: 0.2
batch_sequences: 32
context_token_budget: 48000
anchors: 64
eval_batch_sequences: 4
eval_anchors: 8
seed: 280
memory_limit_gib: 96
cache_target_precision: fp8
YAML
```

这是已用过的 dense 配方起点，不是所有业务/硬件的最佳参数。`memory_limit_gib: 96` 是 PyTorch allocator 上限，启动检查要求空闲显存至少 **112 GiB**；不是会自动占满卡，也不适用于任意显存大小。运行前根据实际长度/容量调整 batch、anchors、验证 batch 和显存上限，并重新 pilot。`cache_target_precision` 必须填写真实采集条件 `fp8` 或 `bf16`，不会转换权重或缓存。历史示例里的 `ALLAVA_CACHE_MANIFEST` 只是环境变量名；这里直接使用业务变量，没有数据集名称硬限制。

## 先检查缓存，再独立试跑

完整缓存检查在 CPU 读取数据，不启动 CUDA 训练，也不创建训练 output；**不要为此传 `--device cpu` 或 `--synthetic-smoke-test`**，前者只允许显式合成测试，后者不是业务兼容开关。

```bash
speculators train-cached-dflash2 --config "$CONFIG" --check-data \
  > "$EXPERIMENT_ROOT/logs/check-data.json"
cat "$EXPERIMENT_ROOT/logs/check-data.json"
```

正常返回 JSON，包含 `checked_records`、`train_images`、`planned_updates`、`cache_target_precision`。退出码必须为 0；步数由本次有效样本和 batch plan 决定，不能套用 ALLaVA 的 2815 步。

`--check-data` 核对 ID/split、路径存在、图像跨集合重复、token/prompt 长度一致、hidden 形状/dtype/有限值、可选缓存摘要及训练 batch plan；D-MTP 还核对 sidecar 身份和有效 anchor。它**不会加载初始 safetensors 或 shared.pt 来证明兼容，不会核验 target 来源、特征层语义、全部 token ID 范围或业务分组，也不会证明 GPU 可运行**。这些来源条件需要人工/采集程序审计，权重加载和真实计算由下一步验证。

检查通过且数据审计完成后，在独立 output 做最多 3 次更新，使用最多 8 条有效验证样本；不复用 pilot 权重启动正式训练：

```bash
if speculators train-cached-dflash2 --config "$CONFIG" \
  --output "$EXPERIMENT_ROOT/dense-pilot" \
  --pilot-steps 3 --pilot-val-images 8 \
  > "$EXPERIMENT_ROOT/logs/pilot.log" 2>&1; then
  echo 0 > "$EXPERIMENT_ROOT/logs/pilot-exit-code.txt"
else
  code=$?
  echo "$code" > "$EXPERIMENT_ROOT/logs/pilot-exit-code.txt"
  exit "$code"
fi
```

核对日志无异常、loss/梯度有限、`summary.json` 中 `pilot_completed=true`、`selector_unchanged=true`。Pilot 的 `completed=false` 是预期；`total_steps` 仍是正式配方计划值，实际 pilot 更新数看 `train.jsonl`。这一步不能代替完整训练或在线评测。

## 正式训练、恢复与验收

确认 pilot 正常结束后，用原配置和原初始化启动独立正式 output。下面是前台命令，长任务应放入内网已授权作业/会话；不要在没有 GPU 分配时自动提交：

```bash
if speculators train-cached-dflash2 --config "$CONFIG" \
  > "$EXPERIMENT_ROOT/logs/train.log" 2>&1; then
  echo 0 > "$EXPERIMENT_ROOT/logs/train-exit-code.txt"
else
  code=$?
  echo "$code" > "$EXPERIMENT_ROOT/logs/train-exit-code.txt"
  exit "$code"
fi
```

在另一个终端设置相同 `EXPERIMENT_ROOT` 后可查看 `logs/train.log`、`dense-kl-3e-5/progress.json` 和 `validation.json`。不要将日志写进尚未创建的训练 output，也不要事先 `mkdir` 该 output；训练器拒绝意外覆盖。

仅在正式进程已结束且需要恢复时，使用它保存的配置和相同数据/初始化/代码；下面命令不是正式训练结束后要自动执行的下一步：

```bash
speculators train-cached-dflash2 \
  --config "$EXPERIMENT_ROOT/dense-kl-3e-5/run.yaml" --resume
```

恢复点是最后完整 epoch 的 `resume.pt`，中途未保存部分会重跑；没有完整 epoch 的持久化点时可能从本轮起点重新执行。不要修改 provenance 来强行恢复。失败若发生在 `run.yaml` 保存前，应保留错误目录并选择新 output 重试。

正式结束需同时核对退出码 0、`summary.completed=true`、实际日志更新数与 `provenance.total_steps` 一致、五轮完整验证/导出及 `selector_unchanged=true`。主要产物：

- `train.jsonl`：每步 loss、学习率、梯度、实际已见样本数。
- `initial.json`、`validation.json`：起点及每轮缓存指标；报告 `teacher_selected.acceptance_rate`（乘 100 为百分比）和 `teacher_selected.accepted_length`（含 bonus）。
- `epochs/epoch-00…05/`：BF16 草稿权重与配置，不含共享 embedding/head；`summary.best_epoch` 按缓存 AL 选出，可能仍是 epoch 0。
- `run.yaml`、`provenance.json`、`train_command.txt`、`speculators.patch`：保留在内网的复现依据。

本路径的训练 loss 是 full-vocab KL，不包含冻结 selector 的 CE；这里的缓存验证不是原生 `Trainer.val_epoch` 的 total val loss。每图 8 个验证 anchor 也是缓存代理指标，不能当作整段在线接受率。最终应在同一批业务验证输入上比较本轮 epoch 0 与各 checkpoint，再用独立 test 复核；target 精度、模板、并发、输出长度和 runtime 条件保持一致。在线输出等价仍有已知未解决问题，见 [VALIDATION.md](VALIDATION.md)，不能因接受率上升就报告无损加速。

## 需要 D-MTP 时的额外输入

以上命令默认 dense。Token/fusion 除同一 dense 初始化、shared 和 target cache 外，每条 manifest 还需要 `mtp_path`（可选 `mtp_sha256`）。MTP `.pt` 必须有与 target 相同的 `ids`、`image_sha256`，并包含：

| 字段                | 形状 / 值                                                       |
| ------------------- | --------------------------------------------------------------- |
| `mtp_sample_hidden` | `[L, H]` BF16，消费 pending A 后、消费 T1 前的单流 LM-head 输入 |
| `mtp_token`         | `[L]` int64，相应首位候选                                       |
| `mtp_valid`         | `[L]` bool；有效样本的被检查/采样 anchor 必须为真               |
| `history`           | `native_causal_target_aligned`                                  |
| `alignment`         | `pending_A_at_a__target_HC_at_a-1__predict_T1_at_a+1`           |

不能用 target hidden 代替 native MTP hidden，也不能只加上述字符串伪造时序。原生 MTP 冻结；后段训练不会提高固定前缀上的原生 T1 能力。

用独立配置/output 将 `arm` 改为 `token` 或 `fusion`；fusion 设置 `fusion_init: residual-zero`。每组都重新做本指南的缓存检查、pilot、正式运行，使用相同初始化/数据/更新预算。不要让三组写同一目录，也不要把 dense 业务 epoch 5 当作其他组起点而仍称等预算对照。新包 D-MTP 的 GPU 迁移回归仍需验证，不能把 dense 已运行视为该路径已验收。
