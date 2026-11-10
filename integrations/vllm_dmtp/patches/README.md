# 完整 vLLM D-MTP patch

[`vllm-dmtp-bbf58932.patch`](vllm-dmtp-bbf58932.patch) 是可用 `git apply` 应用的完整补丁，包含 **14 个修改文件和 8 个新增文件**。应用后的全部 22 个文件与本仓库 [overlay](../overlay) 以及 2026-11-10 实跑使用的版本逐字节一致；本次只补交付形式，没有更改推理算法。

**基线必须是原私有 Qwen-capable vLLM fork 的 `bbf5893213eff1e1dc7bbe76ce410b31eaca1507`。** 这不是面向任意上游 vLLM 0.31.0 或 pip 安装包的通用补丁。目标模型基线代码、模型权重、配套 wheel 的编译/vendored 文件和 CUDA/PyTorch 环境需要操作者另行提供。

## 内容与适用范围

- 原生 MTP 提出 T1，同时向 DFlash2 后六位提供 token，以及 fusion 模式所需的 native hidden。
- Token-only / residual hidden fusion 的模型加载与注册、proposer 调度、query/context 时序、KV 与接受/拒绝后的状态处理。
- 配套配置检查、CUDA graph 和 DFlash2 基础接线；也包含原实跑树所需的早期 n-gram plumbing，但本次 D-MTP 实验未启用 n-gram。

没有把未验证的状态处理称为正确性保证。此次实跑 target-only 输出等价仍未通过，历史 D-MTP 跨 GPU 差异也尚未解决；详细结果见 [实验归档](../../../docs/cached_dflash2/results/allava_dmtp_20261110/README.md)。

## 应用与核对

使用自己的全新 checkout。补丁与 `scripts/vllm_dmtp_runtime.py --apply` 是两种替代安装方式，二选一；不要在已经覆盖 overlay 的工作树再应用补丁。

```bash
export SPECULATORS_ROOT=/personal/src/modelarts-speculators
export VLLM_SOURCE=/personal/src/original-private-vllm
export RUNTIME_ROOT=/personal/src/vllm-dmtp-patched
BASE=bbf5893213eff1e1dc7bbe76ce410b31eaca1507
PATCH_DIR="$SPECULATORS_ROOT/integrations/vllm_dmtp/patches"

# 校验下载的补丁与元数据；下面使用 Linux 的 sha256sum。
(cd "$PATCH_DIR" && sha256sum -c SHA256SUMS)

# 只创建独立工作树，不更改已有运行环境。
git -C "$VLLM_SOURCE" worktree add --detach "$RUNTIME_ROOT" "$BASE"
test "$(git -C "$RUNTIME_ROOT" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$RUNTIME_ROOT" status --porcelain)"
git -C "$RUNTIME_ROOT" apply --check "$PATCH_DIR/vllm-dmtp-bbf58932.patch"
git -C "$RUNTIME_ROOT" apply "$PATCH_DIR/vllm-dmtp-bbf58932.patch"

# 校验 base、全部 22 个文件 SHA256 以及是否有额外修改。
python "$SPECULATORS_ROOT/scripts/vllm_dmtp_runtime.py" \
  --runtime-root "$RUNTIME_ROOT" --check
```

校验器要求 HEAD 仍为该 base，修改保留在工作树中；不要在核对前随意提交成另一 SHA。随后按 [vLLM 推理说明](../../../docs/cached_dflash2/VLLM_INFERENCE.md) 在配套私有环境中完成 editable 安装/编译，确认实际 import 路径，再进行小图加载、输出比较及容量验证。单独应用 patch 不会安装编译扩展、下载目标权重或启动推理。

训练导出的 token/fusion checkpoint 分别使用 `--mode token` / `--mode fusion`，由仓库启动器配置 `method=mtp`、七候选，以及 `additional_config.swx_dmtp.model`。普通上游的 `method=mtp` 本身不会自动启用这里的 DFlash2 后段。

## 补丁验证记录

已在本机从真实 base 创建独立工作树，完成：

1. `git apply --check` 成功，完整 patch 实际应用成功。
2. 应用后 22 个文件的 SHA256 全部与 `source-provenance.json` 相同，已有 runtime checker 返回 `verified_overlay_files=22`。
3. `git apply --reverse --check` 与实际反向应用成功，工作树恢复为干净基线。

[`manifest.json`](manifest.json) 保存 patch 摘要、每个文件的基线/应用后摘要及新增/修改分类；[`SHA256SUMS`](SHA256SUMS) 可验证交付文件。此次没有重跑 GPU；此前的实跑覆盖范围和输出等价限制继续适用。

实验结果目录内原有的 `online/*/vllm.patch` 是当时 `git diff` 生成的 provenance，只包括已跟踪文件的修改；新增文件由 overlay 另行保存。因此不要把那些历史日志补丁单独用作完整安装包。本目录补丁从真实基线和全部 overlay 文件重建，包含这些新增文件。
