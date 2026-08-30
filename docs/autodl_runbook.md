# AutoDL 分阶段运行手册

本手册的目标是让每次昂贵 GPU 运行都有前置检查、明确产物和停止条件。命令默认在 AutoDL 的 EconoChart 仓库根目录执行。

## 0. 同步代码，但不覆盖实验产物

```bash
git status --short
git pull --ff-only origin main
```

如果 `git status` 显示你在 AutoDL 修改过跟踪文件，先处理这些改动，不要用 destructive reset。完整数据和 `outputs/` 被 Git 忽略，正常 `git pull` 不会删除它们。

## 1. 确认模型与当前环境

```bash
python -m pip install -e .
export ECONOCHART_MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct

# 旧 v1 数据被 Git 忽略，本地删除不会随 pull 传播；先预览精确目标，再确认清理。
econochart-clean-legacy
econochart-clean-legacy --apply

econochart-preflight --stage sft \
  --config configs/train/sft_qlora_smoke.yaml \
  --report outputs/preflight/sft_inventory.json
```

清理命令只处理审计过的 `business_data.json`、旧 SFT/DPO JSON 和旧 charts 目录，不触碰 v2、公开数据、模型或训练产物。

第一次报告可能因旧 TRL/PEFT 或未生成数据而失败；它的作用是盘点，不代表代码故障。确认后安装最终依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
python -m pytest
python -m ruff check .
```

不要把 Hugging Face token 写入仓库或命令历史。需要下载公开数据时使用 AutoDL 自己的安全凭据配置。

## 2. 构建并冻结 EconoChart-v2

```bash
econochart-preflight --stage data \
  --config configs/data/econochart_v2.yaml \
  --report outputs/preflight/data.json

econochart-build --config configs/data/econochart_v2.yaml
econochart-validate \
  --dataset-root data/generated/econochart_v2 \
  --full-image-scan

econochart-build-subsets \
  --config configs/data/training_subsets.yaml
```

验收：

- validator 为 `passed`，critical/high issue 均为 0；
- 企业、chart、image 跨 split 交集为 0；
- 目标规模约 3,000 企业、6,000 图、24,000 问答；
- `manifest.json`、split entity 清单和 annotation 哈希已生成；
- `subsets/subset_manifest.json` 的源 manifest 哈希匹配，SFT screen/final、GRPO、val 分别为 4,800/9,600/3,600/512 条；
- SFT final 覆盖全部 2,400 家 train 企业和 4,800 张图，GRPO 覆盖全部 train 企业，development 子集没有 test 行；
- 人工抽查至少 50 张图，确认标签、刻度、图例和单位可读且无 scenario 泄漏。

首次构建后不要在同一实验系列中重新生成 test。如果要更换生成逻辑，必须提高数据版本并重跑 base。

## 3. 准备公开数据（外部泛化阶段需要）

```bash
econochart-prepare-public \
  --config configs/data/public_datasets.yaml \
  --dataset chartqa

econochart-prepare-public \
  --config configs/data/public_datasets.yaml \
  --dataset chartqapro
```

ChartQA train 只有在“公开数据混合消融”中使用。ChartQA test 与全部 ChartQAPro 永远不进入训练、早停或超参数选择。

公开数据命令退出 0 只是第一层门禁，进入 GPU 外部评测前还必须独立读取 manifest 和 annotation 验收：

- ChartQA：固定 test 行数不变；validation 先删除与 test 同图的记录，train 再删除与清理后 validation/test 同图的记录，均不回填；三个 split 的规范化图片 SHA-256 交集最终为 0，删除 ID 与未引用图片清单可对应。
- ChartQAPro：核对 selected source 与 evaluable 行数、空最终 `Answer[-1]` 的排除 ID、Year 长度异常记录、annotation/image 一一对应及 hash；空标签不伪造、不回填，Year 元数据不按问答轮数截断。只有这些审计项全部通过，才可把固定可评分子集用于 Base/SFT/GRPO 外部评测。

任一门失败时保留日志和失败目录，修复代码后使用新的 attempt 日志重建；不要把下载完成、缓存回退或 CPU 单测通过写成公开数据门已通过。

## 4. 保存 base 基线

```bash
econochart-preflight --stage eval \
  --config configs/eval/base_internal.yaml \
  --report outputs/preflight/base_internal.json

econochart-eval --config configs/eval/base_internal.yaml
econochart-eval --config configs/eval/base_external.yaml
```

若评测中断，只能在模型、adapter、数据顺序和 generation 配置完全相同时显式续跑：

```bash
econochart-eval --config configs/eval/base_internal.yaml --resume
```

续跑会校验原始 run manifest、样本 ID 和 reference 字段；不匹配时直接拒绝，避免把不同模型的预测混在一起。

必须在 SFT 前完成，因为后续要做严格配对比较。保存：metrics、predictions、run manifest、峰值显存、总用时和至少 20 个错误案例。

## 5. SFT smoke

```bash
econochart-preflight --stage sft \
  --config configs/train/sft_qlora_smoke.yaml \
  --report outputs/preflight/sft_smoke.json

econochart-sft --config configs/train/sft_qlora_smoke.yaml
```

smoke 只回答工程问题，不用于报告模型效果：

- Qwen3-VL processor 能否读取最终图片/messages；
- NF4、PEFT target modules、completion-only loss 是否兼容；
- 两个 step 是否 loss 有限且能反向传播；
- checkpoint/adapter/processor 是否能保存并重新加载；
- 峰值显存是否为正式训练留下合理余量。

任何一项失败都先修 smoke，不能直接启动全量训练。

## 6. 参数筛选、主 SFT 与断点续训

所有候选都使用固定的 4,800 条嵌套筛选集、1 epoch、同一 seed、有效 batch、图像像素范围、LoRA target modules、优化器和调度器。完整 test 不参与选择。

先在 512 条 val 面板上生成一次未微调 Base 对照。它只用于开发阶段配对比较，不替代已经冻结的 2,496 条正式 Base：

```bash
econochart-eval \
  --config configs/eval/development_val_512.yaml \
  --output-dir outputs/evaluation/base_val512_v1
```

第一阶段固定 `r=16, alpha=32`，只比较三个对数尺度相邻的学习率。`r=16` 是已经通过 4090 smoke 的中间容量锚点，不是预先认定的最优值：

```bash
econochart-sft --config configs/train/sft_qlora_lr5e5_ablation.yaml
econochart-sft --config configs/train/sft_qlora_r16_screen.yaml
econochart-sft --config configs/train/sft_qlora_lr2e4_ablation.yaml
```

每个候选都用同一个 512 条 val 面板做确定性生成评测。下面以 `r16/lr1e-4` 为例，其他候选替换 adapter 和输出目录：

```bash
econochart-preflight --stage eval \
  --config configs/eval/development_val_512.yaml \
  --adapter outputs/screen/sft_qlora_r16_lr1e4/final_adapter \
  --report outputs/preflight/sft_r16_lr1e4_val512.json

econochart-eval \
  --config configs/eval/development_val_512.yaml \
  --adapter outputs/screen/sft_qlora_r16_lr1e4/final_adapter \
  --output-dir outputs/evaluation/screen_r16_lr1e4_val512
```

第二阶段固定第一阶段胜出的学习率，只比较 `r=8` 与 `r=16`。按胜出学习率选择且只运行对应的 `r=8` 配置；`r=16` 结果直接复用第一阶段结果：

```bash
# lr=5e-5 胜出时
econochart-sft --config configs/train/sft_qlora_r8_lr5e5_screen.yaml

# lr=1e-4 胜出时
econochart-sft --config configs/train/sft_qlora_r8_ablation.yaml

# lr=2e-4 胜出时
econochart-sft --config configs/train/sft_qlora_r8_lr2e4_screen.yaml
```

选择以 corrected overall 为主，联合 numeric recall、risk、trend、evidence、困难样本切片、token-cap 命中率、峰值显存、吞吐和稳定性；teacher-forced eval loss 只作辅助。候选之间使用固定 ID 的 paired bootstrap；若质量差异没有充分证据，学习率保留中心值 `1e-4`，rank 选择更小的 `r=8`。`r=32` 只有在 `r=16` 明显优于 `r=8` 且仍存在容量不足证据时才运行。此时不要查看完整 test 或外部测试结果。

若胜出者不是中心候选，先把选择结果及理由写入私有记录，并把正式配置的 rank/alpha/LR 更新为胜出值。然后运行 9,600 条、2 epochs 的主 SFT：

```bash
econochart-sft --config configs/train/sft_qlora_4090.yaml
```

断点续训：

```bash
econochart-sft \
  --config configs/train/sft_qlora_4090.yaml \
  --resume
```

或指定 checkpoint 路径：

```bash
econochart-sft \
  --config configs/train/sft_qlora_4090.yaml \
  --resume outputs/sft_qlora_r16_domain_v1/checkpoint-250
```

另开终端查看：

```bash
nvidia-smi
tensorboard --logdir outputs/sft_qlora_r16_domain_v1/runs --host 0.0.0.0
```

若实际 TensorBoard 子目录由 Trainer 版本生成在其他位置，以 `resolved_config.yaml` 和输出目录为准。

## 7. SFT 评测与准入判断

```bash
export ECONOCHART_ADAPTER_PATH=outputs/sft_qlora_r16_domain_v1/final_adapter

econochart-eval \
  --config configs/eval/base_internal.yaml \
  --adapter "$ECONOCHART_ADAPTER_PATH" \
  --output-dir outputs/evaluation/sft_internal_v1

econochart-eval \
  --config configs/eval/base_external.yaml \
  --adapter "$ECONOCHART_ADAPTER_PATH" \
  --output-dir outputs/evaluation/sft_external_v1

econochart-compare \
  --baseline outputs/evaluation/base_internal_v1/predictions.jsonl \
  --candidate outputs/evaluation/sft_internal_v1/predictions.jsonl \
  --output outputs/evaluation/base_vs_sft_internal.json
```

进入 GRPO 前应满足：

- numeric/trend/overall 的配对差值方向符合假设，且不是只由单一模板切片驱动；
- format/evidence/risk 没有不可接受回退；
- 外部 ChartQA/ChartQAPro 不出现未解释的大幅退化；
- 人工错误分析确认主要剩余问题确实可由现有 reward 约束；
- adapter 可在独立进程加载推理。

若不满足，先做 SFT 数据、rank、LR 或公开数据混合消融，不能用 GRPO 掩盖不稳定 SFT。

## 8. 条件 SFT 消融

每次只改变一个主要因素，并从相同 base、相同数据版本、相同 seed 和相同测试集出发：

默认筛选只运行第 6 节的四个候选。只有 r=16 显示容量不足时才跑 r=32；只有量化误差成为可信解释时才跑 BF16 LoRA；只有正式 SFT 外部 ChartQA 明显退化时才准备并运行 ChartQA 混合。不要为了矩阵完整而运行没有决策价值的组合。

```bash
# 条件执行，不是默认清单
econochart-sft --config configs/train/sft_qlora_r32_ablation.yaml
econochart-sft --config configs/train/sft_lora_4090_ablation.yaml
econochart-sft --config configs/train/sft_qlora_4090_mixed_chartqa.yaml
```

## 9. GRPO smoke

确认环境变量仍指向通过门槛的 SFT adapter：

```bash
econochart-preflight --stage grpo \
  --config configs/train/grpo_qlora_4090_smoke.yaml \
  --report outputs/preflight/grpo_smoke.json

econochart-grpo --config configs/train/grpo_qlora_4090_smoke.yaml
```

检查 reward 分量是否有方差。若同一 prompt 的所有 generation 得分完全相同，group advantage 为零，该批次无法学习。抽查输出，确认模型不是只堆章节、复述数字或拉长文本来刷 reward。

## 10. 正式 GRPO 与最终评测

48GB 配置不是强制要求，而是当前正式实验起点。数据预算固定为 3,600 prompts × 4 generations（14,400 rollouts），不能用随手截断替代覆盖率约束子集：

```bash
econochart-preflight --stage grpo \
  --config configs/train/grpo_qlora_48g.yaml \
  --report outputs/preflight/grpo_48g.json

econochart-grpo --config configs/train/grpo_qlora_48g.yaml
```

完成后以 GRPO `final_adapter` 重跑内部和外部评测，并分别比较 SFT→GRPO、base→GRPO。只有可验证能力改善且 guardrail 未退化，才把 RL 结论写进简历。

## 11. OOM 调整顺序

一次只改一项并记录：

1. SFT `max_pixels`: 1024 → 768 → 512 visual tokens；
2. GRPO `max_completion_length`: 256 → 192 → 128；
3. GRPO generations: 4 → 2，同时保持有效 batch 可整除；
4. 正式 GRPO `beta`: 0.001 → 0，记录失去 KL reference 的取舍；
5. 减少 eval 样本或降低 eval 频率，但不能更换最终测试集；
6. 在版本匹配且 smoke 通过后尝试 FlashAttention 2；
7. 升级 48GB/80GB GPU。

`per_device_train_batch_size` 已为 1。盲目降低 `gradient_accumulation_steps` 会改变有效 batch，并可能破坏 `num_generations` 整除约束。

## 12. 每次运行结束

1. 复制 `experiments/templates/experiment_record.md`；
2. 填写运行 manifest、配置 diff、显存、吞吐、loss/reward、全体与切片指标；
3. 保存成功/失败案例和异常日志摘要；
4. 写出结论、局限、是否继续与下一实验；
5. 只把人工整理后的轻量记录和图表提交到 Git，不提交 outputs/checkpoint。
