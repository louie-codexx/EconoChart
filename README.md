# EconoChart

基于 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 的数字经济经营分析多模态后训练项目。

EconoChart 不把“跑一次 LoRA”当作项目结论，而是建立一条可审计的实验链：无泄漏数据 → base 基线 → LoRA/QLoRA SFT → 分能力评测与消融 → 可验证 GRPO → 失败复盘。目标能力包括图表理解、数值推理、经营风险诊断和证据约束的决策建议。

> 当前状态：数据、训练、奖励、评测、预检和 CPU 测试代码已实现；仓库内 8 家企业/16 张图/64 条问答样例通过完整校验。GPU 基线、SFT 与 GRPO 结果必须在 AutoDL 实际运行后填写，仓库不会预先声称未经验证的提升。

## 为什么这个项目不是普通微调 Demo

- 企业级切分：先按企业划分 train/val/test，再派生多视图和多问题，避免同图或同企业泄漏。
- 可验证监督：回答使用【结论】【数据依据】【风险】【建议】，不监督不可审计的自由式隐藏思维链。
- 业务一致性：收入由用户与 ARPU 推导，利润满足收入减成本，并保留利润率、转化率、CAC、产品和区域结构。
- 无标签捷径：图表不显示经营 scenario，固定尺寸渲染，财务指标和用户指标不再错误共轴。
- 统一对比：base、SFT、GRPO 使用相同测试记录、确定性解码和同一套分能力指标。
- 真正的 RL 问题：GRPO 只训练存在确定 ground truth 的数值、趋势、证据和风险任务，reward 可离线单测。
- 可归因实验：LoRA rank、学习率、量化方式、公开数据混合和 reward 组合均有独立假设与消融计划。

## 实验链路

```mermaid
flowchart LR
    A[确定性业务数据] --> B[无泄漏多视图图表]
    B --> C[数据质检与固定测试集]
    C --> D[Base 内部/外部基线]
    D --> E[QLoRA/LoRA SFT]
    E --> F[成对评测与消融]
    F -->|通过阶段门槛| G[可验证 GRPO]
    G --> H[最终评测与实验复盘]
```

LoRA/QLoRA 是 SFT 的参数更新方式，不是“先全参数 SFT、再 LoRA”的两个必经阶段。单卡 RTX 4090 24GB 默认使用 QLoRA；BF16 LoRA 是方法消融，全参数训练只保留接口，不作为 4090 默认方案。

## 数据体系

| 数据 | 用途 | 是否训练 | 仓库是否保存全量 |
|---|---|---:|---:|
| EconoChart-v2 | 数字经济经营分析主数据 | 是 | 否，可确定性重建 |
| ChartQA train/val | 通用图表 QA 混合消融 | 可选 | 否，按官方来源准备 |
| ChartQA test | 外部分布基准 | 否 | 否 |
| ChartQAPro test | 高难真实图表挑战集 | 否 | 否 |
| `data/samples/econochart_v2` | 代码测试与 GitHub 展示 | 否 | 是 |

默认 v2 配置生成 3,000 家企业、6,000 张图、24,000 条任务对齐问答；实际数量、分布和校验哈希以每次构建生成的 `manifest.json` 为准。正式训练使用确定性、覆盖率约束的预算子集：4,800 条 SFT 参数筛选、9,600 条正式 SFT、3,600 条 GRPO prompt，以及仅来自 val 的 512 条开发面板；2,496 条内部 test 保持完整。公开数据的许可和用途见 [数据卡](docs/data_card.md)。

## AutoDL 最短可信闭环

以下命令在仓库根目录执行。基础模型不进入 Git，通过环境变量指向 AutoDL 已下载目录。

```bash
git pull origin main
python -m pip install -e ".[train,dev]"

export ECONOCHART_MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct

econochart-preflight --stage data --config configs/data/econochart_v2.yaml \
  --report outputs/preflight/data.json
econochart-build --config configs/data/econochart_v2.yaml
econochart-validate --dataset-root data/generated/econochart_v2 --full-image-scan
econochart-build-subsets --config configs/data/training_subsets.yaml

econochart-preflight --stage eval --config configs/eval/base_internal.yaml \
  --report outputs/preflight/base_eval.json
econochart-eval --config configs/eval/base_internal.yaml

econochart-preflight --stage sft --config configs/train/sft_qlora_smoke.yaml \
  --report outputs/preflight/sft_smoke.json
econochart-sft --config configs/train/sft_qlora_smoke.yaml

# 固定 4,800 条筛选集和 512 条 val 面板；只比较高价值候选。
econochart-sft --config configs/train/sft_qlora_r8_ablation.yaml
econochart-sft --config configs/train/sft_qlora_r16_screen.yaml
econochart-sft --config configs/train/sft_qlora_lr5e5_ablation.yaml
econochart-sft --config configs/train/sft_qlora_lr2e4_ablation.yaml

# 根据固定 val 面板选择参数后，再运行 9,600 条、2 epochs 的正式 SFT。
econochart-sft --config configs/train/sft_qlora_4090.yaml
export ECONOCHART_ADAPTER_PATH=outputs/sft_qlora_r16_domain_v1/final_adapter

econochart-eval --config configs/eval/base_internal.yaml \
  --adapter "$ECONOCHART_ADAPTER_PATH" \
  --output-dir outputs/evaluation/sft_internal_v1

econochart-preflight --stage grpo --config configs/train/grpo_qlora_4090_smoke.yaml \
  --report outputs/preflight/grpo_smoke.json
econochart-grpo --config configs/train/grpo_qlora_4090_smoke.yaml
```

不要从数据构建直接跳到 GRPO。必须先保存 base 预测、完成 SFT smoke、评测 SFT adapter，并确认数值/趋势提升没有以格式或外部泛化显著退化为代价。完整执行顺序、环境检查和 OOM 处理见 [AutoDL 手册](docs/autodl_runbook.md)。

## 公开基准（可选准备）

```bash
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqa
econochart-prepare-public --config configs/data/public_datasets.yaml --dataset chartqapro
econochart-eval --config configs/eval/base_external.yaml
```

ChartQAPro 只用于最终测试。评测会额外生成 `chartqapro_official_predictions.json`，可再交给官方脚本或 VLMEvalKit 复核。

## 公平比较

```bash
econochart-compare \
  --baseline outputs/evaluation/base_internal_v1/predictions.jsonl \
  --candidate outputs/evaluation/sft_internal_v1/predictions.jsonl \
  --output outputs/evaluation/base_vs_sft.json
```

比较器严格对齐样本 ID 和 reference 字段，并用成对 bootstrap 输出 95% 置信区间。总体均值之外，还按任务、视图、行业、难度和经营场景切片，回答“哪些能力真的改善、哪些没有”。

长时间评测会定期保存预测；同一模型与配置中断后可加 `--resume`。续跑会核对原 run manifest、确定性样本前缀和 reference，拒绝把不同运行结果拼接在一起。

## 4090 默认设计

| 参数 | 默认值 | 初始依据 |
|---|---:|---|
| SFT 方法 | 4-bit NF4 QLoRA | 24GB 显存下保留训练余量 |
| LoRA target | LLM attention + MLP projections | 默认冻结视觉塔和投影层，先隔离语言/推理适配效果 |
| rank / alpha | 16 / 32 | 容量与成本折中；另测 r=8、32 |
| 有效 batch | 16 | 单卡 batch=1、累积 16，稳定 adapter 更新 |
| SFT LR | `1e-4` | adapter 常用起点；另测 `5e-5`、`2e-4` |
| 参数筛选数据 | 4,800 × 1 epoch | 每张训练图保留 1 题，用固定 512 条 val 面板选参 |
| 正式 SFT 数据 | 9,600 × 2 epochs | 每张训练图保留 2 题，覆盖全部 2,400 家企业和 4,800 张图 |
| 视觉 token 上限 | 1024 | 图表可读性与显存的主要折中旋钮 |
| GRPO smoke | 2 generations、beta=0 | 4090 先验证完整链路，关闭 KL/reference-policy 路径以降低开销 |
| 正式 GRPO | 4 generations、beta=0.001 | 高显存配置再验证相对奖励与 KL 约束 |
| 正式 GRPO 数据 | 3,600 prompts × 4 | 覆盖全部训练企业，共 14,400 次 rollout |

这些值是实验起点，不是事后包装的“最优参数”。最终选择必须引用显存、吞吐、验证集和外部基准结果。详细理由与降级顺序见 [实验协议](docs/experiment_protocol.md)。

## 项目结构

```text
configs/                 数据、SFT、GRPO、评测与消融配置
data/
  manifests/             可跟踪的数据设计清单
  samples/               可跟踪的小型完整样例
docs/                    架构、数据卡、AutoDL、实验与面试材料
experiments/             实验矩阵、记录模板和聚合结果
src/econochart/
  data/                  生成、覆盖率约束子集、公开数据适配、schema、训练视图、质检
  models/                Qwen3-VL/量化/adapter 加载
  training/              SFT 与 GRPO
  rewards/               可单测 reward 组件
  evaluation/            推理、指标、成对比较
  preflight.py           AutoDL 环境与运行前检查
tests/                   不下载模型的 CPU 测试
```

## 测试

```bash
python -m pytest
python -m ruff check .
```

本地 CPU 测试不会证明 Qwen3-VL 与 TRL 在目标 GPU 上一定兼容，因此 SFT 和 GRPO 都有使用最终代码路径的 AutoDL smoke gate。

## GitHub 与训练产物边界

主仓库提交代码、配置、测试、文档、轻量样例、manifest 和聚合实验结论；不提交完整数据、基础模型、adapter/checkpoint、optimizer state、原始日志、预测大文件或 `.env`。最终 adapter 可发布到 Hugging Face/ModelScope/Release，并在实验记录中注明基座和许可。

详见 [数据目录说明](data/README.md) 与 [实验记录规范](experiments/README.md)。

## 文档导航

- [系统架构](docs/architecture.md)
- [数据卡与公开数据许可](docs/data_card.md)
- [AutoDL 分阶段运行手册](docs/autodl_runbook.md)
- [实验、消融、阶段门槛与停止条件](docs/experiment_protocol.md)
- [环境兼容与预检](docs/environment.md)
- [算法面试讲解与追问](docs/interview_guide.md)

## License

项目代码采用 Apache-2.0。基础模型与公开数据分别遵循其原始许可；本仓库不重新分发它们。
