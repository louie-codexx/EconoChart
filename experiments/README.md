# 实验记录规范

`outputs/` 保存机器产物，默认不进 Git。`experiments/records/` 保存完整的私有复盘记录，同样不进 Git；GitHub 只保留实验协议、空白模板、实验矩阵，以及人工筛选和脱敏后的结果摘要。

## 公开与私有边界

| 位置 | 用途 | GitHub |
|---|---|---|
| `records/` | 完整参数推敲、失败过程、成本、机器路径、原始命令与面试复盘 | 不上传（`.gitignore`） |
| `results/` | 经过审核的聚合指标、关键消融表、少量脱敏案例与最终结论 | 可选择上传 |
| `templates/` | 展示项目的实验方法和证据标准 | 上传 |
| `outputs/` | checkpoint、原始日志、完整预测、TensorBoard 等机器产物 | 不上传 |

“不上传详细记录”不等于 GitHub 只放代码。公开仓库仍应给出足以证明项目有效的摘要指标、关键对照实验和可复现配置，但不必暴露完整内部推理和逐次复盘。

## 记录粒度

人工记录按里程碑创建，而不是按每天或每条命令创建：

- 环境与算力方案定版；
- 数据版本冻结及泄漏检查；
- Base 基线；
- 每个真正用于比较的 SFT、消融或 GRPO 实验；
- 最终评测与结论。

应该记录：LoRA rank/alpha/target modules、量化与精度、像素上限、有效 batch、学习率、数据配比、split/seed、GRPO generations/beta/reward 权重、峰值显存，以及 OOM、NaN、数据泄漏、reward hacking 等改变决策的故障。

不必人工记录：`cd`、`git pull`、每次成功 import、CLI 路径、普通安装输出、重复的 `pip check` 和没有改变决策的 smoke 细节。这些只保留在被忽略的 `outputs/` 或机器快照中，需要时再查。

## 准确性规则

- 明确区分 `已审计事实`、`运行中快照`、`解释性推断` 和 `待验证事项`，不得把四者混写成结论。
- 正式训练完成后必须记录开始/结束时间、Git commit、配置与数据身份、优化器与学习率调度、有效 batch、训练/验证历史、最终 adapter 哈希和原始证据路径；teacher-forced loss 与固定测试能力指标必须分栏记录。
- 若运行时 Git 工作树不干净，必须明确写 `git_dirty=true`、列出与运行有关的未提交文件、保存其 SHA-256，并注明 commit 本身不能完整复现实验；远端 `resolved_config` 或等价运行快照是最终参数证据。
- 只有同时具备退出码、完整行数/唯一 ID、manifest 身份和指标重算证据，正式评测才可标记为 `completed`。
- `smoke` 只证明链路兼容、参数发生更新且工件可重载；无论 reward 或 loss 如何，都不能写成能力提升。
- 运行中实验必须记录最后观察时间和进度分母；单次 GPU 利用率、ETA、局部 reward 或中途 loss 只能作为快照，不能外推最终结果。
- 所有修复都要记录失败原因、修复内容、是否重新生成数据/预测，以及修复是否改变实验可比性。不得删除失败尝试。
- 公开摘要必须能回指私有复盘与机器产物；摘要只复制审计后的聚合值、哈希和结论，不复制未经核验的终端片段。

## 每次实验流程

1. 从 `experiment_matrix.yaml` 选择一个能改变决策的实验；
2. 复制 `templates/experiment_record.md`，命名为 `records/YYYYMMDD_<experiment_id>.md`；该目录是私有区；
3. 训练前填写假设、唯一变量、控制项、数据哈希和接受/停止条件；
4. 只记录会影响模型效果、显存/成本、实验可比性、数据可信度，或导致实验方向改变的关键参数与故障；不要把日常命令流水账写入人工记录；
5. 训练后填写真实环境、显存、时长、loss/reward、内部/外部/切片指标；
6. 用 `econochart-compare` 做同样本配对比较；
7. 写成功和失败案例、异常、结论及下一步；
8. 实验结论稳定后，从私有记录中提炼一份脱敏、可验证的公开摘要，将小型聚合 JSON/PNG 放入 `results/`；不要复制 checkpoint、完整 predictions 或完整复盘记录。

## 命名和状态

- `planned`：只有假设；
- `code_ready`：配置和入口已通过 CPU 检查，尚未在目标 GPU 运行；
- `running`：AutoDL 正在执行；
- `failed`：运行失败，必须记录原因和已尝试修正；
- `completed`：产物、指标、案例和结论齐全；
- `rejected`：实验完成但假设不被支持；
- `stopped`：根据预设停止条件主动结束。

不要把失败实验删掉，也不要把 `code_ready` 写成“实验已完成”。

## 可提交内容

- 人工提炼后的实验摘要，而不是完整实验日记；
- 为结论所必需的关键配置差异（移除绝对路径和机器信息）；
- 聚合 metrics、配对比较 JSON；
- loss/reward/显存/吞吐图；
- 少量脱敏成功与失败案例；
- 对应 Git commit 和数据 manifest hash。

## 不可提交内容

- 基础模型、adapter、checkpoint、optimizer state；
- 完整训练数据或公开数据缓存；
- 完整 predictions 和原始 TensorBoard/W&B 目录；
- `records/` 下的详细实验日记、逐次排障命令、成本明细和内部面试复盘；
- `.env`、token、SSH key、AutoDL 私有路径。

## 项目最终摘要

项目于 2026-09-01 收束，交付 checkpoint 为 `Base → Domain SFT → GRPO R1` 的最终 adapter。公开结果目录额外保留：

- `20260901_mmefinance_pair_summary.json`：Base→mixed-SFT 的 1,171 条开放式金融配对诊断、评分边界与工件哈希；
- `20260901_final_model_decision.json`：最终模型身份、真实 attribution、未晋升候选和未执行 future work；
- `docs/final_model_card.md`：面向部署与面试的模型卡。

最终选择不会追溯改写历史实验决策。mixed-SFT、F2 numeric reward、300-step recall-repair GRPO 和 projector 解冻均是优化建议；除 mixed-SFT 本身外，后续方案没有运行，不能写成实验结果。
