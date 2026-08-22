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
