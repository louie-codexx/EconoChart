# 实验记录规范

`outputs/` 保存机器产物，默认不进 Git；`experiments/` 保存人工审核后的轻量证据，用于复现、复盘和面试展示。

## 每次实验流程

1. 从 `experiment_matrix.yaml` 选择一个能改变决策的实验；
2. 复制 `templates/experiment_record.md`，命名为 `records/YYYYMMDD_<experiment_id>.md`；
3. 训练前填写假设、唯一变量、控制项、数据哈希和接受/停止条件；
4. 训练后填写真实环境、显存、时长、loss/reward、内部/外部/切片指标；
5. 用 `econochart-compare` 做同样本配对比较；
6. 写成功和失败案例、异常、结论及下一步；
7. 将小型聚合 JSON/PNG 放入 `results/`，不要复制 checkpoint 或完整 predictions。

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

- 完整实验记录 Markdown；
- 展开后的关键配置差异（避免泄露绝对路径）；
- 聚合 metrics、配对比较 JSON；
- loss/reward/显存/吞吐图；
- 少量脱敏成功与失败案例；
- 对应 Git commit 和数据 manifest hash。

## 不可提交内容

- 基础模型、adapter、checkpoint、optimizer state；
- 完整训练数据或公开数据缓存；
- 完整 predictions 和原始 TensorBoard/W&B 目录；
- `.env`、token、SSH key、AutoDL 私有路径。
