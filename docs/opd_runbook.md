# 双机多模态 OPD 运行手册

本阶段把当前 `Base + GRPO R1 adapter` 作为学生起点，用同系列
`Qwen3-VL-32B-Instruct` 作为候选教师。目标是做可审计的多模态 on-policy
distillation，而不是把教师完整回答当作新的 SFT 固定答案。

## 算力原则

| 阶段 | 96GB 教师机 | 48GB 学生机 | 是否需要 GPU |
|---|---|---|---|
| 代码、数据与模型文件下载 | 开无卡模式 | 关机 | 否 |
| 模型分片、tokenizer、processor 审计 | 开无卡模式 | 关机 | 否 |
| 教师资格推理 | 开 GPU | 关机 | 是，仅 96GB |
| 学生资格基线与 round-1 rollout | 关机 | 开 GPU | 是，仅 48GB |
| 教师 top-k score | 开 GPU | 关机 | 是，仅 96GB |
| 学生 KL 更新与开发集评测 | 关机 | 开 GPU | 是，仅 48GB |
| paired bootstrap 与晋级判断 | 任一台无卡模式 | 关机 | 否 |

两张 GPU 不同时空挂。每个 GPU 阶段完成并确认退出码、manifest 与 SHA256 后立刻关机或切换无卡模式。

## 冻结边界

- 训练 prompt 只来自未参与 SFT/GRPO 的 train 记录；round 1 为 3,000 个 prompt，其中 20% hard prompt 生成两个 rollout，共 3,600 条。
- val 被一次性按图片组拆成教师资格 256 条和 OPD development 256 条；同一图片或同内容哈希图片不能跨面板。
- internal final test、ChartQA、ChartQAPro、MME-Finance 不参与教师选择、prompt 筛选、训练或 round 晋级。
- round 1 更新后必须重新生成 rollout 才能称为 round 2；旧 score 不可重复使用。

## 执行顺序

1. 无卡：两台主机对齐同一 Git commit，构建并审计 OPD 数据；教师模型下载完成后运行模型快照审计和 4B/32B 接口兼容性检查。
2. 48GB GPU：在 `teacher_qualification_256.jsonl` 上评测冻结的 GRPO-4B 学生。
3. 96GB GPU：在完全相同的 256 条上评测 32B 教师；关闭 GPU。
4. 无卡：运行教师 paired qualification。只有 `OPD_TEACHER_QUALIFICATION_PASS` 才允许继续。
5. 48GB GPU：从冻结 GRPO adapter 对 round-1 prompt 生成 3,600 条图像感知 rollout；保存真实 prompt/completion token IDs、图片哈希与生成种子，然后关闭 GPU。
6. 把完整 `outputs/opd/round1/student_rollouts/` 目录传到教师机；96GB GPU 对相同学生前缀生成 top-16 teacher log-prob 分片。每个 shard 都有 SHA256 与 top-k mass 统计；完成后关闭 GPU。
7. 把完整 `outputs/opd/round1/teacher_scores/` 目录传回学生机。学生端允许 manifest 中保留教师机绝对路径，但只会在原路径缺失时回退到 manifest 同目录的同名 shard，并始终重验 SHA256。
8. 48GB GPU：从原 GRPO adapter 做一轮稀疏 forward-KL 更新，3,600 micro-steps、梯度累积 8、450 optimizer steps，保存可恢复 checkpoint 和 final adapter。
9. 48GB GPU：在冻结的 `opd_development_256.jsonl` 上分别评测原 GRPO 基线和 round-1 候选；完成后关闭 GPU。
10. 无卡：运行 paired development gate。通过仅表示 round-1 可晋升为最终候选并允许人工评估 round-2 成本；代码不会自动启动 round 2。失败则保留原 GRPO R1 为最终模型。

## 下载完成后的第一个无卡门

教师模型下载出现 `27/27` 和 exit code 0 后，先让 96GB 实例保持**无卡模式**。在克隆实例中执行以下命令；它们只读取磁盘、计算 SHA256 和 safetensors header，不加载 CUDA：

```bash
set -euo pipefail
cd /root/autodl-tmp/EconoChart
git pull --ff-only origin main

export PYTHONPATH="$PWD/src"
ECONOCHART_PY=/root/miniconda3/envs/econochart/bin/python
TEACHER_PY=/root/autodl-tmp/envs/econochart-teacher-cu130/bin/python

"$ECONOCHART_PY" -m econochart.data.opd \
  --config configs/data/opd_v1.yaml

STUDENT_ADAPTER=outputs/grpo_qlora_48g_domain_v1/final_adapter/adapter_model.safetensors
EXPECTED_ADAPTER_SHA256=d47c083114e26000c1df5bd52676a1ca9dc2ca708a70c7944c97b3824404486b
test -s "$STUDENT_ADAPTER"
test "$(sha256sum "$STUDENT_ADAPTER" | cut -d' ' -f1)" = "$EXPECTED_ADAPTER_SHA256"
echo "OPD_STUDENT_ADAPTER_IDENTITY_PASS"

mkdir -p outputs/opd/model_audits
"$TEACHER_PY" -m econochart.distillation.model_audit \
  --model-dir /root/autodl-tmp/models/Qwen3-VL-32B-Instruct \
  --output outputs/opd/model_audits/qwen3vl32b_snapshot.json \
  --expected-shards 14 \
  --minimum-weight-gib 55 \
  --expected-model-type qwen3_vl \
  --validate-safetensors \
  --hash-shards

"$TEACHER_PY" -m econochart.distillation.model_audit \
  --model-dir /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
  --output outputs/opd/model_audits/qwen3vl4b_snapshot.json \
  --expected-shards 2 \
  --minimum-weight-gib 7 \
  --expected-model-type qwen3_vl \
  --validate-safetensors \
  --hash-shards

"$TEACHER_PY" -m econochart.distillation.compatibility \
  --student-model-dir /root/autodl-tmp/models/Qwen3-VL-4B-Instruct \
  --teacher-model-dir /root/autodl-tmp/models/Qwen3-VL-32B-Instruct \
  --output outputs/opd/model_audits/qwen3vl4b_to_qwen3vl32b_compatibility.json
```

该门不使用 `--force`：如果同名工件已存在，应先审计旧工件，而不是静默覆盖。只有数据 manifest 的泄漏计数全部为 0、两份模型报告均为 `OPD_MODEL_SNAPSHOT_AUDIT_PASS`、接口报告为 `OPD_MODEL_INTERFACE_COMPATIBILITY_PASS`，才安排第一次 GPU 资格推理。任一步非零退出时立即停止并保留原输出；不要打开另一台 GPU 补跑。

## 强制审计点

- processor 产生的视觉 token 数必须大于零；否则立即阻断，不能把纯文本运行冒充多模态 OPD。
- 4B 与 32B 的 vocab、视觉 token、chat template 和 tokenizer 语义指纹必须一致。
- 教师必须按学生实际访问的 completion prefix 评分；固定参考答案不进入 KL 目标。
- prompt、completion、图片、adapter、score shard 与运行配置均保存身份哈希；训练集、资格集、开发集和禁止使用的测试集同时检查 ID、图片路径与图片内容哈希重叠。
- round-1 development 阈值在预测产生前冻结：overall 至少 `+0.01` 且 `P(Δ>0)≥0.90`；numeric recall 至少 `+0.05` 且 `P(Δ>0)≥0.90`；同时执行 numeric precision、trend、evidence、risk、format 护栏。
- 所有阈值都按实际结果如实报告，不允许事后改门或改数字。
