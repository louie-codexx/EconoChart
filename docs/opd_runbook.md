# 双机多模态 OPD 研究手册（已暂停）

> 2026-09-05 收尾状态：本轮项目以 GRPO R1 交付，OPD 状态为 `deferred`。v1 教师资格失败；v2 仅完成代码与本地测试，尚无提交的真机结果。本手册保留历史方案和未来复现步骤，后文命令均不属于当前执行计划；此前安排的 28 条 smoke 与 256 条复跑已取消。决定见 [收尾记录](../experiments/results/20260905_grpo_project_closeout.json)。

保留的方案把 `Base + GRPO R1 adapter` 作为学生起点，用同系列
`Qwen3-VL-32B-Instruct` 作为候选教师。目标是做可审计的多模态 on-policy
distillation，而不是把教师完整回答当作新的 SFT 固定答案。

## 算力原则

| 阶段 | 96GB 教师机 | 48GB 学生机 | 是否需要 GPU |
|---|---|---|---|
| 代码、数据与模型文件下载 | 开无卡模式 | 关机 | 否 |
| 模型分片、tokenizer、processor 审计 | 开无卡模式 | 关机 | 否 |
| 教师提示协议 smoke（28 条 train） | 开 GPU | 关机 | 是，仅 96GB，约一次短跑 |
| 教师资格推理 | 开 GPU | 关机 | 是，仅 96GB |
| 学生资格基线与 round-1 rollout | 关机 | 开 GPU | 是，仅 48GB |
| 教师 top-k score | 开 GPU | 关机 | 是，仅 96GB |
| 学生 KL 更新与开发集评测 | 关机 | 开 GPU | 是，仅 48GB |
| paired bootstrap 与晋级判断 | 任一台无卡模式 | 关机 | 否 |

两张 GPU 不同时空挂。每个 GPU 阶段完成并确认退出码、manifest 与 SHA256 后立刻关机或切换无卡模式。

## 冻结边界

- 训练 prompt 只来自未参与 SFT/GRPO 的 train 记录；round 1 为 3,000 个 prompt，其中 20% hard prompt 生成两个 rollout，共 3,600 条。任务配比先按目标权重分配；若某类冻结未见数据不足，则封顶于真实容量并把缺口按原权重确定性分配给仍有容量的类别，不复制、不过采样。manifest 同时记录目标配额、可用分布、有效配额和调整量。
- val 被一次性按图片组拆成教师资格 256 条和 OPD development 256 条；同一图片或同内容哈希图片不能跨面板。
- internal final test、ChartQA、ChartQAPro、MME-Finance 不参与教师选择、prompt 筛选、训练或 round 晋级。
- round 1 更新后必须重新生成 rollout 才能称为 round 2；旧 score 不可重复使用。

## 教师资格 v1 的真实失败与不可变边界

2026-09-03 的双机实跑已证明推理链路可用：GRPO-4B student 与
Qwen3-VL-32B teacher 都在冻结的 256 条 qualification panel 上自然结束，exit code
均为 0。但 student overall/numeric recall 为 `0.835343/0.628906`，teacher v1 仅为
`0.391326/0.277669`，所以 v1 按预注册资格门失败，不能启动 3,000 条 rollout。

CPU 诊断进一步确认 teacher v1 的四类严格章节标签命中均为 `0/256`。它使用了
Markdown 标题而非 `【结论】` 等机器协议；同时复杂任务经常枚举原始年度值，却没有
优先计算题目点名的累计变化、增速差、利润率百分点变化或 ARPU。因此 v1 不是单纯
评分器误伤，不能通过放宽 format 或移动资格阈值修复。v1 预测、metrics 和日志必须保留，
v2 使用独立目录；可提交摘要见
`experiments/results/20260903_opd_teacher_qualification_v1_summary.json`。

## 历史执行与未来重启顺序（当前不执行）

1. 无卡：两台主机对齐同一 Git commit，构建并审计 OPD 数据；教师模型下载完成后运行模型快照审计和 4B/32B 接口兼容性检查。（已完成）
2. 48GB GPU：在 `teacher_qualification_256.jsonl` 上评测冻结的 GRPO-4B student v1。（已完成，保留）
3. 96GB GPU：在完全相同的 256 条上评测 32B teacher v1。（已完成但资格失败，保留）
4. 96GB GPU：使用 `opd_teacher_protocol_v2`，只在 round-1 train prompt 的七类任务各 4 条上运行 28 条 smoke；不接触 qualification/development/test。
5. 无卡：运行 `OPD_TEACHER_PROMPT_SMOKE` 绝对门。只有 PASS 才允许唯一一次 v2 256 条资格复跑；失败只能继续使用 train smoke 修订提示词。
6. 96GB GPU：在原冻结 256 条上运行 teacher v2；runner 会在加载模型前硬校验 smoke PASS。完成后关闭 GPU。
7. 无卡：运行 v2 paired qualification；其阈值与 v1 完全相同，且会复核 smoke gate 与 candidate run manifest 中的 prompt profile。只有 `OPD_TEACHER_QUALIFICATION_PASS` 才允许继续。
8. 48GB GPU：从冻结 GRPO adapter 对 round-1 prompt 生成 3,600 条图像感知 rollout；保存真实 prompt/completion token IDs、图片哈希与生成种子，然后关闭 GPU。
9. 把完整 `outputs/opd/round1/student_rollouts/` 目录传到教师机；96GB GPU 对相同学生前缀生成 top-16 teacher log-prob 分片。每个 shard 都有 SHA256 与 top-k mass 统计；完成后关闭 GPU。
10. 把完整 `outputs/opd/round1/teacher_scores/` 目录传回学生机。学生端允许 manifest 中保留教师机绝对路径，但只会在原路径缺失时回退到 manifest 同目录的同名 shard，并始终重验 SHA256。
11. 48GB GPU：从原 GRPO adapter 做一轮稀疏 forward-KL 更新，3,600 micro-steps、梯度累积 8、450 optimizer steps，保存可恢复 checkpoint 和 final adapter。
12. 48GB GPU：在冻结的 `opd_development_256.jsonl` 上分别评测原 GRPO 基线和 round-1 候选；完成后关闭 GPU。
13. 无卡：运行 paired development gate。通过仅表示 round-1 可晋升为最终候选并允许人工评估 round-2 成本；代码不会自动启动 round 2。失败则保留原 GRPO R1 为最终模型。

## teacher prompt v2 的保留恢复路线（当前不执行）

仅在未来决定重启该方向时，先使用 96GB 实例的无卡模式，拉取对应复现代码并审计 28 条 train smoke 的输入身份：

```bash
cd /root/autodl-tmp/EconoChart
git pull --ff-only origin main
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=8
TEACHER_PY=/root/autodl-tmp/envs/econochart-teacher-cu130/bin/python

"$TEACHER_PY" -m econochart.preflight \
  --stage eval \
  --config configs/eval/opd_teacher_prompt_smoke_v2.yaml \
  --model /root/autodl-tmp/models/Qwen3-VL-32B-Instruct \
  --inputs-only \
  --report outputs/opd/teacher_prompt_smoke_v2/preflight_inputs.json
```

该报告 PASS 后才打开 96GB GPU，并把 28 条 smoke 放进独立 tmux：

```bash
mkdir -p /root/autodl-tmp/EconoChart/outputs/opd/logs
tmux new-session -d -s opd_teacher_prompt_smoke_v2 \
  "bash -lc 'cd /root/autodl-tmp/EconoChart && export PYTHONPATH=/root/autodl-tmp/EconoChart/src && export OMP_NUM_THREADS=8 && /root/autodl-tmp/envs/econochart-teacher-cu130/bin/python -m econochart.evaluation.runner --config configs/eval/opd_teacher_prompt_smoke_v2.yaml --model /root/autodl-tmp/models/Qwen3-VL-32B-Instruct >> outputs/opd/logs/teacher_prompt_smoke_v2.log 2>&1; code=\$?; echo TEACHER_PROMPT_SMOKE_EXIT_CODE=\$code >> outputs/opd/logs/teacher_prompt_smoke_v2.log; exit \$code'"
```

tmux 结束后立刻切回无卡模式，再运行冻结 smoke gate：

```bash
cd /root/autodl-tmp/EconoChart
export PYTHONPATH="$PWD/src"
/root/autodl-tmp/envs/econochart-teacher-cu130/bin/python \
  -m econochart.distillation.prompt_smoke \
  --config configs/opd/teacher_prompt_smoke_v2.yaml
```

任何非零退出都必须保留预测、日志和失败报告，不得用 `--force` 覆盖，也不得改用
qualification 数据调 prompt；需要修订时创建新的协议版本和独立输出目录。
只有日志出现 `TEACHER_PROMPT_SMOKE_EXIT_CODE=0` 且 gate 为
`OPD_TEACHER_PROMPT_SMOKE_PASS`，才进入 `configs/eval/opd_teacher_qualification_v2.yaml`。

## 下载完成后的第一个无卡门

教师模型下载出现 `27/27` 和 exit code 0 后，先让 96GB 实例保持**无卡模式**。在克隆实例中执行以下命令；它们只读取磁盘、计算 SHA256 和 safetensors header，不加载 CUDA：

```bash
(
set -Eeuo pipefail
trap 'code=$?; echo "OPD_NO_GPU_GATE_FAILED exit_code=$code command=$BASH_COMMAND" >&2' ERR
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
)
```

该门不使用 `--force`：如果同名工件已存在，应先审计旧工件，而不是静默覆盖。只有数据 manifest 的泄漏计数全部为 0、两份模型报告均为 `OPD_MODEL_SNAPSHOT_AUDIT_PASS`、接口报告为 `OPD_MODEL_INTERFACE_COMPATIBILITY_PASS`，才安排第一次 GPU 资格推理。任一步非零退出时立即停止并保留原输出；不要打开另一台 GPU 补跑。

## 强制审计点

- processor 产生的视觉 token 数必须大于零；否则立即阻断，不能把纯文本运行冒充多模态 OPD。
- 4B 与 32B 的 vocab、视觉 token、chat template 和 tokenizer 语义指纹必须一致。
- 教师必须按学生实际访问的 completion prefix 评分；固定参考答案不进入 KL 目标。
- prompt、completion、图片、adapter、score shard 与运行配置均保存身份哈希；训练集、资格集、开发集和禁止使用的测试集同时检查 ID、图片路径与图片内容哈希重叠。
- round-1 development 阈值在预测产生前冻结：overall 至少 `+0.01` 且 `P(Δ>0)≥0.90`；numeric recall 至少 `+0.05` 且 `P(Δ>0)≥0.90`；同时执行 numeric precision、trend、evidence、risk、format 护栏。
- 所有阈值都按实际结果如实报告，不允许事后改门或改数字。
