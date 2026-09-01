# EconoChart GRPO R1 最终模型卡

## 模型身份

项目交付模型固定为 **EconoChart GRPO R1**：

- 基座：`Qwen/Qwen3-VL-4B-Instruct`
- 最终 adapter：`outputs/grpo_qlora_48g_domain_v1/final_adapter`
- adapter SHA-256：`d47c083114e26000c1df5bd52676a1ca9dc2ca708a70c7944c97b3824404486b`
- 训练链：`Base → Domain QLoRA SFT → GRPO R1`
- 状态：项目已关闭，不再启动新增训练

adapter 不是独立模型权重。推理时必须先加载同一 Qwen3-VL-4B-Instruct 基座，再挂载该 LoRA adapter；仓库不提交基座、adapter、checkpoint 或逐条预测。

## 为什么选择它

这是既定后训练主线的最终、可重载工件：3,600 个 prompt、每题 4 个 generation，共 14,400 次 rollout；最终 504 个 adapter 张量、33,030,144 个可训练参数全部通过有限性与重载审计。它把 Base 的内部 overall 从 `0.349780` 推进到最终 `0.806343`，并在 RL 参数实际更新后保持强 SFT 的总体能力，完整交付了 SFT→GRPO 研究链。

固定 2,496 条内部测试上，GRPO overall 为 `0.806343`；相对正式 SFT 的 paired delta 为 `-0.000157`，95% CI `[-0.002294, 0.002112]`。因此可以确认最终 RL 工件没有统计可确认的总体退化；同时，现有证据没有建立 GRPO 的独立增益。Base→GRPO 的 `+0.456563` 是 SFT 与 GRPO 的累计变化，不能全部归因于 RL。

## 训练与 RL 证据

| 项目 | 审计值 |
|---|---:|
| GRPO steps | 3,600 |
| generations / prompt | 4 |
| rollout 总量 | 14,400 |
| reward | numeric / trend / format / evidence / risk / length |
| `beta` / `epsilon` | `0.001` / `0.2` |
| loss | `dr_grpo` |
| adapter 可训练参数 | 33,030,144 |
| 最终 adapter 重载 | PASS |

正式运行经历过 AutoDL 计费关机、Torch checkpoint 安全版本门与 UTF-8 保存故障。恢复过程验证了 optimizer、scheduler、RNG、checkpoint 与 adapter 状态，没有改变数据、seed、reward、batch 或训练目标。

## 评测结果

### 内部测试

| 指标 | GRPO R1 |
|---|---:|
| overall | 0.806343 |
| numeric | 0.593353 |
| numeric recall / precision | 0.567141 / 0.692619 |
| trend | 0.958716 |
| risk | 0.689895 |
| hard overall | 0.754140 |

### 公开 ChartQA / ChartQAPro

固定 4,446 条外部样本上，GRPO overall exact/relaxed 为 `0.448493/0.536207`。相对 SFT 没有建立可测修复；相对 Base 的 ChartQA exact 仍可靠更低，因此公开外评结论保持 `NO_EXTERNAL_GAIN`。

### MME-Finance 边界

MME-Finance 的已完成 paired 实验比较的是 **Base 与并行的 mixed-SFT 候选**，没有评测本模型。其 surrogate exact、ANLS、token-F1 和效率改善，但 numeric recall 从 `0.699934` 降至 `0.282634`。这些是确定性诊断指标，不是官方 MME-Finance 分数，也不能外推成本 GRPO adapter 的表现。

## Mixed-SFT 为什么不是最终模型

Mixed-SFT 从同一 Base 独立训练，加入 3,200 条 ChartQA 后，ChartQA exact 提升 `+0.062000`；但 ChartQAPro relaxed 的 95% CI 下界为 `-0.018805`，未守住预注册的 `-0.01` 非劣门。外部门失败后按计划跳过内部生成评测，因此没有证据证明它能安全替代 domain-SFT/GRPO 主线。

它仍然是有价值的优化诊断：模型获得了更强的短答案、OCR 和实体抽取能力，同时输出长度大幅缩短并损失数值覆盖。该 trade-off 为后续实验提供了明确目标。

## 如果继续优化

优先方案不是直接再跑一轮完整训练，而是：

1. 从 mixed-SFT adapter 启动 300-step recall-repair GRPO pilot；
2. 把当前等权 numeric F1 改为偏重召回的 F2，同时保留 supported-number precision，避免数字倾倒；
3. 只用冻结的内部 validation 选择 pilot；要求 numeric recall 至少提升 3 个百分点，overall 不低于 `-0.01`，numeric precision 不低于 `-0.02`，且无无依据数字、空输出或长度异常；
4. 只有 pilot 通过才扩大训练；ChartQAPro 与 MME-Finance 继续作为 held-out 确认集，不能反复用于调参。

如果错误进一步定位为 OCR、刻度或密集标签读取，再单独解冻 projector；只有 projector 仍不足时才考虑视觉塔。开放式建议与证据质量最终仍需要官方图像感知 judge 或人工偏好评测。

## 使用范围与限制

- 适合：数字经济经营图表分析、可审计实验演示、SFT/GRPO 研究复现。
- 不适合：未经人工复核的投资建议、真实企业决策或把自动指标当作事实保证。
- 没有证明：GRPO 相对强 SFT 的统计显著增益、广泛外部分布提升、官方 MME-Finance 得分或生产就绪性。

聚合证据见 [GRPO 内部摘要](../experiments/results/20260828_grpo_internal_summary.json)、[公开外部摘要](../experiments/results/20260830_external_generalization_summary.json)、[Mixed-SFT 摘要](../experiments/results/20260831_s5_public_mix_result_summary.json)、[MME-Finance 摘要](../experiments/results/20260901_mmefinance_pair_summary.json) 与 [最终选择摘要](../experiments/results/20260901_final_model_decision.json)。
