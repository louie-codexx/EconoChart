# 实验协议、消融与阶段门槛

## 1. 核心原则

每项实验必须有可证伪假设。配置数量不是项目质量；能够说明“这个实验改变了哪项决策”才有价值。

所有模型比较固定：

- 基座与 processor；
- EconoChart 数据版本和 test 哈希；
- generation 配置；
- evaluation 代码版本；
- 样本 ID 集合。

比较器会拒绝 reference 字段不一致的预测文件。

## 2. 三类证据

### 训练证据

- train/eval loss 或各 reward 均值、标准差和零方差比例；
- 学习率、梯度范数、异常 NaN/Inf；
- step、epoch、提前停止/中断原因；
- 峰值显存、吞吐、总时长、checkpoint 大小。

### 能力证据

- 内部：numeric、numeric recall/precision、trend、format、evidence、risk、length、overall；
- 外部：ChartQA relaxed accuracy、ChartQAPro 官方兼容分数；
- 切片：task、view、industry、difficulty、scenario；
- 成对 bootstrap 95% CI 与提升为正的概率。

### 行为证据

- 至少 20 个代表性成功案例和 20 个失败案例；
- 错误类型、是否可由训练数据或 reward 解释；
- 模板刷分、数值幻觉、单位错误、趋势反转、建议与风险错配；
- 对无法从图中判断的问题是否表达不确定，而不是编造因果。

## 3. 阶段设计

### Stage B0：base 基线

问题：基座在哪些领域切片已经足够，真正能力缺口是什么？

若不先测 base，无法判断 SFT 是学会新能力，还是只改变了输出格式。

### Stage S0：SFT smoke

问题：最终数据—processor—模型—adapter—collator—保存/重载链是否成立？

只跑 2 step，不报告能力结论。

### Stage S1：领域 QLoRA 主实验

问题：只训练语言 projection，能否改善领域结构、证据覆盖和可验证推理？

正式预算为 9,600 条、2 epochs，覆盖全部 2,400 家训练企业和 4,800 张图。筛选中心候选为 r=16、LR=1e-4、有效 batch=16；固定 512 条 val 面板最终选择 r=16、alpha=32、LR=2e-4，并在正式配置中锁定。完整 2,496 条 test 只在参数确定后的正式 checkpoint 上运行。

### Stage S2：关键消融

rank/LR 的高价值候选统一使用嵌套的 4,800 条 screen、1 epoch，并在同一组 512 条 val ID 上做生成评测。r=32、BF16 LoRA 和公开数据混合是错误分析触发的条件实验，不为凑矩阵默认执行。中间筛选不得查看 2,496 条 test；参数确定后，正式 checkpoint 才运行完整 test。

| 因素 | 候选 | 要回答的问题 |
|---|---|---|
| rank | 8 / 16 / 32 | 容量收益是否超过显存、吞吐和过拟合成本 |
| LR | 5e-5 / 1e-4 / 2e-4 | 收敛、稳定与遗忘的平衡 |
| 方法 | QLoRA / BF16 LoRA | 量化是否带来可测质量损失 |
| 数据 | domain / domain+ChartQA | 通用图表混合是否改善外部分布且不稀释领域能力 |
| 模块 | LLM only / 可选 projector | 主要瓶颈是推理映射还是视觉对齐（错误分析触发） |

### Stage R0：GRPO smoke

问题：Qwen3-VL-4B 与当前 TRL VLM rollout 是否兼容，reward 是否有区分度且不能轻易刷分？

### Stage R1：正式 GRPO

问题：在相同 SFT 起点上，可验证 group-relative optimization 是否进一步改善数值和趋势，并守住格式、外部泛化和长度？

不把“reward 上升”本身当作成功。只有独立固定测试集和人工案例同时支持，才接受结论。

正式预算为 3,600 prompts × 4 generations，即 14,400 rollouts。选择覆盖全部训练企业和 75% 的训练图，并用显式任务配额提高数值、关系和风险任务占比，避免简单值读取因易得高分而主导优化。

## 4. 阶段门槛

### 数据 → base

- 完整 validator 通过；
- test 已冻结并记录哈希；
- 人工抽检无显式标签泄漏和明显不可读图片。

### base → SFT

- 内部与外部基线预测完整；
- 错误分类表确定主要缺口；
- 训练假设与预期改善指标已写入记录。

### SFT smoke → 正式 SFT

- loss 有限、梯度可更新；
- adapter 可保存和重新加载；
- 显存余量与预计总时长可接受；
- run manifest 记录依赖和 Git revision。

### SFT → GRPO

- numeric/trend/overall 至少有一个与假设一致的稳健改善；
- 关键 guardrail 无未解释的大幅回退；
- 剩余错误属于 reward 能检测的任务；
- 每项 reward 在采样输出上有合理分布；
- 已评估不做 RL、继续补 SFT 数据等更简单替代方案。

执行偏差审计（2026-08-29）：正式 R1 在内部 SFT 评测、adapter 重载和 reward 工程门通过后启动，但当时 ChartQA、ChartQAPro 与定性失败案例护栏尚未完成。这不影响内部 SFT→GRPO 配对比较本身，但意味着 R1 只能作为内部审计的探索实验，不能据此发布外部分布或最终产品结论。外部与定性护栏仍须补齐，不能在事后把它们改写成“启动前已通过”。

### GRPO → 最终结论

- 若要宣称 GRPO 有独立增益，SFT→GRPO 固定测试差值和置信区间必须支持改善；若区间跨 0，则只能报告“未建立可测边际增益”，不能用 Base→GRPO 的累计提升替代；
- external benchmark 与人工抽检没有 reward hacking 或未解释的明显退化；
- 只有 R1 先建立可测非零效应时，才用 reward 消融归因哪些分量有效；若 R1 未建立效应，则不为凑矩阵强行运行 R2；
- 结果至少在一个独立 seed 或复跑中方向一致，或明确说明只做单次探索的局限。

## 5. 参数选择如何记录

每个参数写三层理由：

1. 先验依据：算法机制、官方接口、硬件约束；
2. 搜索范围：为什么比较这些值而不是任意值；
3. 实验依据：验证指标、显存、吞吐和错误案例如何决定最终值。

示例：

```text
LoRA rank=16
- 先验：4B 模型仅适配语言 projection，r=16 是容量/成本中间点。
- 搜索：r=8/16/32，alpha 固定为 2r，其他设置不变。
- 实验：在固定 512 条 val 面板与 LR=2e-4 下，r8-r16 overall=-0.011745，
  95% CI=[-0.020636,-0.002010]；numeric recall/precision 与 trend 也可靠更低。
  r8 的 adapter/可训练参数约减半，但端到端吞吐仅高约 1.36%，没有重复运行区间。
- 决策：选择 r=16/alpha=32。r32 未运行，因为 hard/risk 没有显示仍需更高容量的可靠证据。
```

## 6. Reward 消融

Reward 消融是有条件的归因实验。只有正式 R1 在固定测试上先建立可测非零效应，才优先比较：

- 全 reward；
- 去掉 format+length，检查结构 reward 是否只是表面优化；
- 去掉 numeric 或 trend 中一项，验证主要能力来源；
- 权重敏感性（只有错误分析显示需要时）。

同时报告：各奖励分量均值/方差、按配置权重重算的优化用加权 reward、TRL 日志 `reward/reward_std` 的未加权适用分量和、零方差 group 比例、numeric recall/precision、独立测试指标、平均输出长度、重复章节率和数字幻觉案例。若优化用加权 reward 或日志未加权适用分量和上升而 test numeric 不升，优先怀疑 reward overfitting。

该术语边界固定到实际运行的 [TRL v0.28.0 `GRPOTrainer`](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py)：优势计算使用 `reward_weights`，而同一版本随后记录的聚合 `reward/reward_std` 使用未加权 `nansum`。升级 TRL 后必须重新核对，不能沿用本版本结论。

本次 R1 的 SFT→GRPO overall 为 `-0.000157`，95% CI `[-0.002294, 0.002112]`，所有分项均未建立统计可靠差异，因此 R2 未触发。该决定说明现有预算下无法归因 reward 分量的独立能力收益，不等于 reward 设计已被证明无效。

## 7. 停止条件

出现以下任一情况应暂停扩展实验：

- test 或公开 benchmark 被用于训练、早停或反复人工调参；
- reward 代码与 ground truth 对 reference answer 都不能取得高分；
- 同组 generations reward 长期零方差；
- OOM 通过改变多项参数临时绕过，导致无法归因；
- 外部泛化显著退化但没有解释；
- 训练成本超过预估，而当前实验不能改变任何项目决策。

失败不是需要隐藏的结果。记录失败假设、观察和修正，往往比补一个无意义模型名更能经受面试追问。
