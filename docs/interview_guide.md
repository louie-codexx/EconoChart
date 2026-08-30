# 算法面试讲解与技术追问

这份材料是讲解框架，不是背诵稿。内部与 ChartQA/ChartQAPro 外部指标均使用审计值；机制归因只陈述错误迁移能够支持的范围，不把描述性代理包装成 OCR 或领域过拟合的因果证明。

## 1. 三分钟项目介绍

> EconoChart 是一个基于 Qwen3-VL-4B-Instruct 的数字经济经营图表后训练项目。我的核心工作不是简单调用 LoRA，而是解决了三个问题：第一，构建企业级无泄漏、数值一致且无标签捷径的多视图经营数据；第二，在单卡约束下用 QLoRA SFT 学习可核验的结论—证据—风险—建议结构；第三，把数值、趋势、证据和风险编码成可单测 reward，从通过评测的 SFT adapter 继续做 GRPO。固定 2,496 条内部测试上，Base overall 为 `0.349780`，正式 SFT 为 `0.806499`，Base→SFT 成对提升 `+0.456719`、95% CI `[0.448794, 0.464756]`。GRPO 为 `0.806343`，相对 SFT 的变化只有 `-0.000157`、95% CI `[-0.002294, 0.002112]`，所以不宣称 GRPO 有独立边际增益。固定 4,446 条公开外评上，Base/SFT/GRPO 的 exact 为 `0.467386/0.448493/0.448493`；Base→SFT 的可靠下降主要集中于 numeric，GRPO 没有建立修复，因此结论是 `NO_EXTERNAL_GAIN`。当前执行预注册 mixed-SFT 消融；无卡输入门已通过，但 GPU 训练、开放式长报告人工非退化和独立经济领域外部基准仍待完成。

## 2. 为什么选 Qwen3-VL-4B-Instruct？

- Instruct 基座已经具备多模态对话能力，项目关注领域后训练而不是从零对齐。
- 4B 在单卡 4090 上可用 QLoRA 跑完整闭环，适合做真实消融而不是只演示推理。
- Qwen3-VL 支持可控图像 token 和标准 Transformers processor，便于统一 SFT、RL 和评测。
- 不声称它一定优于所有基座；当前项目固定一个基座，减少比较维度。跨基座是后续工作。

## 3. 为什么自建合成数据，可靠性怎么保证？

公开 ChartQA 主要是短答案图表 QA，缺少数字经济经营结构和长回答。合成数据提供精确底层表、可计算 ground truth 和受控风险场景。

可靠性不来自“随机生成很多图”，而来自：业务恒等式、企业级 split、无 scenario 泄漏、固定尺寸、多视图、任务对齐回答、manifest 哈希、自动 validator 和人工抽检。

合成数据仍不能证明真实业务泛化，所以必须报告 ChartQA/ChartQAPro 与人工失败案例，并明确建议不代表真实企业处方。

## 4. 旧版 5,000 张图为什么没继续用？

量化审计发现：标题直接泄漏经营类型；Users 与财务指标共轴压扁曲线；25,000 条 SFT 只有 3 种答案且同图五问 100% 复用回答；DPO 只有少数模板，eval 还存在列表比较和训练泄漏。继续训练会得到漂亮但不可信的结果，因此保留审计结论，用 v2 确定性替代，并删除旧生成资产。

## 5. SFT、LoRA、QLoRA是什么关系？

SFT 描述训练目标，LoRA/QLoRA 描述参数更新与权重存储方式。项目主路线是“QLoRA 方式的 SFT”，不是全参 SFT 后机械再做 LoRA。

QLoRA 用 4-bit NF4 保存冻结基座、BF16 计算 LoRA 更新；BF16 LoRA 作为方法消融，检查量化节省是否带来质量损失。

## 6. 为什么默认冻结视觉塔和 projector？

首轮假设是基座已经能看懂常见图表，主要缺口是领域分析结构和可验证推理映射。只训练语言 projection 更省显存、遗忘风险低且便于归因。若错误分析显示刻度/OCR 是主瓶颈，再单独解冻 projector，而不是一开始扩大所有可训练模块。

## 7. rank、alpha、学习率怎么选？

起点是 r=16、alpha=32、LR=1e-4。学习率阶段固定 r16，在同一 512 条 val 面板比较 `5e-5/1e-4/2e-4`；`2e-4 - 1e-4` 的 overall 为 `+0.020102`、95% CI `[0.010480, 0.029618]`，numeric recall/precision、trend 和 hard overall 也可靠更高，因此选 `2e-4`。

rank 阶段固定 LR=2e-4 比较 r8/r16。r8 只有 `16,515,072` 个可训练参数、adapter `31.571 MiB`，约为 r16 的一半；但 `r8-r16` overall 为 `-0.011745`、95% CI `[-0.020636,-0.002010]`，numeric recall/precision 与 trend 也可靠更低，而端到端吞吐仅高约 1.36%、没有重复运行区间。因此最终选 r16/alpha32。r32 没有运行，因为 hard/risk 未显示 r16 仍存在可靠容量不足；不能把“未触发”讲成“跑过但失败”。

## 8. 为什么 batch=1 但有效 batch=16？

单样本含图像 token，物理 batch=1 控制峰值显存；梯度累积 16 提供更稳定的 adapter 更新。有效 batch 改变会影响优化噪声，因此不能在 OOM 时无记录地调整。

GRPO 还要求有效 batch 整除 `num_generations`，preflight 会提前拒绝错误组合。

## 9. 为什么不直接做 GRPO？

未经 SFT 的模型可能连输出结构和领域指令都不稳定，RL rollout 成本高且 reward 容易被表面模式利用。先 SFT 建立基本行为，再对残余的可验证错误做 GRPO，更稳定也更容易说明增益来源。

## 10. 为什么是 GRPO，不是 PPO 或 DPO？

任务有可程序化验证的数值、趋势和证据目标，适合对同一 prompt 的多次 generation 做相对比较；GRPO 不需要额外 value model。PPO 引入 value model 和更多显存/稳定性复杂度，对当前问题没有必要。

DPO 需要可信 chosen/rejected 偏好对。旧 DPO 只是模板化 rejected，模型会学习捷径，因此不把 DPO 当必经阶段。未来只有收集真实分析师偏好或严格 rejection sampling 数据后才重新评估。

## 11. Reward 如何防止刷分？

- numeric 要求目标值出现在对应指标的局部上下文，并用目标召回率与底层业务真值数字精确率的调和平均计分；
- trend 绑定具体指标并检查相反趋势；
- format 权重有限，不能靠章节模板主导总分；
- evidence/risk 使用受控目标；
- length 只是软约束且权重最低；
- reference answer 先做离线单测；
- 正式训练区分按配置权重重算的优化 reward，与 TRL 日志中未加权适用分量和 `reward/reward_std`，并报告零方差 group、输出长度、重复章节率和独立测试结果；
- 只有 R1 先建立可测非零效应，才做去掉 format/length、numeric、trend 的归因消融。

局部关联仍不能完整解析复杂句子的时点和主谓关系，因此还要检查数字错配案例；如果优化 reward 或日志 reward 上升但固定 test numeric 不升，就不能说推理能力改善。本次 R1 的 SFT→GRPO 各项区间均未建立可靠差异，所以 R2 未触发；这表示当前预算下没有可归因的 GRPO 边际效果，不表示某个 reward 分量已被证明无效。

## 12. 为什么用 dr_grpo 和不缩放 reward？

原始按完成序列长度归一化可能产生长度偏置；dr_grpo 用固定 `max_completion_length` 归一化。按组 reward 标准差缩放可能让不同难度问题获得不均衡权重，所以初始配置设 `scale_rewards=none`。这两个选择仍需消融，面试时应区分“算法动机”和“本项目实证结论”。

## 13. beta=0 与 beta=0.001 如何解释？

4090 smoke 的目标是兼容性和 reward 链路，beta=0 关闭 KL/reference-policy 计算路径以降低显存和计算开销。正式高显存配置用 beta=0.001 检查 KL 约束，减少策略偏离 SFT。PEFT 场景下参考策略是否复用冻结基座由当前 TRL 实现决定，不能笼统声称一定复制一份完整模型；应以 run manifest 和峰值显存为证据。若最终仍用 beta=0，要说明硬件取舍和外部能力 guardrail，而不是把 smoke 参数包装成理论最优。

## 14. 如何证明不是数据泄漏？

- split 单位是企业，不是 QA 行；
- entity/chart/image 三层交集自动检查；
- test 在训练前冻结并记录哈希；
- ChartQA test 和 ChartQAPro 不参与训练、早停或选参；
- 比较器要求两个预测文件 ID 和 reference 完全一致；
- 任何数据生成逻辑变化都提高版本并重跑 base。

## 15. 指标为什么不能只报 overall？

overall 可能由格式或容易的 value retrieval 拉高。项目分别报告 numeric、trend、evidence、risk、公开基准和效率，并按任务/视图/行业/难度/scenario 切片。成对 bootstrap 还能区分“均值轻微波动”和“样本级一致改善”。

## 16. 训练失败或 OOM 怎么讲？

OOM 的预设顺序仍是先降视觉 token，再降 GRPO completion 长度/生成数，再评估 beta 和 FlashAttention，最后换显卡；每次只改一项并记录能力代价。但本次正式链路的真实故障不是 OOM：AutoDL 欠费关机后从 checkpoint-2900 恢复，先被 Torch 2.5.1 的安全版本门拒绝，升级到已验证的 Torch 2.6；随后 TRL 模型卡模板因进程编码失败，显式固定 UTF-8 后越过 checkpoint-3000 并完成。两次修复都未改数据、seed、reward、batch 或训练目标。r32 则因预注册触发条件不成立而没有运行，不能虚构其显存或质量结果。

## 17. 项目最可能被追问的结果表

面试前必须能现场解释下表，每个数字都能回到 prediction 或 run manifest：

| 模型 | Internal overall | Numeric | Trend | Risk | ChartQA exact / relaxed | ChartQAPro exact / relaxed | 显存证据 | 训练时长 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.349780 | 0.240061 | 0.643119 | 0.059233 | 0.701600 / 0.791200 | 0.166495 / 0.228754 | 未按统一整卡峰值归档 | - |
| SFT QLoRA | 0.806499 | 0.594490 | 0.956881 | 0.685105 | 0.676400 / 0.778000 | 0.155704 / 0.226850 | Trainer 峰值分配增量 2,790 MiB，非整卡峰值 | 26,816.85s（约 7:26:57） |
| SFT + GRPO | 0.806343 | 0.593353 | 0.958716 | 0.689895 | 0.675600 / 0.775600 | 0.156732 / 0.228662 | `nvidia-smi` 单次进程观测 7,614 MiB，非峰值 | 14,168.06s（Trainer 结果；不等于含中断的总计费墙钟） |

Base→SFT 的内部提升可靠；SFT→GRPO 的内部所有分项均为 inconclusive，不能把 Base→GRPO 的累计提升归给 GRPO。外部错误迁移已核验 4,446 个同 ID 样本：282 条由 Base 对变为 SFT 错，其中 212 条也失去 relaxed 分数；至少 144 条是明确内容、数值或答案类型错误，另有 100 条属于表达/评分歧义候选上界。结论不是“全是格式问题”，而是数值能力负迁移夹杂次要评分敏感性；S5 mixed-SFT 正是对这一失败假设的预注册消融，尚不能提前写成修复成功。该短答案样本审计也不能代替开放式报告中的建议—证据一致性人工复核，后者继续单列为最终结论门。
