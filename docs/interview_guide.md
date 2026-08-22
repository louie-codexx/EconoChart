# 算法面试讲解与技术追问

这份材料是讲解框架，不是背诵稿。方括号中的指标必须由真实实验填写。

## 1. 三分钟项目介绍

> EconoChart 是一个基于 Qwen3-VL-4B-Instruct 的数字经济经营图表后训练项目。我的核心工作不是简单调用 LoRA，而是解决了三个问题：第一，构建企业级无泄漏、数值一致且无标签捷径的多视图经营数据；第二，在单卡约束下用 QLoRA SFT 学习可核验的结论—证据—风险—建议结构；第三，把数值、趋势、证据和风险编码成可单测 reward，从通过评测的 SFT adapter 继续做 GRPO。base、SFT、GRPO 在同一测试记录上比较，并用公开 ChartQA/ChartQAPro、任务切片和成对 bootstrap 检查提升是否真实。最终结果是：[填写主要指标]，仍然存在的局限是：[填写真实失败切片]。

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

起点 r=16、alpha=32、LR=1e-4。不是因为它们“行业标准所以最好”，而是取合理中点后做 r=8/16/32、LR=5e-5/1e-4/2e-4 单因素消融。回答时给出：[各组可训练参数、峰值显存、吞吐、内部和外部指标]，再说明最终选择。

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
- 正式训练报告 reward 方差、输出长度、重复章节率和独立测试结果；
- 做去掉 format/length、numeric、trend 的消融。

局部关联仍不能完整解析复杂句子的时点和主谓关系，因此还要检查数字错配案例；如果 reward 上升但固定 test numeric 不升，就不能说推理能力改善。

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

给出真实顺序：先降视觉 token，再降 GRPO completion 长度/生成数，再评估 beta 和 FlashAttention，最后换显卡。每次只改一项，记录显存和能力代价。失败实验要说明它改变了什么决策，例如“r=32 提升不显著但多占 [X]GB，因此选 r=16”。

## 17. 项目最可能被追问的结果表

面试前必须能现场解释下表，每个数字都能回到 prediction 或 run manifest：

| 模型 | Internal overall | Numeric | Trend | Risk | ChartQA | ChartQAPro | 峰值显存 | 训练时长 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | - | - |
| SFT QLoRA | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| SFT + GRPO | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

还要准备三个失败案例：一个视觉读取错误、一个算术/趋势错误、一个建议与证据不一致，并说明下一步最小修正是什么。
