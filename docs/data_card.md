# EconoChart-v2 数据卡

## 数据集概览

EconoChart-v2 是用于数字经济经营图表理解和分析后训练的合成多模态数据集。默认配置以固定种子 `20260821` 生成 3,000 家企业，每家两张图、每图四个任务，目标规模为 6,000 张图和 24,000 条问答。

GitHub 只保存 8 家企业、16 张图、64 条记录的完整样例。全量数据由 `configs/data/econochart_v2.yaml` 在 AutoDL 确定性生成，实际统计与哈希写入构建目录的 `manifest.json`。

## 行业与场景

行业覆盖电子商务、企业软件服务、金融科技、数字内容、云服务和本地生活服务。经营轨迹覆盖高增长、稳定、成本压力、需求放缓、用户流失、恢复、利润率扩张和变现偏弱。

`scenario` 只存在于底层 metadata，用于生成和切片分析；不会出现在图标题、问题或模型输入中的其他标签位置。

## 指标与一致性

五年序列包含：

- 收入、成本、利润、利润率；
- 活跃用户、ARPU、转化率、CAC；
- 产品收入结构与区域结构。

生成器强制：

```text
Revenue = Active Users × ARPU / 100
Profit = Revenue - Cost
Profit Margin = Profit / Revenue
Product shares ≈ 100%
Region shares ≈ 100%
```

数值保存时存在小数舍入，validator 使用与精度相匹配的容差。

## 图表视图

- financial：收入/成本并列柱 + 利润折线；
- operations：用户/ARPU 与转化率/CAC 分面双轴；
- growth：年度同比变化；
- composition：产品与区域结构；
- dashboard：关键经营指标组合视图。

所有图使用固定像素尺寸，不用 `bbox_inches=tight`；图形样式随机源和经营场景随机源隔离，降低视觉捷径。

## 任务与回答

任务覆盖值读取、数值推理、趋势、指标关系、风险诊断、决策支持和综合报告。reference 只监督可展示、可核验的分析结果，不包含声称为模型内部思维过程的长 Chain-of-Thought。

标准输出段落为：

```text
【结论】
【数据依据】
【风险】（题目需要时）
【建议】（题目需要时）
```

建议是根据受控风险规则生成的决策候选，不是现实企业的可靠处方。

## 划分与泄漏控制

数据先按 `entity_id` 划分，再生成图表和问答。validator 检查企业、chart、image 的 train/val/test 交集必须为空，并检查：

- schema、ID 和 split；
- 图片引用、解码、模式和尺寸；
- 企业利润恒等式；
- 数值 ground truth 是否出现在 reference；
- 数值目标是否绑定指标词，以及底层数值支持集是否合法；
- 每图任务粒度、问答复用和答案多样性；
- annotation 与 raw 文件哈希。

## 训练预算子集

完整标注保留为数据母集，正式实验不通过“取 JSONL 前 N 条”降本。固定子集构建器使用种子 `20260821` 和完整数据 manifest 哈希，生成：

| 子集 | 规模 | 覆盖约束 | 用途 |
|---|---:|---|---|
| SFT screen | 4,800 | 每张 train 图 1 题，覆盖 2,400 家企业/4,800 张图 | rank/LR 筛选，1 epoch |
| SFT final | 9,600 | 每张 train 图 2 题，screen 嵌套其中 | 正式 SFT，2 epochs |
| GRPO | 3,600 prompts | 3,600 张不同 train 图，全部 2,400 家企业至少出现 1 次 | 4 generations，共 14,400 rollouts |
| development | 512 | 仅从 val 按任务及业务切片分层抽样 | 中间选参与错误分析 |
| internal test | 2,496 | 完整 test，不参与训练、早停或选参 | Base 与最终里程碑完整评测 |

SFT 优先保留全部图片和企业覆盖、减少同一图的重复问题；GRPO 使用显式任务配额，避免低成本的简单读取任务主导昂贵 rollout。所有子集的源文件哈希、结果哈希、任务/视图/行业/难度/场景分布和覆盖不变量写入 `subsets/subset_manifest.json`。

## 公开数据角色与许可

| 数据集 | 许可 | 本项目角色 | 污染规则 |
|---|---|---|---|
| ChartQA | GPL-3.0 | train/val 可做混合 SFT 消融；test 固定外部评测 | test 不训练、不选参 |
| ChartQAPro | MIT | test-only 高难挑战集 | 全部禁止训练和选参 |
| PlotQA | CC-BY-4.0 | 后续可选的数值推理扩展 | 不属于最小闭环 |

本仓库不重新分发公开数据。准备代码从官方 Hugging Face 数据源读取并在本地生成统一 annotation。ChartQAPro 的多轮样本保留历史问答作为上下文，只预测最后一轮，并导出官方评测格式。原始 `Year` 序列作为答案元数据完整保留，不强制与问答轮数等长；若官方最终目标 `Answer[-1]` 为空，该源行会被确定性排除且不回填，绝不伪造标签。`manifest.json` 同时记录 selected source/evaluable 数量、被排除的源索引与记录 ID、Year 长度异常及保留情况，所生成的是固定可评分子集，不能冒充未经排除的完整官方 test。

ChartQA 准备阶段按规范化 PNG 的完整 SHA-256 检查跨 split 图像重复。固定外部 test 完整保留；先从 validation 删除与 test 同图的全部问答，再从可选 train 删除与清理后 validation 或 test 同图的全部问答，两个可变 split 均不回填。删除前后数量、冲突哈希、分 split 样本 ID、未引用图片文件和最终零重叠检查写入 `manifest.json` 的 `split_decontamination`。

如果发布混合训练后的 adapter 或派生数据，应在发布前重新核对数据集条款与目标平台要求；本数据卡不是法律意见。

## 已知限制

- 合成数据不能证明对真实财报、OCR 噪声、复杂 dashboard 或行业因果关系的泛化。
- 生成轨迹来自有限规则族，存在模板和分布可识别风险。
- reference 建议受规则约束，缺少真实分析师偏好与业务结果反馈。
- 图中数值较干净，不能替代低清晰度、遮挡、密集标注等鲁棒性测试。
- 自动 reward 只能检查被编码的事实，无法完整衡量建议质量和自然语言合理性。
- 指标局部窗口能降低数字错配，但仍不能完整解析复杂自然语言中的主谓与时点关系。

因此项目必须同时报告 EconoChart 内部测试、ChartQA、ChartQAPro、典型错误案例和人工抽检，不能只凭合成测试总分宣称真实业务能力。

## 复现

```bash
econochart-build --config configs/data/econochart_v2.yaml
econochart-validate --dataset-root data/generated/econochart_v2 --full-image-scan
econochart-build-subsets --config configs/data/training_subsets.yaml
```

不使用 `--overwrite` 时，构建器拒绝覆盖非空目录；需要重建时必须显式传入该参数。
