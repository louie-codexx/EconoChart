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

## 公开数据角色与许可

| 数据集 | 许可 | 本项目角色 | 污染规则 |
|---|---|---|---|
| ChartQA | GPL-3.0 | train/val 可做混合 SFT 消融；test 固定外部评测 | test 不训练、不选参 |
| ChartQAPro | MIT | test-only 高难挑战集 | 全部禁止训练和选参 |
| PlotQA | CC-BY-4.0 | 后续可选的数值推理扩展 | 不属于最小闭环 |

本仓库不重新分发公开数据。准备代码从官方 Hugging Face 数据源读取并在本地生成统一 annotation。ChartQAPro 的多轮样本保留历史问答作为上下文，只预测最后一轮，并导出官方评测格式。

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
```

不使用 `--overwrite` 时，构建器拒绝覆盖非空目录；需要重建时必须显式传入该参数。
