# EconoChart

## 面向数字经济经营分析的多模态图表推理模型

EconoChart is a multimodal chart reasoning model designed for digital economy analysis scenarios.

本项目旨在构建一个能够理解经营分析图表，并结合业务问题进行推理分析的多模态大模型。

输入：

- 企业经营数据图表
- 用户分析问题

输出：

- 图表信息理解
- 数据趋势分析
- 业务原因解释
- 经营决策辅助建议


---

# 1. 项目背景

随着数字经济的发展，企业经营分析越来越依赖数据可视化报告。

传统的数据分析流程通常需要人工阅读大量图表，并结合业务经验进行分析。

本项目希望探索：

> 多模态大语言模型是否能够理解经营分析场景中的图表信息，并完成类似分析师的推理任务。


---

# 2. 项目目标

构建一个面向数字经济场景的多模态图表推理模型，实现：

- 图表内容理解
- 数据趋势分析
- 业务逻辑推理
- 分析报告生成


---

# 3. 技术路线

整体流程：
数据构造
    |
    |
图表-问题-答案数据集
    |
    |
Vision Language Model
视觉语言模型
(Qwen3-VL)
    |
    |
Supervised Fine-Tuning
监督微调
(SFT)
    |
    |
Reinforcement Learning
强化学习
(RL)
    |
    |
EconoChart Model

---

# 4. 项目结构
EconoChart/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── evaluation/
│   └── utils/
├── configs/
├── scripts/
├── experiments/
└── checkpoints/

---

# 5. 当前进度

## Phase 0 环境准备

✅ 项目初始化

✅ Git版本控制

✅ GitHub仓库建立


## Phase 1 项目架构设计

🚧 数据构造方案设计

🚧 模型选择


---

# 6. 计划

后续将完成：

- 构建数字经济经营分析图表数据集
- 基于Qwen3-VL进行模型训练
- 完成SFT监督微调
- 使用强化学习优化模型推理能力
- 建立评测体系
- 完成Demo展示


---

# 7. Future Work

未来将进一步探索：

- 更复杂的商业分析任务
- 多轮分析能力
- 工具调用能力
- 自动生成经营分析报告
