# 环境兼容与预检

## 已知与未知

最初的 AutoDL 工程 smoke 使用 RTX 4090 24GB、PyTorch 2.5.1+cu118、Torchvision 0.20.1+cu118、Transformers 4.57.1、TRL 0.28.0、PEFT 0.18.0，并把模型放在：

```text
/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
```

该组合通过了从零两步 SFT smoke，但正式 GRPO 从含 optimizer/scheduler/RNG 的 checkpoint 恢复时，Transformers 4.57.1 的安全门拒绝 PyTorch 2.5.1。项目随后只把 PyTorch/Torchvision 升级到 `2.6.0+cu118/0.21.0+cu118`，checkpoint 深度加载、CUDA/BF16、bitsandbytes、GRPO preflight、续训、最终保存与重载均通过。因此项目把 Torch/Torchvision 2.6/0.21 设为所有模型阶段的支持与安全下限，主动拒绝已知不满足安全恢复门的旧栈。这个策略不否认 2.5.1 曾完成从零 smoke，也不声称每条非恢复代码路径在 2.5.1 都技术上无法运行。

为排除运行时升级对质量比较的混杂，正式 SFT adapter 在 Torch 2.6 下重新运行固定 val512；新旧 512 条预测文本、token 元数据和全部质量指标完全一致，差异仅来自 latency 等运行元数据。这只关闭了本面板上的可观测运行时混杂，不代表所有硬件或内核都逐位一致。

## 测试边界

`pyproject.toml` 固定或限制关键接口版本：

| 包 | 范围 |
|---|---|
| Python | 3.10+ |
| Transformers | `>=4.57.1,<5.16` |
| TRL | `==0.28.0` |
| PEFT | `>=0.18,<0.19` |
| Accelerate | `>=1.4,<2` |
| Datasets | `>=3,<5` |
| Torch | `>=2.6` |
| Torchvision | `>=0.21` |

Torch 没有在仓库中锁死 CUDA 构建，因为它必须与 AutoDL 驱动和镜像匹配。当前实测组合是 Torch/Torchvision `2.6.0+cu118/0.21.0+cu118`；preflight 会成对检查二者的最低版本。其他 CUDA 构建必须重新通过 preflight、checkpoint 深载和目标训练 smoke，不能只凭版本号推定兼容。

## 第一次盘点

先安装项目入口，再运行预检；不要根据历史版本直接升级或覆盖整个环境。

```bash
python -m pip install -e .
export ECONOCHART_MODEL_PATH=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct

econochart-preflight --stage sft \
  --config configs/train/sft_qlora_smoke.yaml \
  --report outputs/preflight/sft_before_install.json
```

报告会检查：Python、平台、关键包版本、CUDA、GPU 型号/显存、BF16、bitsandbytes、模型元数据与权重、训练数据 schema/split。报告失败是盘点结果，不应通过删掉检查来绕过。

## 安装项目训练依赖

如果旧环境不兼容，推荐为项目创建独立环境；是否复用 AutoDL 镜像的 Torch 由第一次预检决定。

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
```

FlashAttention 2 是可选加速项，不是最小复现的硬依赖。只有基础 smoke 已通过且安装版本与 Torch/CUDA 匹配时，再把 `attn_implementation: flash_attention_2` 写入单独实验配置。

## 验收报告应保存什么

每个算力环境至少保存：

- AutoDL 镜像名称或镜像 ID；
- GPU 型号和显存；
- NVIDIA driver、CUDA runtime、Torch CUDA；
- Python、Transformers、TRL、PEFT、bitsandbytes、Datasets；
- 是否支持 BF16；
- 模型和 adapter 文件完整性；
- SFT/GRPO smoke 的峰值显存与吞吐。

preflight 原始 JSON 留在 `outputs/`；人工确认后的兼容结论写入对应实验记录后再提交 Git。
