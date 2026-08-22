# 环境兼容与预检

## 已知与未知

历史会话曾记录 AutoDL 使用 RTX 4090 24GB、PyTorch 2.5.1+cu118、Transformers 4.57.1、TRL 0.22.0、PEFT 0.14.0，并把模型放在：

```text
/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
```

这些只代表历史状态，不代表当前机器。最终实现基于 TRL 0.28.0 的多模态 SFT/GRPO 接口，旧的 TRL 0.22.0 和 PEFT 0.14.0 会被 preflight 判为不兼容。

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
| Torch | `>=2.5` |

Torch 没有在仓库中锁死 CUDA 构建，因为它必须与 AutoDL 驱动和镜像匹配。首次 GPU smoke 通过后，应把真实的 Torch/CUDA/驱动组合写入实验记录，并只把验证过的组合称为“已测试环境”。

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
