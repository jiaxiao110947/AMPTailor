# 运行环境

发布时的训练与导出环境如下。长度模型的 batch 按约 96GB 显存设定（`train_bs=1024`）。六属性 LoRA 的发布超参是 `train_bs=128`、`grad_accum=2`。显存更小的机器需要按比例减小 batch，并相应增加梯度累积，使有效 batch 接近原来的值。

## Python

| 项 | 值 |
| --- | --- |
| 环境路径 | `/root/autodl-tmp/conda_envs/proteogpt_py312` |
| Python | 3.12.12 |
| torch | 2.9.1+cu128 |
| CUDA | 12.8 |
| transformers | 4.57.6 |
| peft | 0.18.1 |
| accelerate | 1.12.0 |
| numpy | 2.4.1 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| safetensors | 0.7.0 |
| tokenizers | 0.22.2 |

激活：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/proteogpt_py312
export PYTHON="$(which python)"
```

其他机器不必使用这个 conda 路径。装好 `requirements.txt` 里列出的版本，并把 `PYTHON` 指到那个解释器即可。`torch==2.9.1` 需要 CUDA 12.8 对应的 wheel，不要装成 CPU 版。

## 本仓库不依赖的东西

- wandb。启动脚本默认 `WANDB_MODE=disabled`，也不传 `--wandb_enable`。
- ampsorter、BioToxiPept、iAMPCN、AMP-Designer、APEX。生成和打分都不调用它们。
- AMP vs non-AMP 判别器。那个头的骨干是 PeptidePrior，没有放进本仓库。

## 离线加载

`configs/paths.env` 会设置 `TRANSFORMERS_OFFLINE=1` 和 `HF_HUB_OFFLINE=1`。权重都在仓库内，不需要访问 Hugging Face。
