# Runtime environment

The training and export environment used for this release is listed below. Length-model batch settings assume approximately 96 GB of GPU memory (`train_bs=1024`). Released six-attribute LoRA hyperparameters use `train_bs=128` and `grad_accum=2`. On GPUs with less memory, reduce the batch size and increase gradient accumulation to keep the effective batch size close to the original.

## Python

| Item | Value |
| --- | --- |
| Environment path | `/root/autodl-tmp/conda_envs/proteogpt_py312` |
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

Activate the original environment:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/proteogpt_py312
export PYTHON="$(which python)"
```

Other machines do not need to use this conda path. Install the versions listed in `requirements.txt` and point `PYTHON` to the resulting interpreter. For `torch==2.9.1`, use the CUDA 12.8 wheel rather than the CPU build.

## Dependencies not required by this repository

- wandb: Launch scripts default to `WANDB_MODE=disabled` and do not pass `--wandb_enable`.
- ampsorter, BioToxiPept, iAMPCN, AMP-Designer, and APEX: Generation and scoring do not invoke them.
- The AMP vs non-AMP discriminator: This head uses a PeptidePrior backbone and is not included in this repository.

## Offline loading

`configs/paths.env` sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`. Once the weights have been downloaded into the local project directories, model loading does not require Hugging Face access.
