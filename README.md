# AMPTailor

**Composable generative-expert guidance enables programmable antimicrobial peptide design**  
**基于可组合生成专家引导的可编程抗菌肽设计**

AMPTailor 面向用户指定的多维功能要求进行 *de novo* 抗菌肽（antimicrobial peptide, AMP）生成。框架将短肽语言模型提供的长度可控生成能力，与可组合的抗菌活性、安全性及物种特异性效力条件相结合，用于探索不同条件组合下的候选肽序列，而不是仅依赖先生成、后筛选的流程。

本仓库提供可复现的**源代码、训练入口、生成与评估脚本及文档**。体积较大的**模型权重、Tokenizer、配置和语料库**统一存放在 Hugging Face，避免将数 GB 的二进制权重放入 GitHub 提交历史。发布材料不包含训练日志、Weights & Biases（wandb）记录、无关的中间 checkpoint 或第三方打分器。

> **当前发布状态：** Hugging Face 资源仓库暂设为 **Private**，仅授权用户可访问。GitHub 中提供资源链接并不代表模型和数据已经公开。待论文及相关许可确认后，可将 Hugging Face 仓库调整为 Public。

**资源入口：**

| 资源 | 位置 | 当前用途 |
| --- | --- | --- |
| 项目代码 | 当前 GitHub 仓库 | 训练、生成、评估及复现文档 |
| [AMPTailor — Hugging Face](https://huggingface.co/53386ljx/AMPTailor) | `53386ljx/AMPTailor` | 模型与语料资源（Private） |
| [模型权重](https://huggingface.co/53386ljx/AMPTailor/tree/main/models) | `models/` | 预训练模型、条件生成模型、判别器 |
| [训练数据](https://huggingface.co/53386ljx/AMPTailor/tree/main/data) | `data/` | 语料及数据划分 |
| [运行环境说明](ENVIRONMENT.md) | `ENVIRONMENT.md` | 环境与依赖 |
| [Checkpoint 说明](docs/CHECKPOINTS.md) | `docs/CHECKPOINTS.md` | 模型选择及训练步数 |

---

## 1. Overview | 框架概览

![AMPTailor model and release pipeline](docs/assets/amptailor_pipeline.png)

*图 1 | AMPTailor 的模型预训练、领域适配、条件生成与属性评估模块示意图。该图概括各模块功能；具体的模型调用、条件融合和训练依赖以对应脚本为准。*

### 1.1 可控属性

框架涵盖六个可控维度：

1. **长度（Length）**：通过 `LEN{n}:` 指定肽序列长度。
2. **革兰氏阳性菌活性（Gram-positive activity, GP）**。
3. **革兰氏阴性菌活性（Gram-negative activity, GN）**。
4. **抗真菌活性（Antifungal activity, FU）**。
5. **物种特异性效力（Species-specific MIC potency）**：通过物种与 MIC 档位指定效力要求。
6. **安全性（Safety, SF）**：基于发布模型的毒性二元标签施加约束。

上述属性可按照生成方法组合。不同生成入口支持的条件表达形式并不完全相同，请按各脚本的参数说明使用。

### 1.2 模型演化

四个阶段性语言模型均为 **12 层 decoder-only Transformer**，参数量约 **124M**：

| 模型 | 训练阶段 | 主要作用 |
| --- | --- | --- |
| `BasePLM` | Swiss-Prot 蛋白质序列预训练 | 通用蛋白质序列先验 |
| `PeptidePrior` | 在 5–60 aa 短肽序列上继续预训练 | 短肽生成及长度条件控制 |
| `AMPGuide` | 从 `PeptidePrior` 进行 AMP 领域适配 | AMP 正向生成引导 |
| `CountAMPGuide` | 从 `PeptidePrior` 进行 non-AMP 领域适配 | 对照／反向生成引导 |

训练 `PeptidePrior`、`AMPGuide` 和 `CountAMPGuide` 时，使用序列前的显式长度前缀，例如：

```text
LEN18:WVRLGRYLLRRLKTPFTR<|endoftext|>
```

其中 `LEN18:` 表示所请求的长度为 18；实际生成时使用 `LEN{n}:` 提示。`BasePLM` 是通用蛋白质预训练阶段，不应将其与后续长度条件适配混为一谈。

**主要发布 checkpoint（详见 `docs/CHECKPOINTS.md`）：** `PeptidePrior` step 84000、`AMPGuide` step 1088、`CountAMPGuide` step 12000。六属性相关分支和判别器在下文说明。

---

## 2. Model weights and training data | 模型与数据

模型权重和语料存放在同一 Hugging Face **Model repository**，分别位于两个顶层目录：

```text
AMPTailor/                         # Hugging Face: 53386ljx/AMPTailor
├── models/
│   ├── BasePLM/
│   ├── PeptidePrior/
│   ├── AMPGuide/
│   ├── CountAMPGuide/
│   ├── six_attr_bipolar/
│   ├── six_attr_strict_combo/
│   ├── six_attr_disentangle/
│   ├── six_attr_finetune/
│   └── discriminators/
└── data/
    ├── corpus/
    └── splits/
```

- **`models/`**：发布用权重、Tokenizer、模型配置和适配器等。
- **`data/corpus/`**：用于预训练及领域适配的数据资源。
- **`data/splits/`**：判别器相关的序列划分。
- **`models/six_attr_disentangle/`**：附加的属性相关适配结果；具体用途以对应代码与 checkpoint 文档为准。

### 下载模型与语料

首先安装 Hugging Face CLI：

```bash
python3 -m pip install -U huggingface_hub
```

**在资源仓库仍为 Private 时**，需使用具有访问权限的 Hugging Face 账号登录：

```bash
hf auth login
hf auth whoami
```

在**已克隆的 GitHub 项目根目录**执行：

```bash
hf download 53386ljx/AMPTailor \
  --repo-type model \
  --include "models/**" "data/**" \
  --local-dir .
```

上述命令会在当前目录创建或补齐 `models/` 与 `data/`，并保持 Hugging Face 中的相对路径。不想下载语料时，可仅保留 `--include "models/**"`；仅下载语料则使用 `--include "data/**"`。

> **重要：** 如果你正在维护代码仓库，下载大型权重之前请确认本地 `.gitignore` 已排除 `models/` 和不准备提交到 GitHub 的大语料，以避免意外将模型或语料纳入 Git 提交。将来把 Hugging Face 仓库设为 Public 后，普通下载用户不再需要针对公开仓库进行身份验证。

---

## 3. Generation | 生成方法

AMPTailor 保留四类互补的生成入口。各方法使用不同的条件组织方式，便于分别复现长度控制、对比引导以及六属性组合控制实验。

### 3.1 Contrastive guided decoding | 对比引导解码

通过短肽先验和 AMP / non-AMP 引导模型在生成时组合预测信号。概念形式为：

$$
\mathrm{Guidance}=\mathrm{PeptidePrior}+\alpha\,(\mathrm{AMPGuide}-\mathrm{CountAMPGuide}).
$$

这里表示**生成分数／logits 的引导组合**，并非将三个模型权重直接相加。发布配置采用 `alpha=2.0`、`alpha_eos=0.0`，长度提示为 `LEN{n}:`。

```bash
bash scripts/generate_guided.sh
```

自定义示例：

```bash
TARGET_LENGTH=12 NSAMPLES=100 ALPHA=2.0 bash scripts/generate_guided.sh
```

### 3.2 Bipolar fusion | 双极条件融合

使用 `models/six_attr_bipolar/` 的适配权重，对所请求的属性条件与反向条件进行 `contrast` 或 `mix` 融合；默认采用 `contrast`，`alpha=2`。

```bash
bash scripts/generate_bipolar_fusion.sh
```

例如请求同时具备 GP、GN、FU 活性并满足低毒性约束：

```bash
BITS=1110 LENGTH=18 N=16 KIND=contrast ALPHA=2 \
  bash scripts/generate_bipolar_fusion.sh
```

### 3.3 Strict-combination conditioning | 严格组合条件

使用 `models/six_attr_strict_combo/`，通过单条联合条件前缀指定目标属性，例如：

```text
GP1+GN1+FU1+SF0+LEN18:
```

运行：

```bash
bash scripts/generate_strict_combo.sh
```

自定义示例：

```bash
BITS=1110 LENGTH=18 N=16 bash scripts/generate_strict_combo.sh
```

**`BITS` 编码规则：** 四位依次为 GP、GN、FU、SF。对 GP/GN/FU，`1` 表示请求相应抗微生物活性；对于当前发布的 SF 标签，`1` 表示有毒、`0` 表示低毒。因此 `1110` 对应 **GP+ / GN+ / FU+ / 低毒**，而不是四项都为 `1`。

### 3.4 Six-attribute fine-tuning | 六属性微调

由 `AMPGuide` 继续微调的五种策略保存在 `models/six_attr_finetune/`：

| 方法 | 训练／适配策略 | 发布 checkpoint step |
| --- | --- | ---: |
| `lora_r16` | LoRA rank 16 | 2500 |
| `full_ft` | 全参数微调 | 500 |
| `freeze_top2` | Freeze-top2 | 1000 |
| `lora_r64` | LoRA rank 64 | 1500 |
| `bitfit` | BitFit | 5000 |

以上 step 为按 composite 指标选出的 checkpoint，不代表五种方法的训练总步数相同。

```bash
bash scripts/generate_six_attr_finetune.sh
bash scripts/eval_six_attr_finetune.sh
```

指定评估／生成方法时：

```bash
METHOD=lora_r16 bash scripts/generate_six_attr_finetune.sh
```

---

## 4. Discriminators and evaluation | 判别器与评估

发布的五个 hybrid 判别头基于 `AMPGuide` 骨干，分别评估：

| 缩写 | 属性 |
| --- | --- |
| `GP` | 革兰氏阳性菌活性 |
| `GN` | 革兰氏阴性菌活性 |
| `FU` | 抗真菌活性 |
| `SF` | 安全性／毒性标签 |
| `MIC` | 物种特异性 MIC 效力档位 |

评估入口：

```bash
bash scripts/score_sequences.sh outputs/generate/bipolar_contrast_L18.fasta
```

六属性微调的历史 checkpoint 协议位于：

```text
src/multi_attr/07.eval_checkpoint_protocol_five_attr_single_lora.py
```

该协议使用 GP、GN、FU、SF 四项预测和长度条件计算对应的条件满足率，并记录 MIC 分箱分布；**MIC 分布记录与所有六项严格联合满足率并不是同一个指标**。请按具体实验的评估脚本与定义复现论文中的结果。

较早的判别器训练配方（v1/v2/v3、氨基酸词表方案、全参数与原子 LoRA 等）保存在 `src/discriminators/`；模型资源仓库仅保留最终采用的 hybrid 五个判别头。

---

## 5. Installation and quick start | 环境与快速运行

完整安装要求、Python 环境和软硬件说明见 [ENVIRONMENT.md](ENVIRONMENT.md)。

```bash
# 从你已克隆的 GitHub 项目根目录运行
source configs/paths.env
```

`PROJECT_ROOT` 应指向当前仓库。请不要把路径改回原训练服务器上的绝对路径。

**下载权重后，无需重新训练即可执行：**

```bash
bash scripts/generate_guided.sh
bash scripts/generate_bipolar_fusion.sh
bash scripts/generate_strict_combo.sh
bash scripts/score_sequences.sh outputs/generate/bipolar_contrast_L18.fasta
```

请注意，最后一条打分命令需要指定的 FASTA 文件已由上游生成任务产生；具体输出路径可能随脚本参数而改变。

---

## 6. Re-training | 重新训练

重训输出写入 `outputs/`，不会主动覆盖 `models/` 中的发布权重。长度模型训练的默认 batch 配置按约 **96 GB GPU 显存**环境设置；显存更小时需要调整 batch size 或相关训练参数。

```bash
bash scripts/train_PeptidePrior.sh
bash scripts/train_AMPGuide.sh
bash scripts/train_CountAMPGuide.sh
bash scripts/train_six_attr_bipolar.sh
bash scripts/train_six_attr_strict_combo.sh
bash scripts/train_disentangled_attr_experts.sh
bash scripts/train_hybrid_discriminators.sh
```

五种六属性微调选择其一：

```bash
# FINETUNE_MODE: lora_r16 | full | freeze_top | lora_r64 | bitfit
FINETUNE_MODE=lora_r16 bash scripts/train_six_attr_finetune.sh
```

注意：训练脚本中的 `FINETUNE_MODE=full` / `freeze_top` 与生成评估脚本所用的 `METHOD=full_ft` / `freeze_top2` **命名不同**，应按照各自脚本接受的参数取值设置。

每个发布权重的训练超参数、对应原 checkpoint 和选择标准见 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)。

---

## 7. Repository layout | 代码目录

以下为 GitHub 中的代码组织示意。大型 `models/` 和 `data/corpus/` 建议通过前述 Hugging Face 命令单独获取。

```text
AMPTailor/
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── configs/
│   └── paths.env
├── docs/
│   ├── CHECKPOINTS.md
│   └── assets/
│       └── amptailor_pipeline.png
├── scripts/                  # 训练、生成、评估入口
├── src/
│   ├── pretrain/             # BasePLM / PeptidePrior / AMPGuide / CountAMPGuide
│   ├── multi_attr/           # 六属性适配、微调与评估协议
│   ├── discriminators/       # 判别器训练与历史配置
│   └── generation/           # 引导解码、条件生成与打分
├── data/
│   ├── splits/               # 小体积划分文件（可根据发布策略保留）
│   └── corpus/               # 大语料，托管于 Hugging Face
├── models/                   # 模型权重，托管于 Hugging Face
└── outputs/                  # 生成结果与重训输出（不纳入版本库）
```

### GitHub 与 Hugging Face 的文件分工

由于 `model.safetensors`、`best.pt` 及部分语料文件均可能超过 GitHub 普通 Git 文件大小限制，本项目建议：

- **GitHub：** `src/`、`scripts/`、`configs/`、`docs/`、`README.md`、运行环境文件及适合版本控制的小体积数据。
- **Hugging Face：** `models/`、`data/corpus/` 及需要统一管理的数据资源。
- **不提交：** `outputs/`、训练日志、wandb 目录和无关的中间 checkpoint。

即使已在 `.gitattributes` 中配置 Git LFS，也**不意味着 GitHub 的免费 LFS 存储与带宽配额足以容纳全部模型**。如果模型或语料已被暂存，需要在推送 GitHub 前检查 Git 索引与提交历史，避免把大文件（包括 LFS 对象）一并推送。

---

## 8. Reproducibility and release notes | 复现与发布说明

- **可复现性：** 模型配置、Tokenizer、checkpoint 选择标准、训练与评估入口均应与对应的公开版本一致。
- **私有阶段：** 当前 Hugging Face 链接仅供拥有访问权限的人员下载；公开 GitHub README 并不会自动开放私有权重。
- **正式公开：** 公开之前应核对模型训练数据及上游资源的再分发许可，尤其是 Swiss-Prot 来源语料、其他收集的 AMP / non-AMP 数据和原始预训练权重。
- **发布许可：** 在确定代码、模型和数据的授权条件后，分别补充合适的 LICENSE、Model Card 和数据集说明。
- **论文引用：** 论文题目、正式引文、DOI 和版本化发布信息在确认后补充；在此之前不使用未经确认的引用条目。

> **Public release checklist:** 验证所有模型与数据文件、补充模型使用示例、核对许可与数据再分发权限、确定公开版本，并在对外开放 Hugging Face 仓库后更新本 README 的访问状态说明。
