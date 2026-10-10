# 发布权重对应的原 checkpoint

推理文件只保留 `config`、tokenizer、`model.safetensors` 或 `adapter_model.safetensors`，以及 LoRA 的 `stop_head.pt`。没有 `optimizer.pt`、中间 step 和训练日志。

`adapter_config.json` 里的 `base_model_name_or_path` 写成 `models/AMPGuide`。加载代码会先找这个相对路径，找不到时再用仓库里的 `models/AMPGuide`。请在仓库根目录运行脚本。

## 生成模型

- `models/BasePLM`
  - 文章中的 BasePLM。12 层 decoder-only Transformer，约 124M 参数，在 Swiss-Prot 上预训练。PeptidePrior 从这里继续训练。
  - 只保留 `model.safetensors`，不保留重复的 `pytorch_model.bin`。
- `models/PeptidePrior`
  - 文章中的 PeptidePrior。短肽继续预训练，原目录 `length_base_model1/MAE优先的checkpoint`，等于 `checkpoint-84000`。
  - 按长度 MAE 选取。说明见目录里的 `FROZEN_README.txt`。
- `models/AMPGuide`
  - 文章中的 AMPGuide。由 PeptidePrior 在 AMP 序列上适配，原目录 `length_expert_model1/best_model`，等于 `checkpoint-1088`。
  - 按 eval loss 选取。训练最后跑到 step 1472，发布的是 eval loss 最好的 1088。
- `models/CountAMPGuide`
  - 文章中的 CountAMPGuide。由 PeptidePrior 在 non-AMP 短肽上适配，原目录 `length_anti_model1/best_model`，等于 `checkpoint-12000`。
- `models/six_attr_bipolar`
  - 原目录 `length_expert_model1_six_attr_bipolar_notraineval_20260913_114605/checkpoint-5000`。
  - 与 `..._offlineeval_20260915_093322/checkpoint-5000` 是同一份 adapter，只留这一份。
  - LoRA r=16，alpha=32，包含 `lm_head`。
- `models/six_attr_strict_combo`
  - 原目录 `length_expert_model1_six_attr_strict_combo_notraineval_20260912_093641/checkpoint-5000`。
- `models/six_attr_disentangle`
  - 原目录 `length_expert_model1_six_attr_disentangle_lora/run1/adapters`。
  - 训练保存时每个属性目录里都复制了全部 adapter，内容相同。这里每个属性只留一份，目录直接是 `gp/`、`gn/`、`fu/`、`sf/`、`len/`，与 `load_multi_adapters` 的读法一致。

## 六属性五种微调

同一套六属性数据、从 `models/AMPGuide` 出发，训练 5000 step，每 500 step 存一次。发布目录只留每种方法按 composite 选出的那一档，目录是 `models/six_attr_finetune/<方法>/`。全参三种附带 tokenizer 和 `config.json`。LoRA 两种的 `base_model_name_or_path` 是 `models/AMPGuide`。

| 目录 | 方式 | step | 学习率 | 训练 batch |
|------|------|------|--------|------------|
| `lora_r16` | LoRA r=16, alpha=32 | 2500 | 1.5e-5 | 128 × 2 |
| `full_ft` | 全参数 | 500 | 8e-6 | 168 × 1 |
| `freeze_top2` | 冻结底部 10 层，训练顶部 2 层 | 1000 | 1e-5 | 168 × 1 |
| `lora_r64` | LoRA r=64, alpha=128 | 1500 | 1.5e-5 | 78 × 3 |
| `bitfit` | 只训练 bias 和 lm_head | 5000 | 2e-4 | 90 × 2 |

对应原目录：

- `lora_r16`：`length_expert_model1_six_attr_lora_notraineval_20260609_101447/checkpoint-2500`
- `full_ft`：`six_attr_full_ft_20260707_033616/checkpoint-500`
- `freeze_top2`：`six_attr_freeze_top2_20260707_095800/checkpoint-1000`
- `lora_r64`：`six_attr_lora_r64_20260707_095800/checkpoint-1500`
- `bitfit`：`six_attr_bitfit_20260707_095800/checkpoint-5000`

重训入口是 `scripts/train_six_attr_finetune.sh`。生成是 `scripts/generate_six_attr_finetune.sh`。评估是 `scripts/eval_six_attr_finetune.sh`，协议脚本是 `src/multi_attr/07.eval_checkpoint_protocol_five_attr_single_lora.py`。属性打分用本仓库的 GP、GN、FU、SF、MIC 五个专家骨干判别器。SF 的 1 表示有毒。MIC 只写入五档分布，不进入四比特组合条件。

## 判别器

骨干都是 `models/AMPGuide`。来源是 `six_attr_discriminators_hybrid` 的 `best.pt`，与后来打包装的 english 分类器里这五个头是同一份文件。

- `models/discriminators/gram_positive`：GP，二分类，LoRA r=32，按验证 AUROC 选取
- `models/discriminators/gram_negative`：GN
- `models/discriminators/antifungal`：FU
- `models/discriminators/safety`：SF，0 为低毒，1 为有毒
- `models/discriminators/mic`：MIC 五档。输入是原子 `SPxx` 加氨基酸 id。tokenizer 在 `models/discriminators/mic_tokenizer`

每个头的目录里：

- `weights/best.pt`：权重
- `backbone`：指向 `models/AMPGuide`
- `tokenizer`：二分类头指向 AMPGuide 的 tokenizer；MIC 指向 `mic_tokenizer`

没有放入 `amp.pt`。那个判别器的骨干是 PeptidePrior。

## 语料

- `data/corpus/uniprot_short_peptides.txt`：PeptidePrior，3,219,459 行
- `data/corpus/AMP.txt`：AMPGuide，16,062 行
- `data/corpus/nonAMP.txt`：CountAMPGuide，2,821,399 行
- `data/corpus/sequence_species_mic_binned_for_six_attr.csv`：六属性专家和判别器
- `data/splits/split_{train,val,test}.csv`：hybrid 使用的按序列划分，seed=42

长度条件文本和 indexed 划分由 `scripts/train_PeptidePrior.sh`、`scripts/train_AMPGuide.sh`、`scripts/train_CountAMPGuide.sh` 写到 `outputs/prepared/`，seed=42，val/test 比例都是 0.001。这些文件可以重新生成，所以没有放进仓库。

## 没有放进来的实验

lowtox 全参、union20k、五属性 LoRA、判别器 v2/v3/fullft 的旧权重。六属性五种微调只保留上表那一档，不保留每 500 step 的中间 checkpoint。判别器更早配方的训练脚本仍在 `src/discriminators/`，需要的话可以重训，输出在 `outputs/`。
