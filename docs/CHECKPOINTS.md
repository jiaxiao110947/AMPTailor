# Original checkpoints for released weights

Inference releases retain only `config`, the tokenizer, `model.safetensors` or `adapter_model.safetensors`, and the LoRA `stop_head.pt`. Optimizer state (`optimizer.pt`), intermediate checkpoints, and training logs are excluded.

In `adapter_config.json`, `base_model_name_or_path` is set to `models/AMPGuide`. Loading code first checks this relative path, then falls back to `models/AMPGuide` under the repository root. Run scripts from the repository root.

## Generation models

- `models/BasePLM`
  - BasePLM in the manuscript: a 12-layer decoder-only Transformer with approximately 124M parameters, pretrained on Swiss-Prot. PeptidePrior continues training from this model.
  - Only `model.safetensors` is retained; the duplicate `pytorch_model.bin` is excluded.
- `models/PeptidePrior`
  - PeptidePrior in the manuscript: continued pretraining on short peptides. The original length-MAE-selected checkpoint under `length_base_model1` corresponds to `checkpoint-84000`.
  - Selected by length MAE. See `FROZEN_README.txt` in the model directory for details.
- `models/AMPGuide`
  - AMPGuide in the manuscript: adaptation of PeptidePrior to AMP sequences. Original directory: `length_expert_model1/best_model`, corresponding to `checkpoint-1088`.
  - Selected by evaluation loss. Training ended at step 1472; the release uses step 1088, which had the best evaluation loss.
- `models/CountAMPGuide`
  - CountAMPGuide in the manuscript: adaptation of PeptidePrior to non-AMP short peptides. Original directory: `length_anti_model1/best_model`, corresponding to `checkpoint-12000`.
- `models/six_attr_bipolar`
  - Original directory: `length_expert_model1_six_attr_bipolar_notraineval_20260913_114605/checkpoint-5000`.
  - Identical to the adapter in `..._offlineeval_20260915_093322/checkpoint-5000`; only one copy is retained.
  - LoRA r=16, alpha=32, including `lm_head`.
- `models/six_attr_strict_combo`
  - Original directory: `length_expert_model1_six_attr_strict_combo_notraineval_20260912_093641/checkpoint-5000`.
- `models/six_attr_disentangle`
  - Original directory: `length_expert_model1_six_attr_disentangle_lora/run1/adapters`.
  - Training saves copied all adapters into each attribute directory, producing duplicate contents. The release retains one copy per attribute in `gp/`, `gn/`, `fu/`, `sf/`, and `len/`, matching the directory layout expected by `load_multi_adapters`.

## Five six-attribute fine-tuning strategies

All strategies use the same six-attribute data and start from `models/AMPGuide`, training for 5000 steps with checkpoints every 500 steps. Each release retains only the checkpoint selected by the composite metric in `models/six_attr_finetune/<method>/`. The three strategies that save full model weights include a tokenizer and `config.json`. Both LoRA strategies set `base_model_name_or_path` to `models/AMPGuide`.

| Directory | Strategy | Step | Learning rate | Training batch × gradient accumulation |
|------|------|------|--------|------------|
| `lora_r16` | LoRA r=16, alpha=32 | 2500 | 1.5e-5 | 128 × 2 |
| `full_ft` | Full-parameter fine-tuning | 500 | 8e-6 | 168 × 1 |
| `freeze_top2` | Freeze the bottom 10 layers; train the top 2 | 1000 | 1e-5 | 168 × 1 |
| `lora_r64` | LoRA r=64, alpha=128 | 1500 | 1.5e-5 | 78 × 3 |
| `bitfit` | Train only biases and lm_head | 5000 | 2e-4 | 90 × 2 |

Original directories:

- `lora_r16`: `length_expert_model1_six_attr_lora_notraineval_20260609_101447/checkpoint-2500`
- `full_ft`: `six_attr_full_ft_20260707_033616/checkpoint-500`
- `freeze_top2`: `six_attr_freeze_top2_20260707_095800/checkpoint-1000`
- `lora_r64`: `six_attr_lora_r64_20260707_095800/checkpoint-1500`
- `bitfit`: `six_attr_bitfit_20260707_095800/checkpoint-5000`

Retraining: `scripts/train_six_attr_finetune.sh`. Generation: `scripts/generate_six_attr_finetune.sh`. Evaluation: `scripts/eval_six_attr_finetune.sh`, using the protocol in `src/multi_attr/07.eval_checkpoint_protocol_five_attr_single_lora.py`. Attribute scoring uses the five GP, GN, FU, SF, and MIC discriminators with expert-model backbones. SF label 1 means toxic. MIC is recorded only as a five-bin distribution and is not included in the four-bit combination condition.

## Discriminators

All backbones use `models/AMPGuide`. Weights originate from `best.pt` in `six_attr_discriminators_hybrid` and are identical to the five heads subsequently packaged as the English-language classifiers.

- `models/discriminators/gram_positive`: GP, binary classification, LoRA r=32, selected by validation AUROC
- `models/discriminators/gram_negative`: GN
- `models/discriminators/antifungal`: FU
- `models/discriminators/safety`: SF, 0 = low toxicity, 1 = toxic
- `models/discriminators/mic`: Five MIC bins. Input consists of atomic `SPxx` tokens and amino-acid IDs. The tokenizer is in `models/discriminators/mic_tokenizer`.

Each head directory contains:

- `weights/best.pt`: Weights
- `backbone`: Points to `models/AMPGuide`
- `tokenizer`: Binary heads use the AMPGuide tokenizer; MIC uses `mic_tokenizer`

`amp.pt` is excluded. That discriminator uses a PeptidePrior backbone.

## Corpora

- `data/corpus/uniprot_short_peptides.txt`: PeptidePrior, 3,219,459 lines
- `data/corpus/AMP.txt`: AMPGuide, 16,062 lines
- `data/corpus/nonAMP.txt`: CountAMPGuide, 2,821,399 lines
- `data/corpus/sequence_species_mic_binned_for_six_attr.csv`: Six-attribute experts and discriminators
- `data/splits/split_{train,val,test}.csv`: Sequence-based splits used by hybrid training, seed=42

Length-conditioned text and indexed splits are written to `outputs/prepared/` by `scripts/train_PeptidePrior.sh`, `scripts/train_AMPGuide.sh`, and `scripts/train_CountAMPGuide.sh`, using seed=42 and validation/test ratios of 0.001 each. These files can be regenerated and are not included in the repository.

## Experiments excluded from the release

Excluded weights include lowtox full-parameter fine-tuning, union20k, five-attribute LoRA, and earlier discriminator v2/v3/fullft models. For each of the five six-attribute fine-tuning strategies, only the checkpoint listed above is retained; intermediate checkpoints saved every 500 steps are excluded. Earlier discriminator training recipes remain in `src/discriminators/` and can be retrained with outputs written to `outputs/`.
