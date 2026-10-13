# Literal EOS and retained-checkpoint compatibility

Training sequences now use the literal special token `EOS`, for example:

```text
LEN18:WVRLGRYLLRRLKTPFTREOS
```

`EOS` is recognized as a single special token. It is not three ordinary letters
and is not part of the peptide residues.

## Existing model weights

`src/eos_compat.py` renames the tokenizer's original end token while retaining its
exact token ID and vocabulary size. For the previously uploaded GPT-2 vocabulary,
that ID is 50256. The embedding and output-head rows keep their original meanings;
model weights do not need resizing or retraining for this rename.

All repository tokenizer-loading paths use this compatibility loader. Old
checkpoint tokenizer files can remain on disk: loading returns an actual `EOS`
tokenizer, and saving it writes the migrated vocabulary and special-token metadata.
EOS, padding, BOS, and unknown-token aliases sharing the old ID are updated
consistently. Other special tokens, such as atomic species tokens, retain their IDs.

If `EOS` already occupies a different vocabulary ID, migration stops with an
explicit error instead of silently remapping model weights. Such tokenizers need
an individually reviewed migration.

## Legacy corpora and indexes

The original end marker is still accepted when reading old corpora. It is
normalized to `EOS` before tokenization and is not counted as peptide residues.
Newly prepared text uses `EOS`. Existing byte-offset indexes can read legacy files
without rewriting the files or shifting their offsets.

Regenerating split indexes can change historical hash assignments because the
normalized text representation has changed. Preserve frozen split files/indexes
when reproducing a published partition; do not rewrite an indexed corpus in place
without rebuilding its byte offsets.

## Export a migrated tokenizer

To persist a migrated tokenizer separately:

```bash
python src/eos_compat.py \
  --tokenizer_dir models/AMPGuide \
  --output_dir outputs/tokenizers/AMPGuide_EOS
```

Use a distinct empty output directory. The original tokenizer and model weights
are preserved. A plain Hugging Face tokenizer loader can then load the exported
files with `eos_token="EOS"` and the original end-token ID.

This migration does not upload or overwrite the original Hugging Face resources.
Existing models retain natural EOS stopping; requested peptide length remains a
conditioning prompt rather than a forced length.
