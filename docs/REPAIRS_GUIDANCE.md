# Guided generation and evaluation repairs

These repairs preserve the model architecture, adapter layout, generation formulas,
and existing shell entry points. Generation still stops at the first sampled EOS;
target length is a conditioning prompt, not a forced output length.

## Guided decoding

The decoder now accepts and applies repetition penalty, repetition window,
homopolymer limits, and no-repeat n-gram limits. Defaults disable the new
constraints for callers that do not supply them. Constraints operate on generated
residues only and do not penalize EOS or prompt/padding tokens.

Left-padded batches use the last prompt column for next-token logits. Generated
output starts after the padded input width. Forbidden logits remain negative
infinity; invalid or empty sampling distributions raise an explicit error rather
than silently emitting an artificial EOS. Existing `keep_eos_after_filter` behavior
remains an explicit caller option.

Length-conditioned collection has a finite attempt budget (default: ten times the
requested sample count). If empty outputs exhaust that budget, the caller receives
the collected sequences and a warning reporting the shortfall and attempts.

## Independent attribute adapters

The optimizer includes the deduplicated union of trainable parameters from all five
adapters (`gp`, `gn`, `fu`, `sf`, `len`), including adapter-specific saved output heads.
Collecting parameters restores the original active adapter and does not unfreeze
the base model. Gradient clipping includes parameters with accumulated gradients.
The final partial accumulation window is normalized and applied before final export.

## Classifier positions

All six classifier implementations derive content positions from the attention
mask, so real tokens begin at position zero regardless of left-padding width.
Class names and checkpoint parameter keys remain unchanged. Existing checkpoints
can still be loaded, but their scores must be re-evaluated with the corrected
position handling; historical scores are not guaranteed to remain unchanged.

## Requested-attribute metrics

Unspecified `X` attributes and absent target lengths remain neutral in joint
condition checks. They are excluded from individual satisfaction numerators and
denominators. `per_attr_n` reports the actual requested sample count for each
attribute. Attributes with no requested samples are omitted from satisfaction
maps; the average includes only attributes with a nonzero denominator.

The checkpoint protocol and bipolar training evaluator use these same rules. The
bipolar conversion uses each attribute's own denominator, not the total row count.
Scoring remains based on the generated/retained sequence set; these repairs do not
turn satisfaction rates into yields per original generation budget.

## Validation and existing artifacts

Regression tests cover natural stopping, mixed prompt widths, hard masks,
repetition constraints, bounded retries, all-adapter parameter updates, gradient
accumulation, all six classifier position behaviors, and metric conversion.
Micro-model integration checks exercise the guided shell entry point and the
actual multi-adapter training/export loop.

Existing independent-adapter weights are not retroactively repaired by changes to
the optimizer. Re-train affected adapters to apply the corrected training logic.
Re-score existing candidates to apply corrected classifier positions and metric
denominators. Published model quality and full CUDA execution require separate
validation with the actual weights and data.
