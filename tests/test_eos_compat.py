"""Literal EOS uses the original end-token ID and keeps retained weights compatible."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from test_guided_generation import load, ROOT

COMPAT = load('eos_compat_tests', 'src/eos_compat.py')


def original_tokenizer(collision=False):
    vocab = {c: i for i, c in enumerate('ACDEFGHIKLMNPQRSTVWY0123456789:')}
    if collision:
        vocab['EOS'] = len(vocab)
    vocab['<|endoftext|>'] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<|endoftext|>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    t = GPT2TokenizerFast(tokenizer_object=backend, eos_token='<|endoftext|>',
                          bos_token='<|endoftext|>', unk_token='<|endoftext|>')
    t.pad_token = t.eos_token
    return t


class EOSCompatibilityTests(unittest.TestCase):
    def test_actual_special_token_renamed_without_new_ids(self):
        tok = original_tokenizer()
        old_id, old_size = tok.eos_token_id, len(tok)
        COMPAT.migrate_eos_tokenizer(tok)
        self.assertEqual(tok.eos_token, 'EOS')
        self.assertEqual(tok.eos_token_id, old_id)
        self.assertEqual(len(tok), old_size)
        self.assertEqual(tok.encode('EOS', add_special_tokens=False), [old_id])
        self.assertEqual(tok.encode('ACDEEOS', add_special_tokens=False)[-1], old_id)
        self.assertEqual(tok.decode([old_id]), 'EOS')
        self.assertEqual(tok.decode([old_id], skip_special_tokens=True), '')
        self.assertEqual(tok.pad_token_id, old_id)
        self.assertEqual(tok.bos_token_id, old_id)
        self.assertEqual(tok.unk_token_id, old_id)
        self.assertEqual(tok.encode('?', add_special_tokens=False), [old_id])

    def test_exported_files_and_plain_hf_reload_use_eos(self):
        with tempfile.TemporaryDirectory() as folder:
            tok = COMPAT.migrate_eos_tokenizer(original_tokenizer())
            tok.save_pretrained(folder)
            restored = GPT2TokenizerFast.from_pretrained(folder, local_files_only=True)
            self.assertEqual(restored.eos_token, 'EOS')
            self.assertEqual(restored.eos_token_id, tok.eos_token_id)
            self.assertEqual(restored.encode('EOS', add_special_tokens=False), [tok.eos_token_id])
            self.assertNotIn('<|endoftext|>', (Path(folder) / 'tokenizer.json').read_text())
            self.assertNotIn('<|endoftext|>', (Path(folder) / 'tokenizer_config.json').read_text())
            self.assertIs(COMPAT.migrate_eos_tokenizer(restored), restored)

    def test_model_logits_and_weights_are_unchanged(self):
        tok = original_tokenizer()
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=32,
                                          n_embd=8, n_layer=1, n_head=1)).eval()
        before_ids = tok.encode('LEN4:ACDE<|endoftext|>', add_special_tokens=False)
        weights = model.get_input_embeddings().weight.detach().clone()
        with torch.no_grad():
            before = model(torch.tensor([before_ids])).logits
        COMPAT.migrate_eos_tokenizer(tok)
        after_ids = tok.encode('LEN4:ACDEEOS', add_special_tokens=False)
        self.assertEqual(before_ids, after_ids)
        with torch.no_grad():
            after = model(torch.tensor([after_ids])).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertTrue(torch.equal(weights, model.get_input_embeddings().weight))

    def test_species_special_tokens_keep_their_ids(self):
        tok = original_tokenizer()
        tok.add_special_tokens({'additional_special_tokens': ['SP00', 'SP01']})
        before = [tok.convert_tokens_to_ids(t) for t in ['SP00', 'SP01']]
        size = len(tok)
        COMPAT.migrate_eos_tokenizer(tok)
        self.assertEqual(len(tok), size)
        self.assertEqual([tok.convert_tokens_to_ids(t) for t in ['SP00', 'SP01']], before)
        self.assertEqual(tok.additional_special_tokens, ['SP00', 'SP01'])

    def test_additional_alias_of_legacy_eos_is_exported_consistently(self):
        tok = original_tokenizer()
        tok.add_special_tokens({'additional_special_tokens': ['<|endoftext|>', 'SP00']})
        old_id, size = tok.eos_token_id, len(tok)
        COMPAT.migrate_eos_tokenizer(tok)
        with tempfile.TemporaryDirectory() as folder:
            tok.save_pretrained(folder)
            restored = GPT2TokenizerFast.from_pretrained(folder)
            self.assertEqual(len(restored), size)
            self.assertEqual(restored.eos_token_id, old_id)
            self.assertEqual(restored.additional_special_tokens, ['EOS', 'SP00'])

    def test_collision_is_rejected_without_mutation(self):
        tok = original_tokenizer(collision=True)
        before = tok.backend_tokenizer.to_str()
        with self.assertRaisesRegex(ValueError, 'different token ID'):
            COMPAT.migrate_eos_tokenizer(tok)
        self.assertEqual(tok.backend_tokenizer.to_str(), before)
        self.assertEqual(tok.eos_token, '<|endoftext|>')

    def test_legacy_corpus_and_index_are_compatible(self):
        prep = load('eos_index_prep', 'src/pretrain/prepare_indexed_splits.py')
        train = load('eos_model_train', 'src/pretrain/train_model.py')
        self.assertEqual(prep._maybe_add_eos('LEN4:ACDE<|endoftext|>', True), 'LEN4:ACDEEOS')
        self.assertEqual(train._maybe_add_eos('LEN4:ACDEEOS', True), 'LEN4:ACDEEOS')
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'legacy.txt'
            source.write_text('LEN4:ACDE<|endoftext|>\nLEN4:FGHI<|endoftext|>\n')
            out = Path(folder) / 'index'
            subprocess.run([sys.executable, str(ROOT / 'src/pretrain/prepare_indexed_splits.py'),
                            '--data_file', str(source), '--out_dir', str(out),
                            '--val_ratio', '0', '--test_ratio', '0', '--ensure_eos'],
                           check=True, capture_output=True)
            ds = train.IndexedLineDataset(str(source), str(out/'train_offsets.npy'), ensure_eos=True)
            self.assertEqual(ds[0]['text'], 'LEN4:ACDEEOS')
            self.assertEqual(ds[1]['text'], 'LEN4:FGHIEOS')

    def test_cli_migrates_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            out = Path(folder) / 'eos'
            original_tokenizer().save_pretrained(source)
            before = (source/'tokenizer.json').read_bytes()
            subprocess.run([sys.executable, str(ROOT/'src/eos_compat.py'),
                            '--tokenizer_dir', str(source), '--output_dir', str(out)],
                           check=True, capture_output=True)
            self.assertEqual((source/'tokenizer.json').read_bytes(), before)
            self.assertEqual(GPT2TokenizerFast.from_pretrained(out).eos_token, 'EOS')


if __name__ == '__main__':
    unittest.main()
