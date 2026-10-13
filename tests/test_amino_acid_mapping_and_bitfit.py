"""Regression tests for amino-acid coverage and tied BitFit output weights."""
import re
import unittest

import torch
from test_generation_stopping import load_functions

AA20 = 'ACDEFGHIKLMNPQRSTVWY'


class Tokenizer:
    eos_token_id = 0

    def __init__(self, tokens, base_size=None):
        self.tokens = ['<|endoftext|>'] + tokens
        self.vocab_size = len(self.tokens) if base_size is None else base_size

    def __len__(self):
        return len(self.tokens)

    def decode(self, ids, **kwargs):
        return self.tokens[ids[0]]


class ToyLM(torch.nn.Module):
    def __init__(self, tied):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(3, 4)
        self.transformer.projection = torch.nn.Linear(4, 4)
        self.lm_head = torch.nn.Linear(4, 3, bias=False)
        if tied:
            self.lm_head.weight = self.transformer.wte.weight

    def get_output_embeddings(self):
        return self.lm_head


class MappingAndBitFitTests(unittest.TestCase):
    def setUp(self):
        self.maps = load_functions(
            'src/multi_attr/07.five_attr_classification_lora_lib.py',
            ['_build_aa_char_maps'], {'AA20_LIST': list(AA20), 'AA20_SET': set(AA20)})
        self.ft = load_functions('src/multi_attr/07.five_attr_finetune_lib.py',
                                 ['_gpt2_block_idx', 'apply_finetune_mode'], {'re': re})

    def test_duplicate_tokens_do_not_hide_missing_residues(self):
        tok = Tokenizer([c for aa in AA20[:10] for c in (aa, aa.lower())])
        with self.assertRaisesRegex(RuntimeError, 'M.*N.*P.*Q.*R.*S.*T.*V.*W.*Y'):
            self.maps['_build_aa_char_maps'](tok)

    def test_all_twenty_residues_with_duplicates_and_eos(self):
        tok = Tokenizer([c for aa in AA20 for c in (aa, aa.lower())])
        mapping, allowed = self.maps['_build_aa_char_maps'](tok)
        self.assertEqual(set(mapping.values()), set(AA20))
        self.assertEqual(len(mapping), 20)
        self.assertEqual(len(allowed), 21)
        for token_id, aa in mapping.items():
            self.assertEqual(tok.decode([token_id]), aa)
        self.assertIn(tok.eos_token_id, allowed.tolist())
        self.assertNotIn(tok.eos_token_id, mapping)

    def test_uppercase_is_preferred_even_when_lowercase_comes_first(self):
        tok = Tokenizer(list(AA20.lower()) + list(AA20))
        mapping, _ = self.maps['_build_aa_char_maps'](tok)
        for token_id, aa in mapping.items():
            self.assertEqual(tok.decode([token_id]), aa)
            self.assertGreater(token_id, 20)

    def test_lowercase_fallback_without_uppercase_tokens(self):
        tok = Tokenizer(list(AA20.lower()))
        mapping, allowed = self.maps['_build_aa_char_maps'](tok)
        self.assertEqual(set(mapping.values()), set(AA20))
        self.assertEqual(len(allowed), 21)

    def test_added_amino_acid_tokens_are_scanned(self):
        tok = Tokenizer(list(AA20), base_size=11)
        mapping, _ = self.maps['_build_aa_char_maps'](tok)
        self.assertEqual(set(mapping.values()), set(AA20))

    def test_tied_and_untied_heads_receive_gradients(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = ToyLM(tied)
                self.ft['apply_finetune_mode'](model, 'bitfit')
                self.assertTrue(model.lm_head.weight.requires_grad)
                self.assertTrue(model.transformer.projection.bias.requires_grad)
                self.assertFalse(model.transformer.projection.weight.requires_grad)
                self.assertEqual(model.transformer.wte.weight.requires_grad, tied)
                model.lm_head(torch.ones(1, 4)).sum().backward()
                self.assertIsNotNone(model.lm_head.weight.grad)
                if tied:
                    self.assertIs(model.lm_head.weight, model.transformer.wte.weight)
                    self.assertIs(model.lm_head.weight.grad, model.transformer.wte.weight.grad)


if __name__ == '__main__':
    unittest.main()
