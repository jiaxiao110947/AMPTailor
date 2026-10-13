"""Test decoding loops with real tensors and deterministic toy model logits.

Only the functions under test are loaded, avoiding pretrained weights and HF imports.
Run: python -m unittest discover -s tests -v
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    namespace['torch'] = torch
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), relative, 'exec'), namespace)
    return namespace


class Tokenizer:
    eos_token_id = 1

    def __call__(self, prompt, **kwargs):
        return {'input_ids': [0]}


class Model:
    def __init__(self, lengths):
        self.lengths = lengths

    def __call__(self, input_ids):
        step = input_ids.shape[1] - 1
        z = torch.full((*input_ids.shape, 2), -1000.0)
        for row, length in enumerate(self.lengths):
            z[row, -1, int(step >= length)] = 1000.0
        return SimpleNamespace(logits=z)


LIB = SimpleNamespace(
    _build_aa_char_maps=lambda tok: ({0: 'A'}, torch.tensor([0, 1])),
    _top_k_top_p_filtering=lambda z, **kw: z,
    parse_condition=lambda combo, **kw: (combo.split(':')[0], int(combo.split(':')[1])),
)
BIP = SimpleNamespace(
    parse_attr_weights=lambda text: [1] * 5,
    task_weight=lambda task, weights: 1,
    make_task_prompt=lambda task: task,
)


class StoppingTests(unittest.TestCase):
    def run_batch(self, kind, lengths, cap=10):
        ns = {}
        common = dict(temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                      lib=LIB, use_bf16=False, max_new_tokens=cap)
        if kind == 'strict':
            load_functions('src/generation/generate_strict_combo.py', ['generate_until_eos'], ns)
            return ns['generate_until_eos'](Model(lengths), Tokenizer(), torch.device('cpu'),
                                            'LEN2:', 2, len(lengths), **common)
        load_functions('src/generation/generate_bipolar_fusion.py',
                       ['_fwd', '_unique_tasks', 'generate_batch'], ns)
        return ns['generate_batch'](Model(lengths), Tokenizer(), torch.device('cpu'),
                                     '1110:2', len(lengths), kind=kind, alpha=0.0,
                                     attr_weights='1,1,1,1,1', bip=BIP, **common)[2]

    def test_independent_eos_and_lengths_beyond_target(self):
        for kind in ['strict', 'mix', 'contrast']:
            with self.subTest(kind=kind):
                self.assertEqual(self.run_batch(kind, [0, 1, 5]), ['', 'A', 'AAAAA'])

    def test_safety_cap_without_eos(self):
        for kind in ['strict', 'mix', 'contrast']:
            with self.subTest(kind=kind):
                self.assertEqual(self.run_batch(kind, [100], cap=4), ['AAAA'])

    def test_shared_decoder_first_eos_ignores_minimum_preference(self):
        ns = dict(
            _build_aa_char_maps=LIB._build_aa_char_maps,
            fused_combo_logits=lambda generated_ids, **kw: torch.tensor([[-1000., 1000.]]),
            _apply_repetition_penalty=lambda z, *args: z,
            _apply_no_repeat_ngram=lambda z, *args: z,
            _top_k_top_p_filtering=LIB._top_k_top_p_filtering,
            _sample_next=lambda z, **kw: z.argmax(-1, keepdim=True),
        )
        load_functions('src/multi_attr/07.five_attr_classification_lora_lib.py',
                       ['_decode_sequence_once'], ns)
        seq, reason = ns['_decode_sequence_once'](
            None, Tokenizer(), torch.device('cpu'), '1110:18', 'fused',
            18, 30, False, 10, 1.0, 1.0, 0, 1.0, 0, True, None, 18)
        self.assertEqual((seq, reason), ('', 'stop'))


if __name__ == '__main__':
    unittest.main()
