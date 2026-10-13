"""Exercise the real guided decoder with deterministic logits, no published weights."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

GUIDED = load('guided_under_test', 'src/generation/generate_amptailor_amp.py')
LENGTH = load('guided_length_under_test', 'src/generation/generate_amptailor_length_amp.py')


class Tok:
    pad_token_id = eos_token_id = 2

    def __call__(self, prompts, **kw):
        width = max(map(len, prompts))
        rows = [[2] * (width - len(p)) + [4] * len(p) for p in prompts]
        masks = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
        return {'input_ids': torch.tensor(rows), 'attention_mask': torch.tensor(masks)}

    def decode(self, ids, **kw):
        return ''.join({0: 'A', 1: 'C', 2: '', 3: 'Z', 4: 'Q'}[i] for i in ids)


class ScriptedModel:
    def __init__(self, plans=None, repeat=False, extreme=False):
        self.width = None
        self.plans = plans
        self.repeat = repeat
        self.extreme = extreme

    def __call__(self, input_ids, attention_mask, **kw):
        if self.width is None:
            self.width = input_ids.shape[1]
        step = attention_mask.shape[1] - self.width
        batch, width = input_ids.shape
        z = torch.full((batch, width, 5), -1000.)
        z[:, :, 2] = 1000.  # Wrong first-step column would immediately stop.
        z[:, -1] = -1000.
        if self.extreme:
            z[:, -1, :3] = -20000.
            z[:, -1, 3:] = 5000.
        elif self.repeat:
            z[:, -1, 0] = 2.
            z[:, -1, 1] = 1.
        else:
            for row, plan in enumerate(self.plans):
                z[row, -1, plan[step] if step < len(plan) else 2] = 1000.
        return SimpleNamespace(logits=z, past_key_values=object())


def generator(model):
    obj = GUIDED.AMPTailorGeneratorAAOnly.__new__(GUIDED.AMPTailorGeneratorAAOnly)
    obj.device = torch.device('cpu')
    obj.tok = Tok()
    obj.eos_id = 2
    obj.base = obj.expert = obj.anti = model
    obj.allowed_token_mask = torch.tensor([True, True, True, False, False])
    obj.aa_no_eos = torch.tensor([True, True, False, False, False])
    obj.rng = torch.Generator().manual_seed(42)
    return obj


def generate(gen, prompts=('Q',), **kw):
    options = dict(max_new_tokens=8, min_new_tokens=0, alpha=0., alpha_eos=0.,
                   temperature=1., top_k=0, top_p=1., keep_eos_after_filter=True)
    options.update(kw)
    return gen.generate_batch(list(prompts), **options)


class GuidedTests(unittest.TestCase):
    def test_mixed_prompt_width_and_independent_eos(self):
        g = generator(ScriptedModel([[0, 2, 1], [1, 0, 0, 2, 1]]))
        self.assertEqual(generate(g, ('Q', 'QQ')), ['QA', 'QQCAA'])

    def test_natural_length_can_exceed_requested_length(self):
        g = generator(ScriptedModel([[0] * 5 + [2]]))
        self.assertEqual(generate(g, ('LEN2:',)), ['LEN2:AAAAA'])

    def test_hard_masks_remain_zero_even_with_extreme_logits(self):
        g = generator(ScriptedModel(extreme=True))
        original = torch.multinomial
        probabilities = []
        def sample(p, *args, **kw):
            probabilities.append(p.clone())
            return original(p, *args, **kw)
        with patch.object(torch, 'multinomial', side_effect=sample):
            generate(g, max_new_tokens=2)
        self.assertTrue(probabilities)
        for p in probabilities:
            self.assertTrue(torch.equal(p[:, 3:], torch.zeros_like(p[:, 3:])))

    def test_repetition_penalty_is_applied(self):
        for penalty, expected in [(1., 'QAA'), (4., 'QAC')]:
            g = generator(ScriptedModel(repeat=True))
            with patch.object(torch, 'multinomial', side_effect=lambda p, **kw: p.argmax(-1, keepdim=True)):
                self.assertEqual(generate(g, max_new_tokens=2, repetition_penalty=penalty), [expected])

    def test_homopolymer_and_ngram_limits_are_applied(self):
        for option in [{'max_homopolymer_run': 2}, {'no_repeat_ngram_size': 2}]:
            g = generator(ScriptedModel(repeat=True))
            with patch.object(torch, 'multinomial', side_effect=lambda p, **kw: p.argmax(-1, keepdim=True)):
                self.assertEqual(generate(g, max_new_tokens=4, **option), ['QAACA'])

    def test_eos_is_not_banned_by_repetition_constraints(self):
        g = generator(ScriptedModel([[0, 2], [0, 0, 1, 2]]))
        self.assertEqual(generate(g, ('Q', 'QQ'), repetition_penalty=2.,
                                  max_homopolymer_run=2, no_repeat_ngram_size=1), ['QA', 'QQAC'])

    def test_empty_sampling_distribution_raises_instead_of_forcing_eos(self):
        g = generator(ScriptedModel(repeat=True))
        with self.assertRaisesRegex(RuntimeError, 'sampling distribution'):
            generate(g, max_new_tokens=3, top_k=1, keep_eos_after_filter=False, no_repeat_ngram_size=1)

    def test_empty_outputs_have_bounded_attempts(self):
        class Empty:
            attempts = 0
            def generate_batch(self, prompts, **kw):
                self.attempts += len(prompts)
                return prompts
        g = Empty()
        seqs, targets = LENGTH.generate_length_cond(
            g, target_length=4, nsamples=3, max_new_tokens=8, alpha=0., alpha_eos=0.,
            temperature=1., top_k=0, top_p=1., batch_size=2, keep_eos_after_filter=True,
            repetition_penalty=1., rep_window=64, max_homopolymer_run=0,
            no_repeat_ngram_size=0, max_attempts_multiplier=2)
        self.assertEqual((seqs, targets, g.attempts), ([], [], 6))


if __name__ == '__main__':
    unittest.main()
