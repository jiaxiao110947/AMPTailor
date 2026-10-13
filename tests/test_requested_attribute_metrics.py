"""Only explicitly requested attributes contribute to satisfaction denominators."""
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from test_guided_generation import load

SCORE = load('requested_attribute_score', 'src/generation/score_expert_discriminators.py')
PROTOCOL = load('requested_attribute_protocol', 'src/multi_attr/07.eval_checkpoint_protocol_five_attr_single_lora.py')
BIPOLAR = load('requested_attribute_bipolar', 'src/multi_attr/07.train_six_attr_bipolar_lora.py')


def row(bits='1XX0', length=18, target=18, gp=.9):
    return dict(id='seq0', bits=bits, length=length, target_len=target,
                p_gp=gp, p_gn=.01, p_fu=.01, p_sf=.1)


class RequestedAttributeTests(unittest.TestCase):
    def test_unspecified_attributes_are_not_successes(self):
        for gp, combo, mean in [(.9, 1., 1.), (.1, 0., 2/3)]:
            stats = SCORE.adherence_from_rows([row(gp=gp)], .5)
            self.assertEqual(stats['combo_adherence_mean'], combo)
            self.assertAlmostEqual(stats['attr_mean_satisfaction'], mean)
            self.assertNotIn('gn', stats['per_attr'])
            self.assertNotIn('fu', stats['per_attr'])
            self.assertEqual(stats['per_attr_n'], dict(gp=1, gn=0, fu=0, sf=1, length=1))

    def test_no_target_length_is_excluded(self):
        stats = SCORE.adherence_from_rows([row(target=None)], .5)
        self.assertNotIn('length', stats['per_attr'])
        self.assertEqual(stats['per_attr_n']['length'], 0)
        self.assertEqual(stats['attr_mean_satisfaction'], 1.)

    def test_mixed_conditions_have_individual_denominators(self):
        stats = SCORE.adherence_from_rows([row(), row(bits='X1X0')], .5)
        self.assertEqual(stats['per_attr_n'], dict(gp=1, gn=1, fu=0, sf=2, length=2))
        self.assertEqual(stats['per_attr']['gn'], 0.)

    def test_protocol_excludes_unrequested_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            fasta = Path(folder) / 'test.fa'
            fasta.write_text('>seq0|combo=1XX0:18|LEN=18\n' + 'A' * 18 + '\n')
            with patch.object(PROTOCOL, '_score_fasta_with_expert_discs', return_value={'rows': [row()]}):
                result = PROTOCOL.run_control_expert_fasta_eval(str(fasta), str(Path(folder) / 'out'))
            self.assertEqual(set(result['per_attr_satisfaction']), {'gram_pos', 'safety', 'length'})
            self.assertEqual(result['per_attr_n']['gram_neg'], 0)
            self.assertEqual(result['attr_mean_satisfaction'], 1.)

    def test_bipolar_callback_does_not_log_missing_attributes_as_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            payload_dir = Path(folder) / 'combo_eval_callback' / 'step_0000001' / 'regular6'
            payload_dir.mkdir(parents=True)
            (payload_dir / 'eval_payload.json').write_text(json.dumps({
                'per_attr_satisfaction': {'gram_pos': {'satisfaction': 1., 'n_common': 1.}}
            }))
            callback = BIPOLAR.ComboEvalEnglishDisc.__new__(BIPOLAR.ComboEvalEnglishDisc)
            callback.output_dir = Path(folder)
            callback._cache_combo_metrics = lambda metrics, step: None
            metrics = {}
            with patch.object(BIPOLAR.ComboEvalEnglishDisc.__bases__[0], 'on_evaluate', return_value=None):
                callback.on_evaluate(SimpleNamespace(), SimpleNamespace(global_step=1), None, metrics=metrics)
            self.assertEqual(metrics['eval_regular6_attr_gram_pos'], 1.)
            self.assertNotIn('eval_regular6_attr_gram_neg', metrics)
            self.assertNotIn('eval_regular6_attr_fungus', metrics)

    def test_bipolar_conversion_preserves_actual_denominators(self):
        class Tok:
            pad_token_id = 0
        stats = SCORE.adherence_from_rows([row(), row(bits='X1X0')], .5)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(BIPOLAR.SCORE, 'read_fasta', return_value=[('seq0', 'A'*18)]), \
                 patch.object(BIPOLAR.SCORE, 'make_loader', return_value=None), \
                 patch.object(BIPOLAR.SCORE, 'predict_p1', return_value=np.array([.9])), \
                 patch.object(BIPOLAR.SCORE, 'adherence_from_rows', return_value=stats):
                result = BIPOLAR.payload_from_english_fasta('unused', folder,
                            {name: object() for name in BIPOLAR.HEADS},
                            {name: Tok() for name in BIPOLAR.HEADS}, 'cpu')
            fields = result['per_attr_satisfaction']
            self.assertNotIn('fungus', fields)
            self.assertEqual(fields['gram_pos']['n_common'], 1.)
            self.assertEqual(fields['safety']['n_common'], 2.)


if __name__ == '__main__':
    unittest.main()
