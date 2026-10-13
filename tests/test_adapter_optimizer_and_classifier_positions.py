"""Real tiny GPT-2/PEFT tests for optimizer membership and padding invariance."""
import tempfile
import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from test_guided_generation import load

ADAPTER = load('adapter_training_test', 'src/multi_attr/train_disentangled_attr_experts.py')


def config():
    return GPT2Config(vocab_size=24, n_positions=32, n_embd=8, n_layer=1, n_head=1,
                      attn_pdrop=0., resid_pdrop=0., embd_pdrop=0.)


def adapter_model():
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
                     target_modules=['c_attn'], modules_to_save=['lm_head'], fan_in_fan_out=True)
    m = get_peft_model(GPT2LMHeadModel(config()), cfg, adapter_name='gp')
    for name in ['gn', 'fu', 'sf', 'len']:
        m.add_adapter(name, cfg)
    m.set_adapter('gp')
    return m


class AdapterOptimizerTests(unittest.TestCase):
    def test_all_adapters_update_and_base_stays_frozen(self):
        m = adapter_model()
        params = ADAPTER._params_for_all_adapters(m)
        self.assertEqual(m.active_adapter, 'gp')
        self.assertEqual(len(params), len({id(p) for p in params}))
        member_ids = {id(p) for p in params}
        frozen_before = m.get_base_model().transformer.wte.weight.detach().clone()
        opt = torch.optim.AdamW(params, lr=.01, weight_decay=0.)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        for name in ['gp', 'gn', 'fu', 'sf', 'len']:
            m.set_adapter(name)
            active = [p for p in m.parameters() if p.requires_grad]
            self.assertTrue(all(id(p) in member_ids for p in active))
            before = [p.detach().clone() for p in active]
            ids = torch.tensor([[1, 2, 3, 4]])
            m(input_ids=ids, labels=ids).loss.backward()
            ADAPTER._step_adapter_optimizer(params, opt, sched, 1, 1)
            self.assertTrue(any(not torch.equal(old, p.detach()) for old, p in zip(before, active)), name)
        self.assertTrue(torch.equal(frozen_before, m.get_base_model().transformer.wte.weight))

    def test_accumulation_updates_both_active_adapters(self):
        m = adapter_model()
        params = ADAPTER._params_for_all_adapters(m)
        opt = torch.optim.AdamW(params, lr=.01, weight_decay=0.)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        snapshots = {}
        for name in ['gp', 'gn']:
            m.set_adapter(name)
            snapshots[name] = [(p, p.detach().clone()) for p in m.parameters() if p.requires_grad]
            ids = torch.tensor([[1, 2, 3, 4]])
            (m(input_ids=ids, labels=ids).loss / 2).backward()
        ADAPTER._step_adapter_optimizer(params, opt, sched, 2, 2)
        for name, pairs in snapshots.items():
            self.assertTrue(any(not torch.equal(p.detach(), before) for p, before in pairs), name)
        self.assertTrue(all(p.grad is None for p in params))

    def test_partial_final_window_is_flushed_and_normalized(self):
        p = torch.nn.Parameter(torch.tensor(0.))
        opt = torch.optim.SGD([p], lr=1.)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        (p / 4).backward()
        ADAPTER._step_adapter_optimizer([p], opt, sched, 1, 4)
        self.assertAlmostEqual(float(p.detach()), -1., places=5)
        self.assertIsNone(p.grad)
        self.assertEqual(sched.last_epoch, 1)


class ClassifierPositionTests(unittest.TestCase):
    def test_all_six_classifier_classes_ignore_left_padding_width(self):
        disc = 'src/discriminators/'
        modules = {name: load('position_test_' + name, disc + name + '.py') for name in [
            'train_six_attr_discriminators', 'train_six_attr_discriminators_v2',
            'train_six_attr_discriminators_v3', 'train_six_attr_discriminators_fullft',
            'train_six_attr_discriminators_lora_atomic', 'train_six_attr_discriminators_hybrid']}
        with tempfile.TemporaryDirectory() as folder:
            GPT2Model(config()).save_pretrained(folder)
            factories = [
                lambda: modules['train_six_attr_discriminators'].MultiHeadDiscriminator(folder, ('GP',), 0.),
                lambda: modules['train_six_attr_discriminators_v2'].LoraDiscriminator(folder, 2, 0., 2, 4, 0.),
                lambda: modules['train_six_attr_discriminators_v3'].V3Discriminator(folder, 2, False, 0., 2, 4, 0., 0),
                lambda: modules['train_six_attr_discriminators_fullft'].FullFTDisc(GPT2Model(config()), 2, 0.),
                lambda: modules['train_six_attr_discriminators_lora_atomic'].AtomicLoraDisc(GPT2Model(config()), 2, 0., 2, 4, 0., 16),
                lambda: modules['train_six_attr_discriminators_hybrid'].MicLoraDisc(GPT2Model(config()), 0., 2, 4, 0., 16),
            ]
            for factory in factories:
                model = factory().eval()
                with self.subTest(model=type(model).__name__):
                    results = []
                    for ids, mask in [([[1, 2, 3]], [[1, 1, 1]]),
                                      ([[0, 0, 0, 0, 1, 2, 3]], [[0, 0, 0, 0, 1, 1, 1]])]:
                        kw = {}
                        if type(model).__name__ == 'V3Discriminator':
                            kw = dict(aa_ids=torch.tensor([[1, 2, 3]]), aa_mask=torch.ones(1, 3),
                                      phys=torch.zeros(1, 8), species_ids=torch.tensor([20]))
                        with torch.no_grad():
                            result = model(torch.tensor(ids), torch.tensor(mask), **kw)
                        results.append(result['GP'] if isinstance(result, dict) else result)
                    torch.testing.assert_close(*results, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
