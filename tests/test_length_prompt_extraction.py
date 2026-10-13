"""Preserve real LEN residues and remove only explicit echoed control markers."""
import re
import unittest
from test_generation_stopping import load_functions
from test_guided_generation import load

COMPAT = load("extraction_eos_compat", "src/eos_compat.py")


class LengthPromptExtractionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions('src/pretrain/amp_generate_lib.py',
                                 ['_clean_seq', 'drop_echoed_len_control',
                                  'extract_peptide_after_length_prompt'],
                                 {'re': re, 'AA20': 'ACDEFGHIKLMNPQRSTVWY',
                                  'EOS_TEXT': 'EOS', 'normalize_eos_text': COMPAT.normalize_eos_text})

    def test_real_len_residues_survive(self):
        extract = self.ns['extract_peptide_after_length_prompt']
        for sequence in ['LENACDEF', 'LENLENAC', 'LEN', 'LENELEELQA', 'ACDEF']:
            with self.subTest(sequence=sequence):
                self.assertEqual(extract('LEN18:' + sequence, 'LEN18:'), sequence)

    def test_explicit_echoes_removed_before_residue_cleaning(self):
        extract = self.ns['extract_peptide_after_length_prompt']
        self.assertEqual(extract('LEN18:LEN18:len4:ACDE', 'LEN18:'), 'ACDE')
        self.assertEqual(extract('LEN18:LEN10:LENELEELQA', 'LEN18:'), 'LENELEELQA')

    def test_first_eos_ends_sequence(self):
        extract = self.ns['extract_peptide_after_length_prompt']
        self.assertEqual(extract('LEN4:ACDE<|endoftext|>WYWY', 'LEN4:'), 'ACDE')
        self.assertEqual(extract('LEN4:<|endoftext|>ACDE', 'LEN4:'), '')
        self.assertEqual(extract('LEN4:ACDEEOSWYWY', 'LEN4:'), 'ACDE')


if __name__ == '__main__':
    unittest.main()
