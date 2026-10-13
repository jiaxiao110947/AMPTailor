"""Preserve real LEN residues and remove only explicit echoed control markers."""
import re
import unittest
from test_generation_stopping import load_functions


class LengthPromptExtractionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions('src/pretrain/amp_generate_lib.py',
                                 ['_clean_seq', 'drop_echoed_len_control',
                                  'extract_peptide_after_length_prompt'],
                                 {'re': re, 'AA20': 'ACDEFGHIKLMNPQRSTVWY',
                                  'EOS_TEXT': '<|endoftext|>'})

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


if __name__ == '__main__':
    unittest.main()
