"""Regression tests for length-prefix preprocessing (standard library only)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "src/pretrain/prepare_length_train_data.py"
spec = importlib.util.spec_from_file_location("length_prep", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class LengthPrefixTests(unittest.TestCase):
    def test_prefix_is_not_part_of_peptide(self):
        for raw in ["LEN4:ACDE", "len4:acde<|endoftext|>",
                    " LEN4: A C D E ", "LEN7:LEN4:ACDE"]:
            with self.subTest(raw=raw):
                self.assertEqual(prep._clean_seq(raw), "ACDE")

    def test_real_residues_are_preserved(self):
        self.assertEqual(prep._clean_seq("LENACDE"), "LENACDE")
        self.assertEqual(prep._clean_seq("LEN7:LENACDE"), "LENACDE")

    def test_cli_recomputes_length_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            output = Path(folder) / "output.txt"
            repeated = Path(folder) / "repeated.txt"
            source.write_text("LEN4:ACDE\nLEN18:ACDE<|endoftext|>\nACDE\nLEN4:<|endoftext|>\n")
            for inp, out in [(source, output), (output, repeated)]:
                subprocess.run([sys.executable, str(SCRIPT), "--input", str(inp),
                                "--output", str(out), "--ensure_eos"],
                               check=True, capture_output=True, text=True)
            self.assertEqual(output.read_text(), "LEN4:ACDE<|endoftext|>\n" * 3)
            self.assertEqual(output.read_bytes(), repeated.read_bytes())
            stats = json.loads(output.with_suffix(".stats.json").read_text())
            self.assertEqual(stats["length_histogram"], {"4": 3})
            self.assertEqual(stats["n_drop"], 1)


if __name__ == "__main__":
    unittest.main()
