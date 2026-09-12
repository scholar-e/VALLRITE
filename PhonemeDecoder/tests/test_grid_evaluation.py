"""Unit tests for GRID decoder data generation helpers."""
from pathlib import Path
import tempfile
import unittest

from PhonemeDecoder.decoder import Lexicon
from PhonemeDecoder.evaluate_grid import build_lexicon, edit_distance


class GridEvaluationTests(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance(["bin", "blue"], ["bin", "blue"]), 0)
        self.assertEqual(edit_distance(["bin", "blue"], ["bin"]), 1)
        self.assertEqual(edit_distance(["bin"], ["pin", "now"]), 2)

    def test_dictionary_conversion_merges_and_normalizes_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.dict"
            path.write_text("lettera EY1\nagain AH0 G EH1 N\nagain AH0 G EY1 N\n")
            document = build_lexicon(path)
        Lexicon(document)
        words = {word["id"]: word for word in document["words"]}
        self.assertEqual(words["lettera"]["text"], "a")
        self.assertEqual(len(words["again"]["pronunciations"]), 2)
        self.assertAlmostEqual(sum(p["prior"] for p in words["again"]["pronunciations"]), 1)
        self.assertEqual(words["lettera"]["pronunciations"][0]["phones"], ["EY"])


if __name__ == "__main__":
    unittest.main()
