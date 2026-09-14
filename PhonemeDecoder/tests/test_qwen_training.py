import json
import tempfile
import unittest
from pathlib import Path

from PhonemeDecoder.train_qwen import candidate_prompt, training_rows


class TrainingDataTests(unittest.TestCase):
    def record(self):
        return dict(clip_id="s1:example", speaker_id=1, split="train",
                    greedy_visual_phones=["B", "AE", "T"],
                    reference_phones=["SECRET"], reference_words=["bat"],
                    candidates=[dict(words=["bat"], text="bat")])

    def test_prompt_does_not_leak_reference(self):
        record = self.record()
        before = candidate_prompt(record)
        record["reference_words"] = ["hidden"]
        record["reference_phones"] = ["hidden"]
        self.assertEqual(before, candidate_prompt(record))
        self.assertNotIn("SECRET", before)

    def test_absent_reference_skipped_and_heldout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            present = self.record()
            absent = dict(present, clip_id="s1:absent", reference_words=["pat"])
            path.write_text("\n".join(map(json.dumps, [present, absent])))
            rows, skipped = training_rows(path)
            self.assertEqual((len(rows), skipped), (1, 1))
            self.assertEqual(rows[0]["target"], "bat")
            path.write_text(json.dumps(dict(present, speaker_id=9)))
            with self.assertRaises(ValueError):
                training_rows(path)
