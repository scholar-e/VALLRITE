import json
from pathlib import Path
import tempfile
import unittest

from CompetitiveWordDecoder.train_distilled_reranker import load_teacher_scores, prepare_records


class DistillationDataTest(unittest.TestCase):
    def test_prepare_records_joins_scores_and_reference(self) -> None:
        record = {
            "clip_id": "s1:x", "greedy_visual_phones": ["B"],
            "reference_words": ["bin"],
            "candidates": [
                {"text": "bin", "words": ["bin"], "log_score": -1.0},
                {"text": "tin", "words": ["tin"], "log_score": -2.0},
                {"text": "far", "words": ["far"], "log_score": -20.0},
            ],
        }
        scores = [
            {"clip_id": "s1:x", "candidate_index": 0, "score": -0.2},
            {"clip_id": "s1:x", "candidate_index": 1, "score": -0.8},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "examples.jsonl"
            teacher = root / "scores.jsonl"
            examples.write_text(json.dumps(record) + "\n")
            teacher.write_text("".join(json.dumps(row) + "\n" for row in scores))
            prepared = prepare_records(examples, teacher, 5.0)
        self.assertEqual(load_teacher_scores.__name__, "load_teacher_scores")
        self.assertEqual(prepared[0]["candidate_texts"], ["bin", "tin"])
        self.assertEqual(prepared[0]["reference_index"], 0)


if __name__ == "__main__":
    unittest.main()

