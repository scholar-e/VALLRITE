import unittest

from CompetitiveWordDecoder.noisy_teacher import nbest_prompt


class NoisyTeacherTest(unittest.TestCase):
    def test_prompt_contains_all_hypotheses_and_relative_scores(self) -> None:
        record = {"phone_hypotheses": [
            {"rank": 1, "phones": ["B", "IH", "N"], "ctc_log_score": -10.0},
            {"rank": 2, "phones": ["P", "IH", "N"], "ctc_log_score": -10.25},
        ]}
        prompt = nbest_prompt(record)
        self.assertIn("1. B IH N [relative_score=0.000]", prompt)
        self.assertIn("2. P IH N [relative_score=-0.250]", prompt)
        self.assertNotIn("reference", prompt.lower())


if __name__ == "__main__":
    unittest.main()
