import unittest

from CompetitiveWordDecoder.evaluate_vallr_qwen import edit_distance, vallr_prompt, words


class EvaluatorTest(unittest.TestCase):
    def test_vallr_prompt_matches_training_tags(self) -> None:
        self.assertEqual(vallr_prompt(["DH", "AH", "K", "AE", "T"]), (
            "<S2S>\n<PHONEMES>\nDH AH K AE T\n</PHONEMES>\n<TEXT>\n"
        ))

    def test_word_normalization_and_distance(self) -> None:
        self.assertEqual(words("Hello, WORLD! It's me."), ["hello", "world", "it's", "me"])
        self.assertEqual(edit_distance(["the", "cat"], ["a", "cat"]), 1)


if __name__ == "__main__":
    unittest.main()
