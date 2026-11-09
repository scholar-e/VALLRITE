import unittest

from VisualPhoneme.prepare_seamless import check_words, clean_word, utterances


def word(text, start, end, score=0.9):
    return {"word": text, "start": start, "end": end, "score": score}


class PrepareSeamlessTest(unittest.TestCase):
    def test_utterances_split_at_pauses_and_length(self):
        segment = {"words": [word("a", 0.0, 0.2), word("b", 0.3, 0.5), word("c", 1.5, 1.7),
                             word("d", 1.8, 7.6)]}
        groups = utterances([segment], max_gap=0.5, max_seconds=6.0)
        self.assertEqual([[w["word"] for w in g] for g in groups], [["a", "b"], ["c"], ["d"]])

    def test_untimed_word_drops_partial_utterance(self):
        segment = {"words": [word("a", 0.0, 0.2), {"word": "x", "start": None, "end": None},
                             word("b", 0.3, 0.5)]}
        self.assertEqual([[w["word"] for w in g] for g in utterances([segment], 0.5, 6.0)], [["b"]])

    def test_clean_word(self):
        self.assertEqual(clean_word("Challenge,"), "challenge")
        self.assertEqual(clean_word("That’s"), "that's")
        self.assertIsNone(clean_word("10"))
        self.assertIsNone(clean_word("e-mail"))

    def test_check_words_rejections(self):
        good = [word("we", 0, 1), word("would", 1, 2), word("create", 2, 3)]
        transcript, phones = check_words(good, 3, 0.5)
        self.assertEqual(transcript, "we would create")
        self.assertEqual(phones[:2], ["W", "IY"])
        self.assertEqual(check_words(good[:2], 3, 0.5), "too_few_words")
        self.assertEqual(check_words(good[:2] + [word("create", 2, 3, 0.2)], 3, 0.5), "low_word_score")
        self.assertEqual(check_words(good[:2] + [word("2020", 2, 3)], 3, 0.5), "non_alphabetic")
        self.assertEqual(check_words(good[:2] + [word("zxqvbn", 2, 3)], 3, 0.5), "unknown_word")


if __name__ == "__main__":
    unittest.main()
