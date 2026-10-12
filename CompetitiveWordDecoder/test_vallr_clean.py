"""Offline regression tests for the original-recipe clean decoder."""
import unittest
from unittest.mock import patch

from CompetitiveWordDecoder.train_vallr_clean import edit_distance, example, encode, encode_eos


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class CleanRecipeTests(unittest.TestCase):
    def test_word_edits(self):
        for reference, hypothesis, expected in [([], [], 0), ([], ["hello"], 1),
                (["hello"], [], 1), (["the", "cat"], ["the", "big", "cat"], 1),
                (["the", "cat"], ["a", "dog"], 2)]:
            self.assertEqual(edit_distance(reference, hypothesis), expected)

    @patch("pronouncing.phones_for_word")
    def test_stress_first_pronunciation_and_unknown(self, lookup):
        lookup.side_effect = [["K AE1 T", "K AE2 T"], []]
        row = example("  Cat Zzzunknown  ")
        self.assertEqual(row["prompt"],
                         "<S2S>\n<PHONEMES>\nK AE T UNK(zzzunknown)\n</PHONEMES>\n<TEXT>\n")
        self.assertEqual(row["target"], "Cat Zzzunknown")
        self.assertEqual(row["unknown_words"], 1)
        self.assertEqual(lookup.call_args_list[0].args, ("cat",))

    def test_padding_and_target_only_labels(self):
        row = encode({"prompt": "aa", "target": "bbb"}, FakeTokenizer())
        self.assertEqual(row["input_ids"], [97, 97, 98, 98, 98] + [0]*507)
        self.assertEqual(row["attention_mask"], [1]*5 + [0]*507)
        self.assertEqual(row["labels"], [-100]*2 + [98]*3 + [-100]*507)

    def test_original_long_prompt_truncation(self):
        row = encode({"prompt": "a"*700, "target": "b"*100}, FakeTokenizer())
        self.assertEqual(row["input_ids"], [97]*508 + [98]*4)
        self.assertEqual(row["labels"], [-100]*508 + [98]*4)
        self.assertEqual(row["attention_mask"], [1]*512)

    def test_target_uses_remaining_space(self):
        row = encode({"prompt": "a"*400, "target": "b"*300}, FakeTokenizer())
        self.assertEqual(row["labels"], [-100]*400 + [98]*112)

    def test_eos_is_supervised_even_when_it_equals_padding(self):
        row = encode_eos({"prompt": "aa", "target": "bbb"}, FakeTokenizer())
        self.assertEqual(row["input_ids"], [97]*2 + [98]*3 + [0]*507)
        self.assertEqual(row["labels"], [-100]*2 + [98]*3 + [0] + [-100]*506)
        self.assertEqual(row["attention_mask"], [1]*6 + [0]*506)

    def test_eos_exact_boundary(self):
        row = encode_eos({"prompt": "a"*400, "target": "b"*111}, FakeTokenizer())
        self.assertEqual(row["labels"], [-100]*400 + [98]*111 + [0])
        self.assertEqual(row["attention_mask"], [1]*512)

    def test_eos_rejects_oversized_complete_pairs(self):
        self.assertIsNone(encode_eos({"prompt": "a"*400, "target": "b"*112}, FakeTokenizer()))
        self.assertIsNone(encode_eos({"prompt": "a"*600, "target": "b"}, FakeTokenizer()))

    def test_eos_requires_stop_token(self):
        tokenizer = FakeTokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaises(ValueError):
            encode_eos({"prompt": "a", "target": "b"}, tokenizer)

    def test_custom_context_preserves_full_target_and_eos(self):
        example_row = {'prompt': 'a'*600, 'target': 'b'*167}
        self.assertIsNone(encode_eos(example_row, FakeTokenizer()))
        encoded = encode_eos(example_row, FakeTokenizer(), max_length=768)
        self.assertEqual(len(encoded['input_ids']), 768)
        self.assertEqual(encoded['labels'], [-100]*600 + [98]*167 + [0])
        self.assertIsNone(encode_eos(dict(example_row, target='b'*168), FakeTokenizer(), 768))


if __name__ == "__main__":
    unittest.main()
