import unittest

import torch

from CompetitiveWordDecoder.evaluate_vallr_qwen import vallr_prompt
from CompetitiveWordDecoder.posterior_decoder import CLASSES, PosteriorEncoder, prompt_parts


class PosteriorDecoderTest(unittest.TestCase):
    def test_encoder_lengths_and_zero_initial_output(self):
        encoder = PosteriorEncoder(hidden_size=16, width=8)
        for steps in (1, 7, 8, 150, 151):
            output = encoder(torch.randn(2, steps, CLASSES).log_softmax(-1))
            self.assertEqual(output.shape[1], int(encoder.output_lengths(torch.tensor([steps]))[0]))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_right_padding_only_changes_final_tokens(self):
        torch.manual_seed(0)
        encoder = PosteriorEncoder(hidden_size=4, width=8)
        torch.nn.init.normal_(encoder.out.weight)
        clip = torch.randn(1, 40, CLASSES).log_softmax(-1)
        padded = torch.cat((clip, torch.full((1, 24, CLASSES), -20.0)), dim=1)
        length = int(encoder.output_lengths(torch.tensor([40]))[0])
        # Only the last token sees padding through the convolution receptive field.
        self.assertTrue(torch.allclose(encoder(clip)[0, :length - 2], encoder(padded)[0, :length - 2],
                                       atol=1e-5))

    def test_prompt_parts_reassemble_vallr_prompt(self):
        row = {"greedy_phones": ["HH", "AY"]}
        head, tail = prompt_parts(row)
        self.assertEqual(head + tail, vallr_prompt(["HH", "AY"]))
        self.assertTrue(head.endswith("</PHONEMES>\n"))
        self.assertTrue(tail.startswith("<TEXT>"))



class PosteriorDecoderRescoringTest(unittest.TestCase):
    def test_weights_trade_decoder_score_against_ctc(self):
        from CompetitiveWordDecoder.posterior_decoder_rescoring import errors, score
        fluent = {"decoder": -2.0, "ctc": -30.0, "words": 6, "unknown": 0, "errors": 3}
        faithful = {"decoder": -6.0, "ctc": -10.0, "words": 4, "unknown": 0, "errors": 0}
        clips = [{"candidates": [fluent, faithful]}]
        self.assertEqual(errors(clips, (0.0, 0.0, 0.0)), 3)   # decoder alone prefers the fluent one
        self.assertEqual(errors(clips, (0.25, 0.0, 0.0)), 0)  # CTC penalizes unsupported words
        self.assertAlmostEqual(score(faithful, (0.25, 1.0, -5.0)), -6.0 - 2.5 + 4.0)


if __name__ == "__main__":
    unittest.main()
