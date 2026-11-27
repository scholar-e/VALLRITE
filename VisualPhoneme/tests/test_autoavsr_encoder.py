import itertools
import math
import tempfile
import unittest
from pathlib import Path

import torch

from VisualPhoneme.autoavsr_encoder import (AutoAvsrCtc, RelPositionMultiHeadedAttention,
                                            autoavsr_video_transform, ctc_greedy_text,
                                            load_units)
from VisualPhoneme.autoavsr_decoding import _ctc_extend
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts, normalize_words


class AutoAvsrEncoderTest(unittest.TestCase):
    def test_rel_shift_selects_query_minus_key_offsets(self):
        steps = 4
        # Column j of the unshifted scores holds relative offset (T-1) - j.
        offsets = torch.arange(steps - 1, -steps, -1, dtype=torch.float32)
        scores = offsets.expand(1, 1, steps, -1).clone()
        shifted = RelPositionMultiHeadedAttention.rel_shift(scores)[0, 0]
        expected = (torch.arange(steps)[:, None] - torch.arange(steps)[None, :]).float()
        self.assertTrue(torch.equal(shifted, expected))

    def test_padding_does_not_change_valid_outputs(self):
        torch.manual_seed(0)
        model = AutoAvsrCtc(vocabulary=7)
        model.encoder = type(model.encoder)(width=32, heads=4, hidden=64, layers=2, kernel=5)
        model.ctc_lo = torch.nn.Linear(32, 7)
        model.eval()
        short, long = torch.randn(1, 6, 1, 88, 88), torch.randn(1, 9, 1, 88, 88)
        with torch.no_grad():
            alone = model(short, torch.tensor([6]))
            padded = torch.cat((short, torch.zeros(1, 3, 1, 88, 88)), dim=1)
            batched = model(torch.cat((padded, long)), torch.tensor([6, 9]))
        torch.testing.assert_close(batched[0, :6], alone[0], atol=1e-4, rtol=1e-4)

    def test_video_transform_crops_and_normalizes(self):
        frames = torch.full((3, 96, 96), 255, dtype=torch.uint8)
        video = autoavsr_video_transform(frames)
        self.assertEqual(tuple(video.shape), (3, 1, 88, 88))
        torch.testing.assert_close(video.max(), torch.tensor((1 - 0.421) / 0.165))

    def test_units_and_greedy_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "units.txt"
            path.write_text("<unk> 1\n▁HELLO 2\n▁WOR 3\nLD 4\n", encoding="utf-8")
            units = load_units(path)
        self.assertEqual(units[0], "<blank>")
        self.assertEqual(units[-1], "<eos>")
        # Repeats collapse; blank separates the doubled token; <eos> is dropped.
        path_ids = [0, 2, 2, 0, 3, 4, 4, 5, 0]
        log_probabilities = torch.nn.functional.one_hot(torch.tensor(path_ids), len(units)).float()
        self.assertEqual(ctc_greedy_text(log_probabilities, units), "HELLO WORLD")

    def test_ctc_prefix_scores_match_path_enumeration(self):
        torch.manual_seed(1)
        steps, vocabulary = 4, 3
        log_ctc = torch.randn(steps, vocabulary).log_softmax(-1).double()

        def collapse(path):
            output, previous = [], None
            for token in path:
                if token != previous and token != 0:
                    output.append(token)
                previous = token
            return output

        def brute(prefix, exact=False):
            total = -math.inf
            for path in itertools.product(range(vocabulary), repeat=steps):
                output = collapse(path)
                if output == prefix if exact else output[:len(prefix)] == prefix:
                    score = sum(log_ctc[t, token].item() for t, token in enumerate(path))
                    total = float(torch.logaddexp(torch.tensor(total), torch.tensor(score)))
            return total

        empty = torch.stack((torch.full((steps,), -math.inf, dtype=torch.double),
                             log_ctc[:, 0].cumsum(0)), dim=1)[None]
        candidates = torch.tensor([[1, 2]])
        states, prefix = _ctc_extend(log_ctc, empty, torch.tensor([-1]), candidates)
        for column, token in enumerate((1, 2)):
            self.assertAlmostEqual(prefix[0, column].item(), brute([token]), places=6)
        # Extend [1] by a repeat (needs a blank between) and by a new token.
        states, prefix = _ctc_extend(log_ctc, states[0, :1], torch.tensor([1]), candidates)
        self.assertAlmostEqual(prefix[0, 0].item(), brute([1, 1]), places=6)
        self.assertAlmostEqual(prefix[0, 1].item(), brute([1, 2]), places=6)
        final = torch.logaddexp(states[0, 1, -1, 0], states[0, 1, -1, 1]).item()
        self.assertAlmostEqual(final, brute([1, 2], exact=True), places=6)

    def test_word_errors(self):
        self.assertEqual(normalize_words("It's About, the DATA!"), ["it's", "about", "the", "data"])
        self.assertEqual(edit_counts(["a", "b", "c"], ["a", "x", "c", "d"]), (1, 0, 1))
        self.assertEqual(edit_counts(["a", "b"], []), (0, 2, 0))


if __name__ == "__main__":
    unittest.main()


class EncoderLayersTest(unittest.TestCase):
    def test_selected_layers_end_with_default_output(self):
        import torch
        from VisualPhoneme.autoavsr_encoder import AutoAvsrEncoder
        torch.manual_seed(0)
        encoder = AutoAvsrEncoder(width=32, heads=4, hidden=64, layers=3, kernel=3).eval()
        video, lengths = torch.randn(1, 4, 1, 88, 88), torch.tensor([4])
        with torch.no_grad():
            stacked = encoder(video, lengths, [2, 3])
            torch.testing.assert_close(stacked[:, :, -1], encoder(video, lengths))
        self.assertEqual(tuple(stacked.shape), (1, 4, 2, 32))


class CropCorrectionTest(unittest.TestCase):
    def test_identity_matches_standard_transform_and_up_moves_content_up(self):
        import torch
        from VisualPhoneme.autoavsr_encoder import autoavsr_video_transform
        from VisualPhoneme.test_crop_correction import correct
        frames = torch.randint(0, 255, (2, 96, 96), dtype=torch.uint8)
        torch.testing.assert_close(correct(frames, 1.0, 0), autoavsr_video_transform(frames))
        bar = torch.zeros(1, 96, 96, dtype=torch.uint8)
        bar[:, 70:74] = 255  # bright rows 70-73 -> 66-69 in the 88 crop
        row = lambda v: int(v[0, 0].mean(-1).argmax())
        self.assertEqual(row(correct(bar, 1.0, 0)), 66)
        self.assertEqual(row(correct(bar, 1.0, 4)), 62)
        self.assertLess(row(correct(bar, 0.8, 0)), 64)  # zooming out pulls content toward the centre
