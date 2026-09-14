import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from AudioPhonemeLabeler.alignment import MontrealForcedAligner
from AudioPhonemeLabeler.core import (
    TeacherTranscript, Word, add_acoustic_evidence, build_alignment,
    phone_intervals_to_alignment, select_consensus, use_forced_alignment,
)
from AudioPhonemeLabeler.__main__ import cmudict_pronunciation
from AudioPhonemeLabeler.phones import ctc_viterbi_align, map_ipa_tokens
from AudioPhonemeLabeler.tongue_review import review_records
from VisualPhoneme.data import phoneme_target


class AudioPhonemeLabelerTests(unittest.TestCase):
    def test_consensus_chooses_medoid(self):
        def transcript(name, words):
            return TeacherTranscript(name, "en", 0.99, tuple(
                Word(word, index * 0.2, (index + 1) * 0.2, 0.9)
                for index, word in enumerate(words)
            ))
        first = transcript("first", ["set", "blue", "now"])
        second = transcript("second", ["set", "blue", "now"])
        outlier = transcript("outlier", ["get", "green", "soon"])
        selected, agreement = select_consensus([first, second, outlier])
        self.assertEqual(selected.teacher, "first")
        self.assertAlmostEqual(agreement, 0.5)

    def test_alignment_matches_visual_loader_contract(self):
        transcript = TeacherTranscript("teacher", "en", 0.98, (
            Word(" blue ", 0.1, 0.5, 0.9), Word("unknown!", 0.5, 0.8, 0.8),
        ))
        lexicon = {"blue": ["B", "L", "UW1"]}
        document = build_alignment(transcript, 1.0, lambda word: lexicon.get(word, []),
                                   {"source": "clip.wav"}, min_quality=0.0,
                                   min_coverage=0.5)
        self.assertEqual(document["tiers"]["phones"]["entries"][-1][2], "UW1")
        self.assertEqual(document["quality"]["unknown_words"], ["unknown"])
        self.assertEqual(document["label_status"], "accepted")
        with tempfile.TemporaryDirectory() as directory:
            alignment = Path(directory) / "alignment.json"
            alignment.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(len(phoneme_target(str(alignment))), 3)

    def test_low_agreement_is_rejected(self):
        transcript = TeacherTranscript("teacher", "en", 1.0, (
            Word("test", 0.0, 0.4, 1.0),
        ))
        document = build_alignment(transcript, 0.4, lambda _: ["T", "EH1", "S", "T"],
                                   {}, min_quality=0.0, min_coverage=1.0,
                                   min_agreement=0.75)
        self.assertEqual(document["label_status"], "rejected")
        self.assertIn("low_teacher_agreement",
                      document["quality"]["rejection_reasons"])

    def test_real_cmudict_dependency(self):
        self.assertEqual(cmudict_pronunciation("blue"), ["B", "L", "UW1"])
        self.assertEqual(cmudict_pronunciation("gdp"),
                         ["G", "IY1", "D", "IY1", "P", "IY1"])

    def test_direct_ipa_mapping_and_acoustic_rejection(self):
        direct = map_ipa_tokens(["b", "l", "uː", "?unmapped?"])
        self.assertEqual(direct.arpabet, ("B", "L", "UW"))
        self.assertEqual(direct.mapping_coverage, 0.75)
        transcript = TeacherTranscript("teacher", "en", 1.0, (
            Word("blue", 0.0, 0.4, 1.0),
        ))
        document = build_alignment(transcript, 1.0, cmudict_pronunciation, {},
                                   min_quality=0.0)
        add_acoustic_evidence(document, ["S"], 1.0, ["s"], "direct",
                              min_agreement=0.5, min_quality=0.0)
        self.assertEqual(document["label_status"], "rejected")
        self.assertIn("low_acoustic_phone_agreement",
                      document["quality"]["rejection_reasons"])

    def test_forced_alignment_replaces_provisional_timing(self):
        transcript = TeacherTranscript("teacher", "en", 1.0, (
            Word("blue", 0.0, 0.4, 1.0),
        ))
        document = build_alignment(transcript, 1.0, cmudict_pronunciation, {},
                                   min_quality=0.0)
        forced = {"start": 0, "end": 0.5, "tiers": {
            "words": {"entries": [[0.1, 0.5, "blue"]]},
            "phones": {"entries": [[0.1, 0.2, "B"], [0.2, 0.3, "L"],
                                    [0.3, 0.5, "UW1"]]},
        }}
        use_forced_alignment(document, forced, "MFA test")
        self.assertEqual(document["tiers"]["phones"]["entries"][0][:2], [0.1, 0.2])
        self.assertEqual(document["provenance"]["phone_timing"], "MFA test")

    def test_ctc_viterbi_alignment_produces_acoustic_intervals(self):
        import torch

        logits = torch.full((7, 3), -8.0)
        for frame, token in enumerate((0, 1, 1, 0, 2, 2, 0)):
            logits[frame, token] = 0.0
        forced = ctc_viterbi_align(
            logits.log_softmax(dim=-1), [[1], [2]], ["P", "B"], 0, 0.7
        )
        self.assertEqual([entry[2] for entry in forced.entries], ["P", "B"])
        for actual, expected in zip(
                [value for entry in forced.entries for value in entry[:2]],
                [0.1, 0.3, 0.4, 0.6]):
            self.assertAlmostEqual(actual, expected)
        self.assertGreater(forced.path_confidence, 0.99)
        self.assertAlmostEqual(forced.frame_duration, 0.1)

    def test_ctc_alignment_reconstructs_word_boundaries(self):
        transcript = TeacherTranscript("teacher", "en", 1.0, (
            Word("blue", 0.0, 0.4, 1.0), Word("bat", 0.4, 0.8, 1.0),
        ))
        lexicon = {"blue": ["B", "L", "UW1"], "bat": ["B", "AE1", "T"]}
        document = build_alignment(transcript, 1.0,
                                   lambda word: lexicon[word], {}, min_quality=0.0)
        entries = [(0.08, 0.14, "B"), (0.14, 0.22, "L"),
                   (0.22, 0.38, "UW"), (0.43, 0.50, "B"),
                   (0.50, 0.66, "AE"), (0.66, 0.76, "T")]
        forced = phone_intervals_to_alignment(document, entries, 0.75)
        self.assertEqual(forced["tiers"]["words"]["entries"],
                         [[0.08, 0.38, "blue"], [0.43, 0.76, "bat"]])
        self.assertEqual(forced["confidence"], 0.75)

    def test_ctc_viterbi_separates_repeated_phones_with_blank(self):
        import torch

        logits = torch.full((5, 2), -8.0)
        for frame, token in enumerate((0, 1, 0, 1, 0)):
            logits[frame, token] = 0.0
        forced = ctc_viterbi_align(
            logits.log_softmax(dim=-1), [[1], [1]], ["T", "T"], 0, 0.5
        )
        self.assertAlmostEqual(forced.entries[0][1], 0.2)
        self.assertAlmostEqual(forced.entries[1][0], 0.3)

    def test_tongue_review_requires_explicit_human_label(self):
        document = {
            "label_status": "accepted",
            "provenance": {"source": "clip.mp4"},
            "tiers": {"phones": {"entries": [
                [0.0, 0.1, "B"], [0.1, 0.2, "DH"], [0.2, 0.3, "L"],
            ]}},
        }
        records = review_records(document, Path("alignment.json"))
        self.assertEqual([record["phone"] for record in records], ["DH", "L"])
        self.assertTrue(all(record["tongue_visibility"] is None for record in records))
        self.assertTrue(all(not record["admit_to_training"] for record in records))

    @patch("AudioPhonemeLabeler.alignment.shutil.which",
           return_value="/usr/bin/mfa")
    def test_mfa_adapter_uses_json_output(self, _which):
        expected = {"tiers": {"words": {"entries": [[0, 1, "blue"]]},
                              "phones": {"entries": [[0, 1, "B"]]}}}

        def fake_run(command, **_kwargs):
            self.assertEqual(command[1], "align_one")
            self.assertIn("--output_format", command)
            self.assertEqual(command[command.index("--output_format") + 1], "json")
            Path(command[6]).write_text(json.dumps(expected), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as directory, patch(
                "AudioPhonemeLabeler.alignment.subprocess.run",
                side_effect=fake_run):
            work_dir = Path(directory)
            wav = work_dir / "audio.wav"
            wav.touch()
            actual = MontrealForcedAligner().align(wav, ["blue"], work_dir)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
