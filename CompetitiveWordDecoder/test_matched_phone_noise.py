import random
import unittest

from CompetitiveWordDecoder.matched_phone_noise import NoiseModel, align, estimate_profile
from VisualPhoneme.evaluate_autoavsr_ctc import edit_counts


class MatchedPhoneNoiseTest(unittest.TestCase):
    def test_align_records_substitution_deletion_and_insertions(self):
        aligned = align(["K", "AE", "T"], ["G", "AE", "S", "T", "Z"])
        self.assertEqual(aligned[0], ("<s>", "<s>", []))
        self.assertEqual([(ref, out) for ref, out, _ in aligned[1:]],
                         [("K", "G"), ("AE", "AE"), ("T", "T")])
        self.assertEqual(sum(len(extra) for _, _, extra in aligned), 2)
        deleted = align(["K", "AE", "T"], ["K", "T"])
        self.assertIn(("AE", None, []), deleted)

    def test_profile_rate_matches_edit_distance(self):
        records = [
            {"reference_phones": ["K", "AE", "T"], "phone_hypotheses": [{"phones": ["G", "AE"]}]},
            {"reference_phones": ["D", "AO", "G"], "phone_hypotheses": [{"phones": ["D", "AO", "G", "Z"]}]},
        ]
        profile = estimate_profile(records)
        expected = sum(sum(edit_counts(r["reference_phones"], r["phone_hypotheses"][0]["phones"]))
                       for r in records) / 6
        self.assertAlmostEqual(profile["phone_error_rate"], expected)
        self.assertEqual(profile["outcomes"]["K"], {"G": 1})
        self.assertEqual(profile["outcomes"]["T"], {"<del>": 1})

    def test_sampler_reproduces_measured_rate(self):
        rng = random.Random(0)
        phones = "AA AE AH B D K S T".split()
        records = []
        for _ in range(200):
            reference = rng.choices(phones, k=20)
            hypothesis = [rng.choice(phones) if rng.random() < 0.2 else p
                          for p in reference if rng.random() > 0.05]
            records.append({"reference_phones": reference,
                            "phone_hypotheses": [{"phones": hypothesis}]})
        profile = estimate_profile(records)
        model = NoiseModel(profile)
        edits = total = 0
        for record in records:
            noisy, _ = model.corrupt(record["reference_phones"], rng)
            edits += sum(edit_counts(record["reference_phones"], noisy))
            total += len(record["reference_phones"])
        self.assertAlmostEqual(edits / total, profile["phone_error_rate"], delta=0.04)


if __name__ == "__main__":
    unittest.main()
