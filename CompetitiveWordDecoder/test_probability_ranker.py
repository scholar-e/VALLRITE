import unittest
from CompetitiveWordDecoder.evaluate_probability_ranker import choose, summarize


def candidate(text, path, decoder, tokens=1):
    return dict(text=text,phone_path_log_score=path,decoder_log_probability=decoder,scored_tokens=tokens)


class RankerTests(unittest.TestCase):
    def test_visual_and_decoder_evidence(self):
        rows = [candidate('cat',0,-10),candidate('dog',-1,-1)]
        self.assertEqual(choose(rows)[0],1)
        rows[1]['phone_path_log_score'] = -100
        self.assertEqual(choose(rows)[0],0)

    def test_duplicate_sentence_mass_and_normalization(self):
        rows = [candidate('cat',0,-1),candidate('dog',0,-.5),candidate('CAT!',0,-1)]
        selected,mass = choose(rows)
        self.assertEqual(selected,0)
        self.assertEqual(len(mass),2)
        self.assertAlmostEqual(sum(x['relative_mass'] for x in mass),1)

    def test_fixed_length_normalized_diagnostic(self):
        rows = [candidate('cat',0,-3,10),candidate('dog',0,-2,1)]
        self.assertEqual(choose(rows)[0],1)
        self.assertEqual(choose(rows,True)[0],0)

    def test_reference_does_not_change_selection(self):
        choices = [[candidate('cat',0,-10),candidate('dog',-1,-1)]]
        first = summarize([dict(clip_id='a',reference_words=['cat'])],choices)
        second = summarize([dict(clip_id='a',reference_words=['dog'])],choices)
        self.assertEqual(first['details'][0]['selected_index'],second['details'][0]['selected_index'])
        self.assertEqual(first['metrics']['probability_ranked']['wer'],1)
        self.assertEqual(second['metrics']['probability_ranked']['wer'],0)

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            choose([candidate('cat',0,float('nan'))])


if __name__ == '__main__':
    unittest.main()
