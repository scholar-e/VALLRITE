import unittest

from CompetitiveWordDecoder.prepare_source_disjoint import select, video_id


class SourceSplitTests(unittest.TestCase):
    def test_disjoint_deterministic_and_exclusions(self):
        rows = [dict(clip_id=f'lrs3:trainval/video{i}/clip{j}', transcript=f'word {chr(97+i)}',
                     frames=80, phonemes=['AA'], pronunciation_coverage=1)
                for i in range(10) for j in range(2)]
        a, eligible = select(rows, {'video0'}, {'word b'}, (3, 2, 2))
        b, _ = select(reversed(rows), {'video0'}, {'word b'}, (3, 2, 2))
        self.assertEqual(a, b)
        self.assertEqual(eligible, 8)
        selected = [r for group in a.values() for r in group]
        self.assertEqual(len({video_id(r['clip_id']) for r in selected}), 7)
        self.assertFalse({'video0', 'video1'} & {video_id(r['clip_id']) for r in selected})

    def test_short_chunk_and_duplicate_text_rejected(self):
        rows = [dict(clip_id=f'lrs3:trainval/v{i}/clip', transcript='same text',
                     frames=80, phonemes=['AA'], pronunciation_coverage=1) for i in range(4)]
        rows[0]['frames'] = 20
        rows[1]['pretrain_chunk_index'] = 0
        with self.assertRaises(ValueError):
            select(rows, set(), set(), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
