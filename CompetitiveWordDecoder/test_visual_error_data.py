"""Tests for CompetitiveWordDecoder.visual_error_data.build_rows."""
import hashlib
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from CompetitiveWordDecoder.visual_error_data import build_rows


class TestBuildRows(unittest.TestCase):
    """Small test suite for build_rows with synthetic fixtures."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.test_dir = Path(self.temporary.name)

        # Generate training manifest with allowed clip_ids (0-2999)
        all_clip_ids = list(range(3000))
        train_manifest_lines = []
        for cid in all_clip_ids:
            record = {
                'clip_id': cid,
                'reference_words': f'word{chr(97 + cid // 676)}{chr(97 + cid // 26 % 26)}{chr(97 + cid % 26)}',
                'reference_phones': ['B', 'AA'],
                'phone_hypotheses': [{'rank': 1, 'phones': ['P', 'AA'], 'ctc_log_score': -10.0}]
            }
            train_manifest_lines.append(json.dumps(record))
        self.train_manifest_path = (self.test_dir / 'train_manifest.jsonl')
        self.train_manifest_path.write_text('\n'.join(train_manifest_lines))

        # Generate source and heldout data with proper list phones
        source_data = []
        for cid in range(1150):  # clips 0-1149 (enough to get >=1100 after filtering)
            record = {
                'clip_id': cid,
                'reference_words': f'word{chr(97 + cid // 676)}{chr(97 + cid // 26 % 26)}{chr(97 + cid % 26)}',
                'reference_phones': ['B', 'AA'],
                'phone_hypotheses': [{'rank': 1, 'phones': ['P', 'AA'], 'ctc_log_score': -10.0}]
            }
            source_data.append(record)

        heldout_data = []
        for cid in range(2000, 3000):  # clips 2000-2999 (disjoint from source)
            record = {
                'clip_id': cid,
                'reference_words': f'word{chr(97 + cid // 676)}{chr(97 + cid // 26 % 26)}{chr(97 + cid % 26)}',
                'reference_phones': ['B', 'AA'],
                'phone_hypotheses': [{'rank': 1, 'phones': ['P', 'AA'], 'ctc_log_score': -10.0}]
            }
            heldout_data.append(record)

        self.source_path = (self.test_dir / 'source.jsonl')
        self.heldout_path = (self.test_dir / 'heldout.jsonl')
        self.source_path.write_text('\n'.join(json.dumps(r) for r in source_data))
        self.heldout_path.write_text('\n'.join(json.dumps(r) for r in heldout_data))

    # ------------------------------------------------------------------
    # Test 1: Mixed mode preserves selected IDs and counts (2000 train, 200 val)
    # ------------------------------------------------------------------
    def test_mixed_mode_preserves_ids_and_counts(self):
        """Mixed mode returns exactly 2000 training + 200 validation clips."""
        result, metadata = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )

        train_ids = [r['clip_id'] for r in result['train']]
        val_ids = [r['clip_id'] for r in result['validation']]
        self.assertEqual(len(train_ids), 2000)
        self.assertEqual(len(val_ids), 200)
        self.assertFalse(set(train_ids) & set(val_ids))

    # ------------------------------------------------------------------
    # Test 2: Clean mode uses only reference_phones (clean prompts duplicated)
    # ------------------------------------------------------------------
    def test_clean_mode_uses_only_reference_phones(self):
        """Clean mode uses only reference_phones, not hypotheses."""
        result, metadata = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='clean'
        )

        for split in ['train', 'validation']:
            for row in result[split]:
                self.assertEqual(row['input_kind'], 'clean')

    # ------------------------------------------------------------------
    # Test 3: Determinism across runs
    # ------------------------------------------------------------------
    def test_determinism_across_runs(self):
        """Running build_rows twice with same inputs yields identical results."""
        result1, _ = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )
        result2, _ = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )

        for split in ['train', 'validation']:
            ids1 = {r['clip_id'] for r in result1[split]}
            ids2 = {r['clip_id'] for r in result2[split]}
            self.assertEqual(ids1, ids2)

    # ------------------------------------------------------------------
    # Test 4: Heldout SHA256 computation
    # ------------------------------------------------------------------
    def test_heldout_sha256_computation(self):
        """Metadata should contain correct SHA256 of heldout file."""
        expected_sha = hashlib.sha256(Path(self.heldout_path).read_bytes()).hexdigest()

        result, metadata = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )

        self.assertEqual(metadata['heldout_sha256'], expected_sha)

    # ------------------------------------------------------------------
    # Test 5: Heldout-ID overlap rejects
    # ------------------------------------------------------------------
    def test_heldout_id_overlap_rejects(self):
        """Records with clip_ids in heldout should be rejected."""
        overlapping_data = [
            {
                'clip_id': 2000,
                'reference_words': 'wordabc',
                'reference_phones': ['B'],
                'phone_hypotheses': [{'rank': 1, 'phones': ['P'], 'ctc_log_score': -10.0}]
            }
        ]
        overlapping_path = self.test_dir / 'overlapping.jsonl'
        overlapping_path.write_text(json.dumps(overlapping_data[0]))

        with self.assertRaisesRegex(ValueError, 'non-training or held-out'):
            build_rows(
                overlapping_path,
                self.heldout_path,
                str(self.train_manifest_path),
                mode='mixed'
            )

    # ------------------------------------------------------------------
    # Test 6: Invalid mode rejects
    # ------------------------------------------------------------------
    def test_invalid_mode_rejects(self):
        """Modes other than 'mixed' or 'clean' should be rejected."""
        with self.assertRaises(ValueError):
            build_rows(
                self.source_path,
                self.heldout_path,
                str(self.train_manifest_path),
                mode='invalid'
            )

    # ------------------------------------------------------------------
    # Test 7: Record format fields
    # ------------------------------------------------------------------
    def test_record_format_fields(self):
        """Each output row should have prompt, target, unknown_words, clip_id, input_kind."""
        result, _ = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )

        for split in ['train', 'validation']:
            for row in result[split]:
                self.assertIn('prompt', row)
                self.assertIn('target', row)
                self.assertIn('unknown_words', row)
                self.assertIn('clip_id', row)
                self.assertIn('input_kind', row)

    # ------------------------------------------------------------------
    # Test 8: Mixed mode alternates clean/visual prompts
    # ------------------------------------------------------------------
    def test_mixed_mode_alternating(self):
        """Mixed mode should alternate between clean and visual input kinds."""
        result, _ = build_rows(
            self.source_path,
            self.heldout_path,
            str(self.train_manifest_path),
            mode='mixed'
        )

        # Check that both kinds appear in each split
        for split in ['train', 'validation']:
            kinds_in_split = {row['input_kind'] for row in result[split]}
            self.assertIn('clean', kinds_in_split)
            self.assertIn('visual', kinds_in_split)


if __name__ == '__main__':
    unittest.main()
