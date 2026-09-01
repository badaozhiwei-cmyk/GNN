import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import aggregate_ablation_results as aggregate


def write_config(root, suffix, descriptor='Mphys', gated=False, seeds=None):
    mode_dir = os.path.join(root, f'HFC_loro_{descriptor}_{suffix}')
    os.makedirs(mode_dir)
    config = {
        'family': 'HFC',
        'mode': 'loro',
        'descriptor_mode': descriptor,
        'use_adaptive_gate': gated,
        'seeds': seeds or [42, 43, 44],
    }
    with open(os.path.join(mode_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle)
    return mode_dir


class AblationIntegrityTests(unittest.TestCase):
    def test_gated_and_ungated_are_discovered_separately(self):
        with tempfile.TemporaryDirectory() as root, patch.object(aggregate, 'BASE_DIR', root):
            plain = write_config(root, 'plainhash', gated=False)
            gated = write_config(root, 'gatehash', gated=True)

            self.assertEqual(aggregate.discover_mode_dir('Mphys')[0], plain)
            self.assertEqual(aggregate.discover_mode_dir('Mphys_gated')[0], gated)

    def test_ambiguous_config_requires_explicit_hash(self):
        with tempfile.TemporaryDirectory() as root, patch.object(aggregate, 'BASE_DIR', root):
            write_config(root, 'hashone', descriptor='M0')
            selected = write_config(root, 'hashtwo', descriptor='M0')

            with self.assertRaises(RuntimeError):
                aggregate.discover_mode_dir('M0')
            self.assertEqual(aggregate.discover_mode_dir('M0', 'hashtwo')[0], selected)

    def test_incomplete_fold_set_is_rejected(self):
        rows = []
        for seed in [42, 43, 44]:
            rows.append({
                'seed': seed,
                'refrigerant': 'R23',
                'true_x1': 0.1,
                'pred_x1': 0.2,
            })
        with self.assertRaises(RuntimeError):
            aggregate.validate_prediction_completeness(
                pd.DataFrame(rows), [42, 43, 44], 'synthetic-result'
            )


if __name__ == '__main__':
    unittest.main()
