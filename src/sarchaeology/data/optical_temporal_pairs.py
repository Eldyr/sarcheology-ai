"""Deterministic temporal pairing built over Phase 3.3.2 MultiYearOpticalDataset.

Each anchor pairs with a different observation of the SAME Phase 1 patch_id.
The validation split retains the original 20 geographic locations, 2024 only.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset


class TemporalPairs(Dataset):
    def __init__(self, optical_dataset, same_quarter_probability=0.75, seed=42):
        self.base = optical_dataset
        self.same_quarter_probability = float(same_quarter_probability)
        if not 0 <= self.same_quarter_probability <= 1:
            raise ValueError('same_quarter_probability must be in [0,1]')
        self.seed = int(seed)
        self.epoch = 0
        self.by_patch = defaultdict(list)
        for idx, sample in enumerate(self.base.samples):
            self.by_patch[str(sample['patch_id'])].append(idx)
        for patch_id, indices in self.by_patch.items():
            if len(indices) < 2:
                raise ValueError(f'{patch_id} has fewer than two temporal observations')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.base)

    def partner_index(self, index):
        a = self.base.samples[index]
        choices = [i for i in self.by_patch[str(a['patch_id'])] if i != index]
        same_q = [i for i in choices if self.base.samples[i]['quarter'] == a['quarter']
                  and self.base.samples[i]['year'] != a['year']]
        cross_q = [i for i in choices if self.base.samples[i]['quarter'] != a['quarter']]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        choose_same = rng.random() < self.same_quarter_probability
        pool = (same_q if choose_same else cross_q) or (cross_q if choose_same else same_q) or choices
        return int(rng.choice(pool))

    def __getitem__(self, index):
        i = self.partner_index(index)
        a, b = self.base[index], self.base[i]
        assert a['patch_id'] == b['patch_id']
        return {'image_a': a['image'], 'valid_a': a['valid_mask'],
                'image_b': b['image'], 'valid_b': b['valid_mask'],
                'patch_id': a['patch_id'], 'quarter_a': a['quarter'],
                'quarter_b': b['quarter'], 'year_a': a['year'], 'year_b': b['year']}
