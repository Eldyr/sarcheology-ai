"""Read-only overlap and spatial-dependence audit of Phase 3.8 optical exports.

Inputs: one patch_embeddings.csv per split (2024) with patch_id,row,col,split.
No model fitting, scoring, or modification of held-out test outputs.

Usage:
 uv run python scripts/phase3_optical/audit_optical_spatial_splits.py \
   --train outputs/exports/optical_phase38/train_2024/patch_embeddings.csv \
   --val outputs/exports/optical_phase38/val_2024/patch_embeddings.csv \
   --test outputs/exports/optical_phase38/test_2024/patch_embeddings.csv
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

PATCH_SIZE = 128
SPLITS = ('train', 'val', 'test')


def load_locations(paths: dict[str, Path], patch_size: int = PATCH_SIZE):
    problems = []
    records = []
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {split} CSV: {path}')
        df = pd.read_csv(path)
        needed = {'patch_id', 'split', 'year', 'quarter', 'row', 'col'}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        if df.empty:
            raise ValueError(f'{path}: empty CSV')
        if not (df['split'].astype(str) == split).all():
            problems.append({'type': 'incorrect_split', 'split': split})
        if not (df['year'].astype(int) == 2024).all():
            problems.append({'type': 'unexpected_year', 'split': split})
        if df.duplicated(['patch_id', 'year', 'quarter']).any():
            problems.append({'type': 'duplicate_patch_quarter', 'split': split})
        grouped = df.groupby('patch_id', sort=True)
        for pid, group in grouped:
            coords = group[['row', 'col']].drop_duplicates()
            if len(coords) != 1:
                problems.append({'type': 'inconsistent_patch_coordinates', 'split': split, 'patch_id': str(pid)})
                continue
            row, col = map(int, coords.iloc[0])
            if row < 0 or col < 0:
                problems.append({'type': 'negative_patch_coordinates', 'split': split, 'patch_id': str(pid)})
            quarters = sorted(set(map(int, group['quarter'])))
            if quarters != [1, 2, 3, 4] or len(group) != 4:
                problems.append({'type': 'incomplete_quarters', 'split': split, 'patch_id': str(pid), 'quarters': quarters, 'observations': int(len(group))})
            records.append({'split': split, 'patch_id': str(pid), 'row': row, 'col': col,
                            'bottom_row': row+patch_size, 'right_col': col+patch_size,
                            'observations': int(len(group))})
    return records, problems


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def join(self, a, b):
        x, y = self.root(a), self.root(b)
        if x == y:
            return
        if self.rank[x] < self.rank[y]:
            x, y = y, x
        self.parent[y] = x
        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1


def audit(records, patch_size=PATCH_SIZE, separation_pixels=0):
    if separation_pixels < 0:
        raise ValueError('separation_pixels must be >= 0')
    n = len(records)
    uf = UnionFind(n)
    pairs = []
    overlap_counts = Counter()
    separation_counts = Counter()
    cross_issues = []
    for i in range(n):
        a = records[i]
        for j in range(i+1, n):
            b = records[j]
            dr = max(0, abs(a['row'] - b['row']) - patch_size)
            dc = max(0, abs(a['col'] - b['col']) - patch_size)
            # axis-aligned edge-to-edge separation in pixel units
            distance = (dr * dr + dc * dc) ** .5
            h = max(0, patch_size - abs(a['row'] - b['row']))
            w = max(0, patch_size - abs(a['col'] - b['col']))
            intersection = h * w
            name = '/'.join(sorted((a['split'], b['split'])))
            if intersection > 0:
                uf.join(i, j)
                overlap_counts[name] += 1
                pair = {'split_a': a['split'], 'patch_a': a['patch_id'], 'split_b': b['split'],
                        'patch_b': b['patch_id'], 'overlap_pixels': intersection,
                        'overlap_fraction_of_patch': round(intersection/(patch_size*patch_size), 6),
                        'row_delta': abs(a['row']-b['row']), 'col_delta': abs(a['col']-b['col'])}
                pairs.append(pair)
                if a['split'] != b['split']:
                    cross_issues.append(pair)
            if a['split'] != b['split'] and distance < separation_pixels and intersection == 0:
                separation_counts[name] += 1
    components = defaultdict(list)
    for i, item in enumerate(records):
        components[uf.root(i)].append(item)
    clusters = []
    for members in components.values():
        splits = sorted(set(x['split'] for x in members))
        clusters.append({'cluster_id': None, 'n_patches': len(members), 'splits': splits,
                         'patches': sorted([{'split': x['split'], 'patch_id': x['patch_id']} for x in members],
                                           key=lambda x:(x['split'], x['patch_id']))})
    clusters.sort(key=lambda c:(-c['n_patches'], c['patches'][0]['split'], c['patches'][0]['patch_id']))
    assignments = []
    for k, cluster in enumerate(clusters, 1):
        cluster['cluster_id'] = f'C{k:04d}'
        for patch in cluster['patches']:
            assignments.append({'cluster_id': cluster['cluster_id'], **patch})
    cross_cluster = [c for c in clusters if len(c['splits']) > 1]
    summary = {
        'patch_size_pixels': patch_size, 'buffer_rule': 'minimum edge-to-edge distance between patch bounding boxes',
        'buffer_pixels': separation_pixels,
        'unique_patch_counts': dict(Counter(x['split'] for x in records)),
        'overlap_pair_counts': dict(overlap_counts),
        'cross_split_overlap_pairs': len(cross_issues),
        'cross_split_nearby_pairs_below_buffer_nonoverlapping': dict(separation_counts),
        'overlap_component_count': len(clusters),
        'components_spanning_multiple_splits': len(cross_cluster),
        'largest_component_patches': max((x['n_patches'] for x in clusters), default=0),
        'component_counts_by_split_membership': dict(Counter('+'.join(c['splits']) for c in clusters)),
        'interpretation': ('A zero cross-split overlap count excludes literal shared pixels between patches, '
                           'not spatial autocorrelation, neighboring geophysical features or shared preprocessing. '
                           'Overlapping-patch clusters are spatial dependence units, not independent locations.'),
        'test_use_policy': 'Do not select/retrain models based on this already-audited held-out test set.'
    }
    return summary, pairs, clusters, assignments


def write_csv(path, records, columns):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument('--'+split, required=True, type=Path, help=f'{split} patch_embeddings.csv')
    parser.add_argument('--output', type=Path, default=Path('outputs/quality/optical_phase38/spatial_audit'))
    parser.add_argument('--buffer-pixels', type=int, default=128, help='Additional proximity threshold in source pixels (default 128 = 1.28 km for 10m pixels)')
    parser.add_argument('--patch-size', type=int, default=128)
    args = parser.parse_args(argv)
    if args.patch_size < 1 or args.buffer_pixels < 0:
        parser.error('Patch size must be positive and buffer must not be negative')
    data, problems = load_locations({x: getattr(args,x) for x in SPLITS}, args.patch_size)
    # Strong guard against location IDs shared across splits.
    id_splits = defaultdict(set)
    for x in data:
        id_splits[x['patch_id']].add(x['split'])
    duplicates = [{'patch_id': k, 'splits': sorted(v)} for k,v in id_splits.items() if len(v)>1]
    if duplicates:
        problems.append({'type':'patch_ids_across_splits', 'items':duplicates})
    summary, pairs, clusters, assignments = audit(data, args.patch_size, args.buffer_pixels)
    summary['input_issues'] = problems
    summary['status'] = 'FAIL' if problems or summary['cross_split_overlap_pairs'] else ('PASS_WITH_WARNINGS' if summary['components_spanning_multiple_splits'] or any(summary['cross_split_nearby_pairs_below_buffer_nonoverlapping'].values()) else 'PASS')
    summary['meaning_of_pass'] = 'No cross-split overlapping pixels or buffer violations detected at requested threshold; within-split overlap is reported, not failed.'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'spatial_audit_report.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    write_csv(args.output/'overlapping_patch_pairs.csv', pairs, ['split_a','patch_a','split_b','patch_b','overlap_pixels','overlap_fraction_of_patch','row_delta','col_delta'])
    write_csv(args.output/'spatial_cluster_assignments.csv', assignments, ['cluster_id','split','patch_id'])
    (args.output/'overlap_components.json').write_text(json.dumps(clusters,indent=2), encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return 2 if summary['status']=='FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
