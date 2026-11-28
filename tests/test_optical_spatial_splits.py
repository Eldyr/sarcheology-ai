"""Run with: uv run python -m pytest tests/test_optical_spatial_splits.py -q"""
import importlib.util
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / 'scripts/phase3_optical/audit_optical_spatial_splits.py'
spec = importlib.util.spec_from_file_location('spatial_audit', TARGET)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def entry(split, name, row, col):
    return {'split': split, 'patch_id': name, 'row': row, 'col': col}


def test_detects_literal_cross_split_overlap():
    summary, pairs, clusters, assignments = audit_module.audit([entry('train','a',0,0), entry('test','b',0,64)])
    assert summary['cross_split_overlap_pairs'] == 1
    assert pairs[0]['overlap_pixels'] == 128*64
    assert summary['components_spanning_multiple_splits'] == 1
    assert assignments[0]['cluster_id'] == assignments[1]['cluster_id']


def test_touching_edges_do_not_overlap():
    summary, pairs, _, _ = audit_module.audit([entry('train','a',0,0), entry('test','b',0,128)])
    assert summary['cross_split_overlap_pairs'] == 0
    assert pairs == []


def test_within_test_overlap_forms_one_dependence_cluster():
    summary, _, clusters, _ = audit_module.audit([entry('test','a',0,0), entry('test','b',0,64), entry('train','c',500,500)])
    assert summary['cross_split_overlap_pairs'] == 0
    assert summary['overlap_pair_counts']['test/test'] == 1
    assert sorted(x['n_patches'] for x in clusters) == [1,2]


def test_buffer_separate_from_overlap():
    summary, _, _, _ = audit_module.audit([entry('train','a',0,0), entry('val','b',0,200)], separation_pixels=100)
    assert summary['cross_split_overlap_pairs'] == 0
    assert summary['cross_split_nearby_pairs_below_buffer_nonoverlapping']['train/val'] == 1


def test_zero_buffer_no_adjacency_warning():
    summary, _, _, _ = audit_module.audit([entry('train','a',0,0), entry('val','b',0,200)], separation_pixels=0)
    assert not summary['cross_split_nearby_pairs_below_buffer_nonoverlapping']
