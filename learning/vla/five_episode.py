"""Audited five-parent extension of the successful one-parent feedback recipe."""
import json
import math
from pathlib import Path

import numpy as np

from learning.vla.gripper_feedback import open_hold_anchors
from learning.vla.smolvla_data import CONTRACT, LEROBOT_VERSION, sha256
from learning.vla.window_training import CommandWindows


def select_parents(receipt):
    train = receipt['datasets']['train']['episodes']
    heldout = receipt['datasets']['validation']['episodes']
    names = [r['episode'] for r in train]
    if (receipt['contract'] != CONTRACT or receipt['lerobot_version'] != LEROBOT_VERSION
            or len(set(names)) != len(names) or set(names) & {r['episode'] for r in heldout}
            or len(train) < 5 or any(r['split'] != 'train' for r in train)):
        raise ValueError('Reviewed, disjoint training parents required')
    control = next(r for r in train if r['episode'] == 'train_20002')
    others = sorted((r for r in train if r != control), key=lambda r:r['episode'])
    return [control]+[others[round(i*(len(others)-1)/3)] for i in range(4)]


def build_selection(collection, export):
    collection, export = Path(collection).resolve(), Path(export).resolve()
    receipt = json.loads((export / 'openarm_export.json').read_text())
    windows = CommandWindows(receipt['datasets']['train']['episodes'])
    records, fingerprint = [], None
    for row in select_parents(receipt):
        folder = (collection / row['episode']).resolve()
        if folder.parent != collection:
            raise ValueError('Unsafe source parent')
        meta = json.loads((folder / 'metadata.json').read_text())
        if (sha256(folder / 'episode.npz') != row['data_sha256'] or meta['data_sha256'] != row['data_sha256']
                or meta['seed'] != row['seed'] or meta['outcome'] != 'success'):
            raise ValueError('Source checksum/seed/outcome differs')
        identity = {k:meta[k] for k in ('scene_sha256', 'source_hashes', 'robot_config', 'instruction')}
        if fingerprint is not None and identity != fingerprint:
            raise ValueError('Mixed source/robot scenes are not permitted')
        fingerprint = identity
        with np.load(folder / 'episode.npz', allow_pickle=False) as raw:
            states, actions = raw['observation_state'], raw['action_applied']
            eligible = open_hold_anchors(states, actions)
            if len(actions) != row['steps'] or not eligible or len(actions) < 1450:
                raise ValueError('Source length or stable open-hold coverage insufficient')
        position = receipt['datasets']['train']['episodes'].index(row)
        start = windows.ends[position-1] if position else 0
        records.append({**row, 'start':start, 'end':start+row['steps'], 'open_hold_local':eligible,
                        'metadata_sha256':sha256(folder / 'metadata.json')})
    return {'schema':'five-parent-openhold-v1', 'parents':records,
        'collection':str(collection), 'export_sha256':sha256(export / 'openarm_export.json'),
        'scene_identity':fingerprint, 'actor_contract_changed':False,
        'selection':'Known training control plus four evenly spaced remaining lexicographically sorted training parents; no validation/outcome tuning'}


def verify_selection(selection, export):
    if selection != build_selection(selection['collection'], export):
        raise ValueError('Five-parent selection differs from immutable source/export')


def audit_updates(rows, parents, expected_updates):
    if len(rows) != expected_updates:
        raise ValueError('Incomplete update budget')
    counts, perturbed = {p['episode']:0 for p in parents}, 0
    for step, row in enumerate(rows, 1):
        parent = parents[(step-1) % len(parents)]
        indices, offsets = row['indices'], row['gripper_sensor_offsets_m']
        eligible = {parent['start']+i for i in parent['open_hold_local']}
        if (row['step'] != step or row['episode'] != parent['episode'] or len(indices) != 8
                or len(offsets) != 4 or not all(i in eligible for i in indices[:4])
                or not all(parent['start'] <= i < parent['end'] for i in indices)
                or not all(math.isfinite(row[k]) for k in ('loss','grad_norm','lr'))
                or not all(math.isfinite(x) and abs(x) <= .002001 for x in offsets)):
            raise ValueError('Parent/quota/augmentation/finite-update violation')
        counts[parent['episode']] += 1
        perturbed += sum(x != 0 for x in offsets)
    return {'updates':len(rows), 'updates_per_parent':counts, 'sampled_anchors':len(rows)*8,
            'guaranteed_open_hold_anchors':len(rows)*4, 'nonzero_perturbed_anchors':perturbed,
            'actor_contract_or_raw_data_changed':False}
