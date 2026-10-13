"""Preload only owned FIT/DEV/CAL slices for one shared decision scorer."""

import copy
import json
from collections import defaultdict

import torch
from safetensors import safe_open

from .features import file_sha256
from .preference_head_retrain import check_rows
from .train import membership_identity


FIELDS = ('query', 'candidates', 'mask', 'targets', 'baseline', 'ordinal_values')


class SharedFeatureCache:
    def __init__(self, features, rows, role, device='cuda:0'):
        check_rows(rows, role)
        if not features.read_only:
            raise ValueError('Shared training requires a read-only corpus')
        self.rows = copy.deepcopy(rows)
        self.role = role
        self.positions = {row['id']: index for index, row in enumerate(self.rows)}
        self.membership = membership_identity(self.rows)
        self.identity_sha256 = file_sha256(features.directory / 'identity.json')
        version = features.db.execute('PRAGMA data_version').fetchone()[0]
        shards, metadata = defaultdict(list), []
        for index, row in enumerate(self.rows):
            owned = features.db.execute('SELECT * FROM records WHERE id=?', (row['id'],)).fetchone()
            keys = ('role', 'primitive', 'source', 'group_', 'shard', 'row_')
            if owned is None or any(owned[key] != row[key] for key in keys):
                raise ValueError('Training row ownership changed')
            meta = json.loads(owned['metadata'])
            count = len(meta['candidates'])
            if (meta.get('has_baseline') is not True or not 2 <= count <= 16
                    or any(meta.get(key) != row[key] for key in ('id', 'role', 'primitive', 'source'))
                    or meta.get('group') != row['group_']):
                raise ValueError('Missing original control or changed training metadata')
            shards[row['shard']].append((index, row))
            metadata.append(meta)
        width = max(len(meta['candidates']) for meta in metadata)
        data = {'query': torch.empty(len(rows), 2560),
                'candidates': torch.zeros(len(rows), width, 2560),
                'mask': torch.zeros(len(rows), width, dtype=torch.bool),
                **{key: torch.zeros(len(rows), width) for key in ('targets', 'baseline', 'ordinal_values')}}
        files = {}
        for shard, selected in sorted(shards.items()):
            path = features.directory / f'shard-{shard:06d}.safetensors'
            before = file_sha256(path)
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                slices = {key: handle.get_slice(key) for key in FIELDS}
                for index, row in selected:
                    values = {key: tensor[row['row_']:row['row_'] + 1][0] for key, tensor in slices.items()}
                    meta = metadata[index]
                    count = len(meta['candidates'])
                    mask, target = values['mask'], values['targets']
                    if (values['query'].shape != (2560,) or mask.ndim != 1
                            or values['candidates'].shape != (mask.numel(), 2560)
                            or any(values[key].shape != mask.shape for key in ('targets', 'baseline', 'ordinal_values'))
                            or mask.dtype != torch.bool or mask.numel() < count
                            or not mask[:count].all() or mask[count:].any()
                            or any(not torch.isfinite(value).all() for value in values.values())):
                        raise ValueError('Invalid owned training tensor geometry or values')
                    if (target.min() < 0 or target[count:].any()
                            or not torch.isclose(target.sum(), torch.tensor(1.), rtol=1e-7, atol=1e-7)
                            or not torch.equal(target[:count], torch.tensor(meta['target'], dtype=torch.float32))):
                        raise ValueError('Training target differs from its pinned metadata')
                    if row['primitive'] == 'ordinal':
                        expected = torch.tensor([c['ordinal_value'] for c in meta['candidates']], dtype=torch.float32)
                        if not torch.equal(values['ordinal_values'][:count], expected) or not (expected.diff() > 0).all():
                            raise ValueError('Ordinal values must match ordered candidate levels')
                    data['query'][index] = values['query']
                    for key in FIELDS[1:]:
                        data[key][index, :count] = values[key][:count]
            if before != file_sha256(path):
                raise ValueError('Training shard changed during preloading')
            files[path.name] = before
        if (version != features.db.execute('PRAGMA data_version').fetchone()[0]
                or self.identity_sha256 != file_sha256(features.directory / 'identity.json')):
            raise ValueError('Corpus changed during preloading')
        self.files = files
        self.data = {key: value.to(device) for key, value in data.items()}
        self.bytes = sum(value.numel() * value.element_size() for value in self.data.values())

    def batch(self, rows):
        check_rows(rows, self.role)
        indices = []
        for row in rows:
            index = self.positions.get(row['id'])
            if index is None or row != self.rows[index]:
                raise ValueError('Batch does not belong to the pinned role cache')
            indices.append(index)
        index = torch.tensor(indices, dtype=torch.long, device=self.data['query'].device)
        return {key: value.index_select(0, index) for key, value in self.data.items()}

    def provenance(self):
        return {'role': self.role, 'membership': self.membership,
                'feature_identity_sha256': self.identity_sha256, 'files': self.files,
                'cache_bytes': self.bytes, 'tensor_access': 'owned row slices only',
                'device': str(self.data['query'].device)}
