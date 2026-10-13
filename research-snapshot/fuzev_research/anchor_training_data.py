"""Owned training-role slices, cached once for repeated CUDA head epochs."""

from collections import defaultdict
import copy

import torch
from safetensors import safe_open

from .features import file_sha256
from .preference_head_retrain import check_rows
from .train import membership_identity


FIELDS = ('query', 'candidates', 'mask', 'targets', 'baseline')


class OwnedFeatureCache:
    def __init__(self, features, rows, role, primitive, device='cuda:0'):
        check_rows(rows, role, primitive)
        if not features.read_only or primitive not in ('choice', 'noul'):
            raise ValueError('Require read-only training-role Choice/Noul features')
        self.rows = copy.deepcopy(rows)
        rows = self.rows
        self.role = role
        self.primitive = primitive
        self.positions = {row['id']: index for index, row in enumerate(rows)}
        self.membership = membership_identity(rows)
        self.identity_sha256 = file_sha256(features.directory / 'identity.json')
        version = features.db.execute('PRAGMA data_version').fetchone()[0]
        groups = defaultdict(list)
        widths = []
        for index, row in enumerate(rows):
            owned = features.db.execute(
                "SELECT role,primitive,source,group_,shard,row_,"
                "json_extract(metadata,'$.has_baseline'),json_array_length(metadata,'$.candidates') "
                'FROM records WHERE id=?', (row['id'],)).fetchone()
            expected = tuple(row[key] for key in ('role', 'primitive', 'source', 'group_', 'shard', 'row_'))
            if owned is None or tuple(owned[:6]) != expected or owned[6] != 1 or not 2 <= owned[7] <= 16:
                raise ValueError('Missing native control or changed owned feature row')
            groups[row['shard']].append((index, row))
            widths.append(owned[7])
        width = max(widths)
        data = {'query': torch.empty(len(rows), 2560),
                'candidates': torch.zeros(len(rows), width, 2560),
                'mask': torch.zeros(len(rows), width, dtype=torch.bool),
                'targets': torch.zeros(len(rows), width),
                'baseline': torch.zeros(len(rows), width)}
        files = {}
        # Opening once per shard avoids repeated metadata/I/O; only owned row slices are read.
        for shard, selected in sorted(groups.items()):
            path = features.directory / f'shard-{shard:06d}.safetensors'
            before = file_sha256(path)
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                slices = {key: handle.get_slice(key) for key in FIELDS}
                for index, row in selected:
                    values = {key: tensor[row['row_']:row['row_'] + 1][0] for key, tensor in slices.items()}
                    count = widths[index]
                    mask = values['mask']
                    if (values['query'].shape != (2560,) or mask.ndim != 1
                            or values['candidates'].shape != (mask.numel(), 2560)
                            or values['targets'].shape != mask.shape or values['baseline'].shape != mask.shape
                            or mask.dtype != torch.bool or mask.numel() < count
                            or not mask[:count].all() or mask[count:].any()):
                        raise ValueError('Invalid owned feature/candidate mask dimensions')
                    if any(not torch.isfinite(value).all() for value in values.values()):
                        raise ValueError('Nonfinite owned training feature')
                    target = values['targets']
                    if (target.min() < 0 or target[count:].any()
                            or not torch.isclose(target.sum(), torch.tensor(1.), rtol=1e-7, atol=1e-7)):
                        raise ValueError('Invalid owned target distribution')
                    data['query'][index] = values['query']
                    for key in FIELDS[1:]:
                        data[key][index, :count] = values[key][:count]
            if before != file_sha256(path):
                raise ValueError('Owned feature shard changed while preloading')
            files[path.name] = before
        if (version != features.db.execute('PRAGMA data_version').fetchone()[0]
                or self.identity_sha256 != file_sha256(features.directory / 'identity.json')):
            raise ValueError('Corpus changed while preloading owned features')
        self.files = files
        self.data = {key: value.to(device) for key, value in data.items()}
        self.bytes = sum(value.numel() * value.element_size() for value in self.data.values())

    def batch(self, rows):
        check_rows(rows, self.role, self.primitive)
        indices = []
        for row in rows:
            index = self.positions.get(row['id'])
            if index is None or row != self.rows[index]:
                raise ValueError('Batch does not belong to the pinned training-role cache')
            indices.append(index)
        index = torch.tensor(indices, dtype=torch.long, device=self.data['query'].device)
        return {key: value.index_select(0, index) for key, value in self.data.items()}

    def provenance(self):
        return {'role': self.role, 'primitive': self.primitive, 'membership': self.membership,
                'feature_identity_sha256': self.identity_sha256, 'files': self.files,
                'cache_bytes': self.bytes, 'tensor_access': 'owned row slices only',
                'device': str(self.data['query'].device)}
