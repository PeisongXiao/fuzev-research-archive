"""Pure, per-lane retained-reference DEV statistics; no file or model access.

panel() takes (before, after) controls.describe records and a mapping from
component_sha256 to its complete source signature (a sequence of source names).
Only student may differ within a pair. The caller owns provenance, complete
official-DEV membership and CUDA inference. All 15 cells remain in the output.
"""

from collections import Counter, defaultdict

import numpy as np


CLASS_TRACKS = tuple(sorted((
    'boolq/noul/text/boolean_truth',
    'helpsteer2/choice/text/main',
    'helpsteer2/choice/text/correctness_pairwise_choice',
    'helpsteer2/noul/text/pairwise_preference_yes_no',
    'helpsteer3/choice/text/main',
    'helpsteer3/noul/text/pairwise_preference_yes_no',
    'mm-rlhf/choice/image/best_response',
    'mm-rlhf/choice/image/pairwise_preference',
    'mm-rlhf/choice/image/faithfulness_pairwise_choice',
    'mm-rlhf/noul/image/pairwise_preference_yes_no',
)))
ORDINAL_TRACKS = tuple('helpsteer2/ordinal/text/' + name for name in
                       ('coherence', 'complexity', 'correctness', 'helpfulness', 'verbosity'))
TRACKS = CLASS_TRACKS + ORDINAL_TRACKS
DRAWS, SEED, MIN_SUPPORT = 1000, 17, 100


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _winners(values, count):
    _require(isinstance(values, (list, tuple)) and len(values) > 0
             and all(type(value) is int and 0 <= value < count for value in values)
             and len(set(values)) == len(values), 'Invalid winner indices')
    return list(values)


def panel(paired_records, signatures):
    """Return complete workload metrics and their joint confirmation status.

    Average eligible records within each source/group, groups within each
    component, then components equally. Resample whole components within source
    signatures with the same multiplicities across every cell and both heads.
    Missing cells, <100 eligible components or any undefined draw are inconclusive.
    This panel has no self-improvement, macro-NLL or promotion requirement.
    """
    grouped = defaultdict(list)
    totals, eligible = Counter(), Counter()
    seen, components, group_components = set(), set(), {}
    for before, after in paired_records:
        _require({k: v for k, v in before.items() if k != 'student'} ==
                 {k: v for k, v in after.items() if k != 'student'},
                 'Paired input/gold/teacher metadata differs')
        identifier, track = before['id_sha256'], before['track']
        component, group = before['component_sha256'], before['group_sha256']
        _require(all(isinstance(v, str) and v for v in (identifier, component, group))
                 and identifier not in seen, 'Missing identity or duplicate paired record')
        _require(track in TRACKS and before['primitive'] == track.split('/')[1],
                 'Undeclared workload or primitive')
        _require(group_components.setdefault(group, component) == component,
                 'An original source/group crosses components')
        seen.add(identifier)
        components.add(component)
        totals[track] += 1
        count = before['valid_options']
        _require(type(count) is int and 2 <= count <= 16, 'Invalid valid-option count')
        gold = _winners(before['gold_winners'], count)
        _require(type(before['gold_tie']) is bool and before['gold_tie'] == (len(gold) > 1),
                 'Gold tie flag differs from original winners')
        if track in ORDINAL_TRACKS:
            values = [record['student']['ordinal_mae'] for record in (before, after)]
            _require(all(isinstance(v, (int, float, np.integer, np.floating))
                         and not isinstance(v, (bool, np.bool_))
                         and np.isfinite(v) and 0 <= v <= 4 for v in values),
                     'Invalid expected-rating MAE')
        else:
            values = []
            for record in (before, after):
                student = record['student']
                winners = _winners(student['winners'], count)
                _require(type(student['score_tie']) is bool
                         and student['score_tie'] == (len(winners) > 1),
                         'Prediction tie flag differs from winners')
                values.append(float(len(winners) == 1 and winners == gold))
            if before['gold_tie']:
                continue
        grouped[component, track, group].append(values)
        eligible[track] += 1

    components = sorted(components)
    positions = {component: i for i, component in enumerate(components)}
    columns = {track: i for i, track in enumerate(TRACKS)}
    strata = defaultdict(list)
    for component in components:
        signature = signatures.get(component)
        _require(isinstance(signature, (list, tuple)) and len(signature) > 0
                 and all(isinstance(source, str) and source for source in signature)
                 and len(set(signature)) == len(signature), 'Missing or invalid source signature')
        strata[tuple(sorted(signature))].append(positions[component])

    sums = np.zeros((len(components), len(TRACKS), 2), dtype=np.float64)
    groups = np.zeros((len(components), len(TRACKS)), dtype=np.float64)
    for (component, track, group), values in sorted(grouped.items()):
        row, column = positions[component], columns[track]
        sums[row, column] += np.mean(values, axis=0)
        groups[row, column] += 1

    # Normalize here: leaving group counts as weights would overweight linked groups.
    means = np.divide(sums, groups[:, :, None], out=np.zeros_like(sums),
                      where=groups[:, :, None] > 0)
    present = (groups > 0).astype(np.float64)
    support = present.sum(axis=0)
    point = np.divide(means.sum(axis=0), support[:, None],
                      out=np.full((len(TRACKS), 2), np.nan), where=support[:, None] > 0)
    deltas = means[:, :, 1] - means[:, :, 0]
    delta = np.divide(deltas.sum(axis=0), support,
                      out=np.full(len(TRACKS), np.nan), where=support > 0)
    draws = np.full((DRAWS, len(TRACKS)), np.nan)
    rng = np.random.default_rng(SEED)
    pools = [np.asarray(strata[key], dtype=np.int64) for key in sorted(strata)]
    for start in range(0, DRAWS, 16):
        size = min(16, DRAWS - start)
        multiplicities = np.zeros((size, len(components)), dtype=np.float64)
        for pool in pools:
            selected = pool[rng.integers(0, len(pool), size=(size, len(pool)))]
            np.add.at(multiplicities, (np.arange(size)[:, None], selected), 1)
        numerator = multiplicities @ deltas
        denominator = multiplicities @ present
        draws[start:start + size] = np.divide(numerator, denominator,
            out=np.full_like(numerator, np.nan), where=denominator > 0)

    result = {}
    for track, column in columns.items():
        difference = draws[:, column]
        finite_draws = int(np.isfinite(difference).sum())
        bounds = np.quantile(difference, [.025, .975]).tolist() if finite_draws == DRAWS else None
        supported = support[column] >= MIN_SUPPORT and bounds is not None
        ordinal = track in ORDINAL_TRACKS
        passes = bool(supported and (bounds[1] <= .1 if ordinal else bounds[0] >= -.02))
        before, after = point[column]
        result[track] = {
            'metric': 'expected_rating_mae' if ordinal else 'unique_gold_success',
            'control': float(before) if np.isfinite(before) else None,
            'candidate': float(after) if np.isfinite(after) else None,
            'candidate_minus_control': float(delta[column]) if np.isfinite(delta[column]) else None,
            'paired_95': bounds,
            'records': totals[track], 'eligible_records': eligible[track],
            'eligible_groups': int(groups[:, column].sum()), 'eligible_components': int(support[column]),
            'bootstrap_draws': DRAWS, 'bootstrap_seed': SEED, 'valid_bootstrap_draws': finite_draws,
            'passes': passes, 'status': 'pass' if passes else 'fail' if supported else 'inconclusive',
        }
    return {'workloads': result, 'passed': all(cell['passes'] for cell in result.values()),
            'records': len(seen), 'required_workloads': len(TRACKS),
            'passed_workloads': sum(cell['passes'] for cell in result.values()),
            'weighting': 'equal component after records-within-group and groups-within-component means',
            'claim_scope': 'Retained-reference DEV noninferiority only; not main quality or qualification.'}
