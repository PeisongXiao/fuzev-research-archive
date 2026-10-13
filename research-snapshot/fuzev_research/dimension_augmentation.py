"""Pure phase-1 Choice adapters for existing human dimension ratings.

MM faithfulness uses the five-level visual-accuracy rubric (1..5); HS2 uses
human correctness ratings (0..4). Higher wins and equal ratings yield .5/.5,
not absolute truth or a probability obtained by rescaling the rating.
MM final_ranking is lower-is-better; NVIDIA signed preference is positive for
response 2. Neither supplies labels here. Candidate orientation and the one
pair retained per group are label-independent; group sampling is proportional
over first/second/tie labels. No files, models, datasets or pipelines are opened.
The caller must bind parents, raw-file pins and group ownership to the original
corpus before extraction; these pure checks are not a qualification receipt.
"""

from collections import defaultdict
import copy
import hashlib
from itertools import combinations
import json
import math


ADAPTER_VERSION = 'human-dimension-choice-v1'
REVISION = '2026-10-05-v4-dimension-choice'
DIMENSION_REVISION = REVISION
SEED = 20261005
ROLES = ('fit', 'development', 'calibration')
SOURCES = ('mm-rlhf', 'helpsteer2')
STRATA = ('first', 'second', 'tie')
PHASE1_BUDGETS = {
    'fit': {'mm-rlhf': 3072, 'helpsteer2': 1024},
    'development': {'mm-rlhf': 768, 'helpsteer2': 256},
    'calibration': {'mm-rlhf': 768, 'helpsteer2': 256},
}
MM_PREFIX = 'Which response best answers the question about the image?\n'
DIMENSIONS = {'mm-rlhf': ('faithfulness', 1, 5), 'helpsteer2': ('correctness', 0, 4)}
ADAPTATIONS = {'mm-rlhf': 'faithfulness_pairwise_choice', 'helpsteer2': 'correctness_pairwise_choice'}
OUTPUTS = {'text': 'dimension-choice-text.jsonl', 'image': 'dimension-choice-images.jsonl'}
MANIFESTS = {'text': 'dimension-choice-text-splits.json', 'image': 'dimension-choice-images-splits.json'}
REQUIRED_PROVENANCE_FIELDS = (
    'adapter_version', 'orientation_seed', 'candidate_permutation', 'dimension', 'label_semantics',
    'dimension_scores', 'parent_ids', 'parent_record_sha256s', 'parent_input_sha256', 'parent_records',
    'source_provenance', 'source_annotations', 'rating_record_ids',
)
OPTIONAL_PROVENANCE_FIELDS = ('parent_preparation', 'parent_feature_identity_sha256')
PROVENANCE_FIELDS = REQUIRED_PROVENANCE_FIELDS + OPTIONAL_PROVENANCE_FIELDS
DIMENSION_METADATA = PROVENANCE_FIELDS


class QuotaInfeasible(ValueError):
    def __init__(self, deficits):
        self.deficits = copy.deepcopy(deficits)
        super().__init__('Approved phase-1 quotas are infeasible; no reduction: ' + json.dumps(deficits, sort_keys=True))


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def _hash(*values):
    return hashlib.sha256(_bytes(values)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _seed(value):
    if type(value) is not int or value < 0:
        raise ValueError('Require an explicit nonnegative integer adapter seed')


def _frame(parent, source, group_roles, *, canonical=True):
    if not isinstance(parent, dict) or parent.get('source') != source or source not in SOURCES:
        raise ValueError('Only admitted MM-RLHF and HelpSteer2 parents are supported')
    if (parent.get('role') not in ROLES or parent.get('original_split') != 'train'
            or parent.get('evaluation_scope', 'training') != 'training'
            or parent.get('dataset_purpose', 'training') != 'training'):
        raise ValueError('TEST, official heldout and external parents are excluded')
    if any(not _text(parent.get(key)) for key in ('id', 'revision', 'group')):
        raise ValueError('Missing original parent identity/revision/group')
    if not isinstance(group_roles, dict) or group_roles.get(parent['group']) != parent['role']:
        raise ValueError('Parent does not match original group ownership')
    expected = 'image' if source == 'mm-rlhf' else 'text'
    if parent.get('modality') != expected:
        raise ValueError('Parent source/modality mismatch')
    context = parent.get('context')
    if (not isinstance(context, list) or any(not isinstance(message, dict)
            or message.get('role') not in {'user', 'assistant', 'system'}
            or not isinstance(message.get('content'), str) for message in context)):
        raise ValueError('Invalid original conversation')
    candidates, target = parent.get('candidates'), parent.get('target')
    if (not _text(parent.get('question')) or parent.get('primitive') not in {'choice', 'ordinal'}
            or not isinstance(candidates, list) or not 2 <= len(candidates) <= 16
            or any(not isinstance(candidate, dict) or not _text(candidate.get('id'))
                   or not _text(candidate.get('text')) for candidate in candidates)
            or len({candidate['id'] for candidate in candidates}) != len(candidates)
            or not isinstance(target, list) or len(target) != len(candidates)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in target)
            or not math.isclose(sum(target), 1., rel_tol=1e-9)):
        raise ValueError('Invalid original decision inputs/target')
    if canonical:
        from .data import stable_id

        # Legacy IDs use Python/dict rendering. Rebuild the original constructors'
        # field order so sorted JSON persistence cannot change their meaning.
        candidate_fields = ('id', 'text', 'ordinal_value') if parent['primitive'] == 'ordinal' else ('id', 'text')
        if (any(set(message) != {'role', 'content'} for message in context)
                or any(set(candidate) != set(candidate_fields) for candidate in candidates)):
            raise ValueError('Original input fields differ from the canonical source construction')
        original_context = [{'role': message['role'], 'content': message['content']} for message in context]
        original_candidates = [{key: candidate[key] for key in candidate_fields} for candidate in candidates]
        expected_id = stable_id(parent['source'], parent['group'], original_context, parent['question'],
                                json.dumps(original_candidates), parent['primitive'], target)
        if parent['id'] != expected_id:
            raise ValueError('Original parent ID does not bind its complete input and target')
    if source == 'mm-rlhf' and (parent.get('official_holdout_overlap') is not False
            or not _sha(parent.get('image_sha256')) or not _text(parent.get('image_path'))
            or not _text(parent.get('source_image_id'))):
        raise ValueError('Missing admitted image provenance or holdout exclusion')


def _provenance(value, parent):
    filename = 'data.jsonl' if parent['source'] == 'mm-rlhf' else 'train.jsonl.gz'
    if (not isinstance(value, dict) or set(value) != {'source', 'revision', 'filename', 'sha256'}
            or value['source'] != parent['source'] or value['revision'] != parent['revision']
            or value['filename'] != filename or not _sha(value['sha256'])):
        raise ValueError('Require matching pinned original training annotation file provenance')
    return copy.deepcopy(value)


def _rating(value, source):
    _, low, high = DIMENSIONS[source]
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Human rating is missing, nonfinite or outside the documented rubric')
    return value


def _target(scores):
    return [1., 0.] if scores[0] > scores[1] else [0., 1.] if scores[0] < scores[1] else [.5, .5]


def _ordinal_target(score):
    low, high = int(score), min(4, int(score) + 1)
    target = [0.] * 5
    target[low] += 1 - (score - low)
    target[high] += score - low
    return target


def _input_key(view):
    return _hash({key: view.get(key) for key in ('source', 'revision', 'group', 'modality',
                 'context', 'question', 'candidates', 'image_sha256', 'dimension')})


def _identity(view):
    return _hash(ADAPTER_VERSION, view['orientation_seed'], _input_key(view), view['role'],
                 view['parent_records'], view['source_provenance'], view['source_annotations'],
                 view['dimension_scores'], view['rating_record_ids'])


def _rating_id(provenance, annotation, candidate):
    return _hash('human-rating-record-v1', provenance, annotation, candidate)


def _make(parents, candidates, scores, annotations, provenance, question, context, seed):
    _seed(seed)
    source = parents[0]['source']
    result = copy.deepcopy(parents[0])
    for field in ('attribute', 'original_score', 'original_preference'):
        result.pop(field, None)
    dimension = DIMENSIONS[source][0]
    result.update(primitive='choice', question=question, context=copy.deepcopy(context),
                  candidates=copy.deepcopy(candidates), dimension=dimension,
                  adaptation=dimension + '_pairwise_choice', label_semantics='relative_human_' + dimension + '_rating',
                  adapter_version=ADAPTER_VERSION, orientation_seed=seed,
                  dataset_purpose='training', evaluation_scope='training', dataset_scope_revision=REVISION)
    # Parent IDs contain labels. Orient using only rendered inputs, never those IDs.
    permutation = [1, 0] if int(_hash('orientation', seed, _input_key(result)), 16) & 1 else [0, 1]
    result['candidates'] = [result['candidates'][index] for index in permutation]
    result['dimension_scores'] = [scores[index] for index in permutation]
    result['source_annotations'] = [copy.deepcopy(annotations[index]) for index in permutation]
    result['candidate_permutation'] = permutation
    result['target'] = _target(result['dimension_scores'])
    result['parent_records'] = copy.deepcopy(parents)
    result['parent_ids'] = [parent['id'] for parent in parents]
    result['parent_record_sha256s'] = [hashlib.sha256(_bytes(parent)).hexdigest() for parent in parents]
    result['source_provenance'] = provenance
    result['rating_record_ids'] = [_rating_id(provenance, annotation, candidate)
                                  for annotation, candidate in zip(result['source_annotations'], result['candidates'])]
    result['parent_input_sha256'] = _input_key(result)
    result['id'] = _identity(result)
    return result


def mm_faithfulness_views(parent, raw, provenance, *, group_roles, seed=SEED):
    """Enumerate original response pairs; selection later retains one per group."""
    parent = copy.deepcopy(parent)
    _frame(parent, 'mm-rlhf', group_roles)
    _seed(seed)
    raw, provenance = copy.deepcopy(raw), _provenance(provenance, parent)
    if (not isinstance(raw, dict) or parent.get('primitive') != 'choice'
            or parent.get('adaptation') != 'best_response' or parent['context'] != []
            or raw.get('id') != parent.get('annotation_id') or not _text(raw.get('question'))
            or raw.get('image') != parent['source_image_id']
            or parent.get('question') != MM_PREFIX + raw['question']):
        raise ValueError('MM raw annotation does not bind to the canonical admitted parent')
    responses, ratings = raw.get('models_output'), raw.get('faithfulness')
    if (not isinstance(responses, list) or not 2 <= len(responses) <= 16
            or any(not _text(response) for response in responses)
            or not isinstance(ratings, list) or len(ratings) != len(responses)
            or parent.get('candidates') != [{'id': str(i), 'text': text} for i, text in enumerate(responses)]):
        raise ValueError('MM response/faithfulness/candidate alignment is invalid')
    ratings = [_rating(value, 'mm-rlhf') for value in ratings]
    ranks = raw.get('final_ranking')
    if (not isinstance(ranks, list) or len(ranks) != len(responses)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 1 for value in ranks)):
        raise ValueError('Missing original MM rank provenance; ranks are not faithfulness labels')
    best = min(ranks)
    if parent['target'] != [float(rank == best) / ranks.count(best) for rank in ranks]:
        raise ValueError('MM parent target does not match original lower-is-better rankings')
    seen = {}
    for response, rating in zip(responses, ratings):
        if response in seen and seen[response] != rating:
            raise ValueError('Identical MM response text has conflicting human ratings')
        seen[response] = rating
    raw_sha = hashlib.sha256(_bytes(raw)).hexdigest()
    question = 'Which response has higher visual faithfulness to the image?\n' + raw['question']
    views = []
    for a, b in combinations(range(len(responses)), 2):
        if responses[a] == responses[b]:
            continue
        annotations = [{'field': 'faithfulness', 'annotation_id': raw['id'], 'response_index': index,
                        'value': ratings[index], 'record_sha256': raw_sha, 'parent_id': parent['id'],
                        'response_sha256': hashlib.sha256(responses[index].encode()).hexdigest()} for index in (a, b)]
        views.append(_make([parent], [parent['candidates'][index] for index in (a, b)],
                           [ratings[a], ratings[b]], annotations, provenance, question, [], seed))
    return views


def hs2_correctness_view(parents, raws, provenance, *, row_indices, group_roles, seed=SEED):
    """Bind two human correctness rows to their original ordinal parents."""
    from .data import conversation, prompt_group

    parents = copy.deepcopy(parents)
    if not isinstance(parents, (list, tuple)) or len(parents) != 2:
        raise ValueError('HS2 needs exactly two original correctness parents')
    for parent in parents:
        _frame(parent, 'helpsteer2', group_roles)
    _seed(seed)
    if (not isinstance(raws, (list, tuple)) or len(raws) != 2
            or not isinstance(row_indices, (list, tuple)) or len(row_indices) != 2
            or any(type(index) is not int or index < 0 for index in row_indices)
            or row_indices[0] == row_indices[1]):
        raise ValueError('HS2 needs two distinct original training row locators')
    raws = copy.deepcopy(raws)
    provenance = _provenance(provenance, parents[0])
    bound = []
    for parent, raw, index in zip(parents, raws, row_indices):
        if (not isinstance(raw, dict) or not _text(raw.get('prompt')) or not _text(raw.get('response'))
                or parent.get('primitive') != 'ordinal' or parent.get('attribute') != 'correctness'):
            raise ValueError('HS2 raw correctness row does not bind to an ordinal parent')
        score = _rating(raw.get('correctness'), 'helpsteer2')
        messages = conversation(raw['prompt'])
        if (parent['context'] != [*messages, {'role': 'assistant', 'content': raw['response']}]
                or parent['group'] != prompt_group(messages)
                or parent.get('question') != "Rate the response's correctness on the HelpSteer2 scale from 0 to 4."
                or parent.get('candidates') != [{'id': str(i), 'text': f'correctness: {i}', 'ordinal_value': i} for i in range(5)]
                or parent.get('target') != _ordinal_target(score)
                or parent.get('original_score') != score
                or any(parent[key] != parents[0][key] for key in ('group', 'role', 'revision'))):
            raise ValueError('HS2 prompt/response/rating or inherited ownership binding mismatch')
        candidate = {'id': _hash('response', parent['source'], parent['revision'], parent['group'], raw['response']),
                     'text': raw['response']}
        annotation = {'field': 'correctness', 'row_index': index, 'value': score,
                      'record_sha256': hashlib.sha256(_bytes(raw)).hexdigest(), 'parent_id': parent['id'],
                      'response_sha256': hashlib.sha256(raw['response'].encode()).hexdigest()}
        bound.append((candidate, score, annotation, parent))
    if bound[0][0]['text'] == bound[1][0]['text']:
        raise ValueError('HS2 needs two distinct original response texts; do not average conflicting annotations')
    bound.sort(key=lambda item: item[0]['id'])
    return _make([item[3] for item in bound], [item[0] for item in bound], [item[1] for item in bound],
                 [item[2] for item in bound], provenance,
                 'Which response has higher correctness on the HelpSteer2 rubric?',
                 bound[0][3]['context'][:-1], seed)


def _check_view(view, group_roles, seed):
    if not isinstance(view, dict) or view.get('source') not in SOURCES:
        raise ValueError('Unknown dimension view source')
    _frame(view, view['source'], group_roles, canonical=False)
    dimension = DIMENSIONS[view['source']][0]
    parents, candidates, annotations = view.get('parent_records'), view.get('candidates'), view.get('source_annotations')
    if (any(key not in view for key in REQUIRED_PROVENANCE_FIELDS)
            or view.get('adapter_version') != ADAPTER_VERSION or view.get('orientation_seed') != seed
            or type(view.get('orientation_seed')) is not int
            or view.get('dataset_scope_revision') != REVISION
            or view.get('primitive') != 'choice' or view.get('dimension') != dimension
            or view.get('adaptation') != dimension + '_pairwise_choice'
            or view.get('label_semantics') != 'relative_human_' + dimension + '_rating'
            or not isinstance(parents, list) or len(parents) != (1 if view['source'] == 'mm-rlhf' else 2)
            or not isinstance(candidates, list) or len(candidates) != 2
            or any(not isinstance(candidate, dict) or not _text(candidate.get('id')) or not _text(candidate.get('text')) for candidate in candidates)
            or candidates[0]['id'] == candidates[1]['id'] or candidates[0]['text'] == candidates[1]['text']
            or not isinstance(annotations, list) or len(annotations) != 2
            or view.get('candidate_permutation') not in ([0, 1], [1, 0])
            or any(type(index) is not int for index in view['candidate_permutation'])):
        raise ValueError('Invalid typed human-dimension Choice view')
    for parent in parents:
        _frame(parent, view['source'], group_roles)
        if any(parent[key] != view[key] for key in ('source', 'role', 'revision', 'group', 'original_split', 'modality')):
            raise ValueError('Dimension view changed original parent ownership')
        if view['source'] == 'mm-rlhf' and any(parent.get(key) != view.get(key) for key in
                ('image_sha256', 'image_path', 'source_image_id', 'official_holdout_overlap')):
            raise ValueError('Dimension view changed admitted image provenance')
    if view['source'] == 'mm-rlhf':
        parent = parents[0]
        if (parent.get('primitive') != 'choice' or parent.get('adaptation') != 'best_response'
                or parent['context'] != [] or not parent['question'].startswith(MM_PREFIX)
                or view['context'] != []
                or view['question'] != 'Which response has higher visual faithfulness to the image?\n'
                   + parent['question'][len(MM_PREFIX):]):
            raise ValueError('Faithfulness view changed its original question or criterion')
    else:
        from .data import prompt_group

        for parent in parents:
            if (parent['primitive'] != 'ordinal' or parent.get('attribute') != 'correctness'
                    or not parent['context'] or parent['context'][-1]['role'] != 'assistant'
                    or parent['group'] != prompt_group(parent['context'][:-1])
                    or parent['question'] != "Rate the response's correctness on the HelpSteer2 scale from 0 to 4."
                    or parent['candidates'] != [{'id': str(i), 'text': f'correctness: {i}', 'ordinal_value': i}
                                                for i in range(5)]
                    or view['context'] != parent['context'][:-1]
                    or view['question'] != 'Which response has higher correctness on the HelpSteer2 rubric?'):
                raise ValueError('Correctness view changed its original prompt or criterion')
    scores = view.get('dimension_scores')
    if not isinstance(scores, list) or len(scores) != 2:
        raise ValueError('Missing aligned human dimension scores')
    for score, annotation, candidate in zip(scores, annotations, candidates):
        _rating(score, view['source'])
        if (not isinstance(annotation, dict) or annotation.get('field') != dimension
                or annotation.get('value') != score or not _sha(annotation.get('record_sha256'))
                or annotation.get('response_sha256') != hashlib.sha256(candidate['text'].encode()).hexdigest()):
            raise ValueError('Human score/annotation provenance mismatch')
        matching = [parent for parent in parents if parent['id'] == annotation.get('parent_id')]
        if len(matching) != 1:
            raise ValueError('Human rating is not bound to an original parent')
        if view['source'] == 'mm-rlhf':
            index = annotation.get('response_index')
            if (type(index) is not int or not 0 <= index < len(matching[0]['candidates'])
                    or matching[0]['candidates'][index] != candidate
                    or annotation.get('annotation_id') != matching[0].get('annotation_id')):
                raise ValueError('Human faithfulness is not bound to its exact original response')
        elif (type(annotation.get('row_index')) is not int or annotation['row_index'] < 0
                or matching[0]['context'][-1] != {'role': 'assistant', 'content': candidate['text']}
                or matching[0].get('original_score') != score
                or matching[0]['target'] != _ordinal_target(score)
                or candidate['id'] != _hash('response', view['source'], view['revision'], view['group'], candidate['text'])):
            raise ValueError('Human correctness is not bound to its exact original response')
    permutation = view['candidate_permutation']
    original_candidates = [candidates[index] for index in permutation]
    original_annotations = [annotations[index] for index in permutation]
    if view['source'] == 'mm-rlhf':
        if (original_annotations[0]['response_index'] >= original_annotations[1]['response_index']
                or annotations[0]['record_sha256'] != annotations[1]['record_sha256']):
            raise ValueError('Faithfulness pair changed original response order or raw row binding')
    elif (original_candidates != sorted(candidates, key=lambda candidate: candidate['id'])
            or len({annotation['parent_id'] for annotation in annotations}) != 2
            or len({annotation['row_index'] for annotation in annotations}) != 2):
        raise ValueError('Correctness pair changed canonical input-only order or distinct row binding')
    original_input = dict(view, candidates=original_candidates)
    expected_permutation = [1, 0] if int(_hash('orientation', seed, _input_key(original_input)), 16) & 1 else [0, 1]
    if permutation != expected_permutation:
        raise ValueError('Candidate orientation is not the predeclared label-independent permutation')
    preparation = view.get('parent_preparation')
    if preparation is not None and (not isinstance(preparation, dict) or not _text(preparation.get('path'))
            or not _sha(preparation.get('sha256')) or not _sha(preparation.get('manifest_sha256'))
            or type(preparation.get('records')) is not int or preparation['records'] <= 0):
        raise ValueError('Invalid optional original preparation descriptor')
    if 'parent_feature_identity_sha256' in view and not _sha(view['parent_feature_identity_sha256']):
        raise ValueError('Invalid optional original feature identity pin')
    _provenance(view.get('source_provenance'), parents[0])
    if (view.get('target') != _target(scores) or view.get('parent_ids') != [parent['id'] for parent in parents]
            or view.get('parent_record_sha256s') != [hashlib.sha256(_bytes(parent)).hexdigest() for parent in parents]
            or view.get('rating_record_ids') != [_rating_id(view['source_provenance'], annotation, candidate)
                                                for annotation, candidate in zip(annotations, candidates)]
            or view.get('parent_input_sha256') != _input_key(view) or view.get('id') != _identity(view)):
        raise ValueError('Dimension input/label/parent integrity mismatch')


def validate_dimension_view(view, *, group_roles=None, seed=SEED):
    """Validate a persisted view without raw I/O; caller proves registry/file pins.

    A supplied original group registry is authoritative. The default checks the
    row's internal inherited ownership only, not corpus membership or readiness.
    """
    _seed(seed)
    if group_roles is None:
        group_roles = {view.get('group'): view.get('role')} if isinstance(view, dict) else {}
    _check_view(view, group_roles, seed)
    return view


def _stratum_quotas(size, counts):
    present = [label for label in STRATA if counts[label]]
    quotas = {label: int(label in present and size >= len(present)) for label in STRATA}
    remaining = size - sum(quotas.values())
    weights = {label: counts[label] - quotas[label] for label in STRATA}
    population = sum(weights.values())
    if remaining:
        for label in STRATA:
            quotas[label] += remaining * weights[label] // population
        spare = size - sum(quotas.values())
        order = sorted(STRATA, key=lambda label: (-(remaining * weights[label] % population), STRATA.index(label)))
        for label in order:
            if spare and quotas[label] < counts[label]:
                quotas[label] += 1
                spare -= 1
    return quotas


def select_dimension_views(views, *, group_roles, seed=SEED):
    """Return (deep-copied views, receipt); fixed quotas fail closed, never shrink.

    Validate every supplied eligible view against original ownership, choose one
    pair per source/group using inputs only, then proportionally stratify labels.
    TEST parents must be filtered by the caller before raw annotations are read.
    """
    _seed(seed)
    grouped, seen = {}, {}
    for supplied in views:
        view = copy.deepcopy(supplied)
        _check_view(view, group_roles, seed)
        input_key = _input_key(view)
        if input_key in seen and seen[input_key] != view['id']:
            raise ValueError('Ambiguous duplicate dimension input; do not invent averaged labels')
        seen[input_key] = view['id']
        key = view['role'], view['source'], view['group']
        rank = (_hash('pair', seed, input_key), input_key)
        if key not in grouped or rank < grouped[key][0]:
            grouped[key] = rank, view
    pools = defaultdict(list)
    for _, view in grouped.values():
        stratum = 'tie' if view['target'] == [.5, .5] else 'first' if view['target'] == [1., 0.] else 'second'
        pools[view['role'], view['source'], stratum].append(view)
    deficits, populations = [], {}
    for role in ROLES:
        populations[role] = {}
        for source in SOURCES:
            counts = {label: len(pools[role, source, label]) for label in STRATA}
            populations[role][source] = counts
            required, available = PHASE1_BUDGETS[role][source], sum(counts.values())
            if available < required:
                deficits.append({'role': role, 'source': source, 'required_groups': required, 'available_groups': available})
    if deficits:
        raise QuotaInfeasible(deficits)
    selected, label_quotas = [], {}
    for role in ROLES:
        label_quotas[role] = {}
        for source in SOURCES:
            quotas = _stratum_quotas(PHASE1_BUDGETS[role][source], populations[role][source])
            label_quotas[role][source] = quotas
            for label, size in quotas.items():
                ordered = sorted(pools[role, source, label],
                                 key=lambda view: (_hash('group', seed, source, view['group']), view['group']))
                selected.extend(ordered[:size])
    selected.sort(key=lambda view: (ROLES.index(view['role']), SOURCES.index(view['source']), view['group']))
    receipt = {'adapter_version': ADAPTER_VERSION, 'dataset_scope_revision': REVISION,
               'outputs': dict(OUTPUTS), 'manifests': dict(MANIFESTS),
               'seed': seed, 'budgets': copy.deepcopy(PHASE1_BUDGETS),
               'records': len(selected), 'population_group_labels': populations, 'selected_group_labels': label_quotas,
               'membership_sha256': _hash([view['id'] for view in selected]),
               'selection': 'one input-hashed pair per original group; proportional label strata; no replacement',
               'scope': 'existing training-source FIT/DEV/CAL groups only; no TEST or external data access',
               'qualification': 'pure adapter selection only; original corpus/raw-file proof remains required'}
    return copy.deepcopy(selected), receipt
