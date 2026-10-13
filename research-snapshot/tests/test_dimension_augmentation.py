import copy
import hashlib
import json
from collections import Counter

import pytest

from fuzev_research import dimension_augmentation as adapter
from fuzev_research.data import conversation, prompt_group, stable_id


def canonical(parent):
    parent['id'] = stable_id(parent['source'], parent['group'], parent['context'], parent['question'],
                             json.dumps(parent['candidates']), parent['primitive'], parent['target'])
    return parent


def pin(source):
    return {'source': source, 'revision': ('a' if source == 'mm-rlhf' else 'b') * 40,
            'filename': 'data.jsonl' if source == 'mm-rlhf' else 'train.jsonl.gz', 'sha256': 'c' * 64}


def mm(number=0, role='fit', ratings=(1, 5, 3, 3), ranks=None):
    responses = [f'Original response {index} for image {number}' for index in range(len(ratings))]
    ranks = ranks or [1] + [2] * (len(responses) - 1)
    raw = {'id': f'annotation-{role}-{number}', 'image': f'image-{role}-{number}',
           'question': f'What appears in image {number}?', 'models_output': responses,
           'faithfulness': list(ratings), 'final_ranking': list(ranks),
           'critique': 'HUMAN_ANNOTATION_NOT_INPUT', 'preference_strength': -99}
    parent = {'source': 'mm-rlhf', 'revision': pin('mm-rlhf')['revision'],
              'group': f'image:{role}-{number}', 'role': role, 'original_split': 'train',
              'modality': 'image', 'context': [], 'question': adapter.MM_PREFIX + raw['question'],
              'candidates': [{'id': str(index), 'text': text} for index, text in enumerate(responses)],
              'target': [float(rank == min(ranks)) / ranks.count(min(ranks)) for rank in ranks],
              'primitive': 'choice', 'adaptation': 'best_response', 'annotation_id': raw['id'],
              'source_image_id': raw['image'], 'image_path': f'/original/image-{role}-{number}.png',
              'image_sha256': 'd' * 64, 'official_holdout_overlap': False,
              'dataset_scope_revision': '2026-10-02-v2', 'dataset_purpose': 'training',
              'evaluation_scope': 'training'}
    return canonical(parent), raw


def hs2(number=0, role='fit', scores=(0, 4), prompt=None):
    prompt = prompt or f'Original request {role} {number}'
    raws = [{'prompt': prompt, 'response': f'Original {index} response {number}',
             'correctness': score, 'helpfulness': 4, 'preference_strength': -99,
             'reference_answer': 'HUMAN_ANNOTATION_NOT_INPUT'} for index, score in enumerate(scores)]
    parents = []
    for raw in raws:
        score = raw['correctness']
        messages = conversation(raw['prompt'])
        parent = {'source': 'helpsteer2', 'revision': pin('helpsteer2')['revision'],
                  'group': prompt_group(messages), 'role': role, 'original_split': 'train',
                  'modality': 'text', 'context': [*messages, {'role': 'assistant', 'content': raw['response']}],
                  'question': "Rate the response's correctness on the HelpSteer2 scale from 0 to 4.",
                  'candidates': [{'id': str(i), 'text': f'correctness: {i}', 'ordinal_value': i} for i in range(5)],
                  'target': adapter._ordinal_target(score), 'primitive': 'ordinal',
                  'attribute': 'correctness', 'original_score': score}
        parents.append(canonical(parent))
    return parents, raws


def mm_views(parent, raw, provenance=None):
    return adapter.mm_faithfulness_views(parent, raw, provenance or pin('mm-rlhf'),
                                         group_roles={parent['group']: parent['role']})


def hs_view(parents, raws, provenance=None, indices=(10, 11)):
    return adapter.hs2_correctness_view(parents, raws, provenance or pin('helpsteer2'), row_indices=indices,
                                        group_roles={parents[0]['group']: parents[0]['role']})


def test_constants_are_the_single_locked_integration_contract():
    assert adapter.REVISION == adapter.DIMENSION_REVISION == '2026-10-05-v4-dimension-choice'
    assert adapter.ADAPTATIONS == {'mm-rlhf': 'faithfulness_pairwise_choice',
                                  'helpsteer2': 'correctness_pairwise_choice'}
    assert adapter.OUTPUTS == {'text': 'dimension-choice-text.jsonl', 'image': 'dimension-choice-images.jsonl'}
    assert adapter.MANIFESTS == {'text': 'dimension-choice-text-splits.json',
                                'image': 'dimension-choice-images-splits.json'}
    assert adapter.PROVENANCE_FIELDS == adapter.DIMENSION_METADATA
    assert adapter.PROVENANCE_FIELDS == adapter.REQUIRED_PROVENANCE_FIELDS + adapter.OPTIONAL_PROVENANCE_FIELDS
    assert len(set(adapter.PROVENANCE_FIELDS)) == len(adapter.PROVENANCE_FIELDS)
    assert sum(sum(quota.values()) for quota in adapter.PHASE1_BUDGETS.values()) == 6144


def test_mm_numeric_higher_faithfulness_not_lower_rank_or_signed_preference():
    parent, raw = mm()
    saved = copy.deepcopy((parent, raw))
    views = mm_views(parent, raw)
    assert len(views) == 6
    for view in views:
        assert set(adapter.REQUIRED_PROVENANCE_FIELDS) <= view.keys()
        assert adapter.validate_dimension_view(view) is view
        assert view['target'] == adapter._target(view['dimension_scores'])
        assert view['dataset_scope_revision'] == adapter.REVISION
        assert view['parent_records'] == [parent]
        assert view['source_provenance'] == pin('mm-rlhf')
        assert view['context'] == []
        assert view['question'] == 'Which response has higher visual faithfulness to the image?\n' + raw['question']
        assert 'HUMAN_ANNOTATION_NOT_INPUT' not in json.dumps([view['context'], view['question'], view['candidates']])
        for candidate, annotation, score in zip(view['candidates'], view['source_annotations'], view['dimension_scores']):
            assert annotation['value'] == score == raw['faithfulness'][int(candidate['id'])]
            assert annotation['record_sha256'] == hashlib.sha256(adapter._bytes(raw)).hexdigest()
            assert annotation['response_sha256'] == hashlib.sha256(candidate['text'].encode()).hexdigest()
    opposite = next(view for view in views if {candidate['id'] for candidate in view['candidates']} == {'0', '1'})
    winner = opposite['candidates'][opposite['target'].index(1.)]['id']
    assert winner == '1' and parent['target'][0] == 1.
    tied = next(view for view in views if {candidate['id'] for candidate in view['candidates']} == {'2', '3'})
    assert tied['target'] == [.5, .5]
    assert (parent, raw) == saved


@pytest.mark.parametrize('source', adapter.SOURCES)
def test_orientation_is_input_only_and_raw_rating_ids_bind_changed_labels(source):
    if source == 'mm-rlhf':
        parent, raw = mm(ratings=(1, 5))
        first = mm_views(parent, raw)[0]
        raw['faithfulness'] = [5, 1]
        second = mm_views(parent, raw)[0]
    else:
        parents, raws = hs2(scores=(0, 4))
        first = hs_view(parents, raws)
        changed_parents, changed_raws = hs2(scores=(4, 0))
        second = hs_view(changed_parents, changed_raws)
    assert first['candidates'] == second['candidates']
    assert first['candidate_permutation'] == second['candidate_permutation']
    assert first['parent_input_sha256'] == second['parent_input_sha256']
    assert first['target'] == second['target'][::-1]
    assert first['id'] != second['id'] and first['rating_record_ids'] != second['rating_record_ids']


def test_orientation_uses_both_sides_not_better_first():
    views = [mm_views(*mm(number=i, ratings=(1, 5)))[0] for i in range(32)]
    assert {tuple(view['candidate_permutation']) for view in views} == {(0, 1), (1, 0)}
    assert {tuple(view['target']) for view in views} == {(1., 0.), (0., 1.)}


@pytest.mark.parametrize('scores', [(0, 4), (4, 0), (2, 2), (1.5, 2.5)])
def test_hs2_exact_human_correctness_and_caller_order_invariance(scores):
    parents, raws = hs2(scores=scores)
    saved = copy.deepcopy((parents, raws))
    view = hs_view(parents, raws)
    reversed_view = hs_view(parents[::-1], raws[::-1], indices=(11, 10))
    assert view == reversed_view
    assert adapter.validate_dimension_view(view) is view
    assert view['context'] == parents[0]['context'][:-1]
    assert view['target'] == adapter._target(view['dimension_scores'])
    assert sorted(view['dimension_scores']) == sorted(scores)
    assert 'original_score' not in view and 'attribute' not in view
    for candidate, annotation in zip(view['candidates'], view['source_annotations']):
        raw = raws[annotation['row_index'] - 10]
        assert candidate['text'] == raw['response']
        assert annotation['value'] == raw['correctness']
        assert annotation['record_sha256'] == hashlib.sha256(adapter._bytes(raw)).hexdigest()
    assert 'HUMAN_ANNOTATION_NOT_INPUT' not in json.dumps([view['context'], view['question'], view['candidates']])
    assert (parents, raws) == saved


def test_fractional_mm_ratings_are_not_rounded_rescaled_or_invented_truth():
    view = mm_views(*mm(ratings=(2.25, 2.5)))[0]
    assert sorted(view['dimension_scores']) == [2.25, 2.5]
    assert sorted(view['target']) == [0., 1.]
    assert view['label_semantics'] == 'relative_human_faithfulness_rating'
    assert view['primitive'] == 'choice'


@pytest.mark.parametrize('source', adapter.SOURCES)
@pytest.mark.parametrize('bad', [None, True, False, '3', float('nan'), float('inf'), -1, 6])
def test_invalid_rubric_values_fail_closed(source, bad):
    if source == 'mm-rlhf':
        parent, raw = mm(ratings=(1, 5))
        raw['faithfulness'][0] = bad
        with pytest.raises(ValueError):
            mm_views(parent, raw)
    else:
        parents, raws = hs2()
        raws[0]['correctness'] = bad
        with pytest.raises(ValueError):
            hs_view(parents, raws)


@pytest.mark.parametrize('fields', [
    {'role': 'test'}, {'original_split': 'validation'}, {'dataset_purpose': 'external'},
    {'evaluation_scope': 'external'}, {'source': 'vl-rewardbench'}, {'revision': ''}, {'group': ''},
    {'official_holdout_overlap': True}, {'official_holdout_overlap': None}, {'image_sha256': 'bad'},
    {'modality': 'text'}, {'id': 'unbound-id'}, {'target': [True, False]}, {'adaptation': 'pairwise_preference'},
])
def test_mm_unadmitted_or_noncanonical_parent_is_rejected(fields):
    parent, raw = mm(ratings=(1, 5))
    parent.update(fields)
    with pytest.raises(ValueError):
        mm_views(parent, raw)


@pytest.mark.parametrize('fields', [
    {'id': 'wrong-annotation'}, {'image': 'wrong-image'}, {'question': 'changed-question'},
    {'models_output': ['changed-text', 'second']}, {'faithfulness': [3]},
    {'final_ranking': [2, 1]}, {'final_ranking': [0, 1]}, {'final_ranking': [True, 2]},
])
def test_mm_raw_parent_alignment_and_original_rank_provenance_are_required(fields):
    parent, raw = mm(ratings=(1, 5))
    raw.update(fields)
    with pytest.raises(ValueError):
        mm_views(parent, raw)


@pytest.mark.parametrize('source', adapter.SOURCES)
@pytest.mark.parametrize('fields', [{'source': 'qasper'}, {'revision': 'different-revision'},
                                    {'filename': 'validation.jsonl.gz'}, {'sha256': 'bad'}])
def test_pinned_raw_source_revision_file_and_sha_are_required(source, fields):
    provenance = {**pin(source), **fields}
    with pytest.raises(ValueError):
        if source == 'mm-rlhf':
            mm_views(*mm(ratings=(1, 5)), provenance)
        else:
            hs_view(*hs2(), provenance)


def test_group_owner_registry_is_authoritative_before_raw_read():
    parent, raw = mm(ratings=(1, 5))
    for registry in ({}, {parent['group']: 'test'}, {parent['group']: 'development'}):
        with pytest.raises(ValueError, match='ownership'):
            adapter.mm_faithfulness_views(parent, raw, pin('mm-rlhf'), group_roles=registry)
    view = mm_views(parent, raw)[0]
    with pytest.raises(ValueError, match='ownership'):
        adapter.validate_dimension_view(view, group_roles={parent['group']: 'test'})


@pytest.mark.parametrize('problem', ['prompt', 'response', 'label', 'attribute', 'question', 'revision', 'role', 'id'])
def test_hs2_binding_is_to_original_correctness_response_not_other_attribute(problem):
    parents, raws = hs2()
    if problem in {'prompt', 'response'}:
        raws[0][problem] += ' changed'
    elif problem == 'label':
        raws[0]['correctness'] = 1
    elif problem == 'attribute':
        parents[0]['attribute'] = 'helpfulness'
    elif problem == 'question':
        parents[0]['question'] = 'Rate helpfulness.'
        canonical(parents[0])
    else:
        parents[0][problem] = 'changed'
    with pytest.raises(ValueError):
        hs_view(parents, raws)


@pytest.mark.parametrize('indices', [(1, 1), (-1, 2), (True, 2), (1,), ('1', 2)])
def test_hs2_distinct_numeric_raw_row_locators_required(indices):
    with pytest.raises(ValueError):
        hs_view(*hs2(), indices=indices)


def test_duplicate_response_conflicts_are_not_averaged():
    parent, raw = mm(ratings=(1, 5))
    raw['models_output'][1] = raw['models_output'][0]
    parent['candidates'][1]['text'] = raw['models_output'][0]
    canonical(parent)
    with pytest.raises(ValueError, match='conflicting'):
        mm_views(parent, raw)
    raw['faithfulness'][1] = 1
    assert mm_views(parent, raw) == []
    parents, raws = hs2()
    raws[1]['response'] = raws[0]['response']
    parents[1]['context'][-1]['content'] = raws[0]['response']
    canonical(parents[1])
    with pytest.raises(ValueError, match='distinct'):
        hs_view(parents, raws)


@pytest.mark.parametrize('source', adapter.SOURCES)
def test_outputs_are_deep_copied_no_caller_aliases(source):
    if source == 'mm-rlhf':
        parent, raw = mm(ratings=(1, 5))
        provenance = pin(source)
        view = mm_views(parent, raw, provenance)[0]
        saved = copy.deepcopy(view)
        parent['candidates'][0]['text'] = 'changed caller parent'
        raw['faithfulness'][0] = 2
    else:
        parents, raws = hs2()
        provenance = pin(source)
        view = hs_view(parents, raws, provenance)
        saved = copy.deepcopy(view)
        parents[0]['context'][0]['content'] = 'changed caller parent'
        raws[0]['correctness'] = 2
    provenance['sha256'] = 'e' * 64
    assert view == saved
    view['candidates'][0]['text'] = 'changed child'
    assert view['parent_records'] == saved['parent_records']


@pytest.mark.parametrize('field', adapter.REQUIRED_PROVENANCE_FIELDS)
def test_persisted_view_requires_every_exported_required_field(field):
    view = mm_views(*mm(ratings=(1, 5)))[0]
    del view[field]
    with pytest.raises(ValueError):
        adapter.validate_dimension_view(view)


@pytest.mark.parametrize('field,value', [
    ('dimension_scores', [3, 3]), ('source_annotations', []), ('parent_records', []),
    ('parent_ids', ['bad']), ('parent_record_sha256s', ['a' * 64]), ('parent_input_sha256', 'b' * 64),
    ('rating_record_ids', ['c' * 64, 'd' * 64]), ('target', [.5, .5]), ('dimension', 'helpfulness'),
    ('question', 'Does this response tell the truth?'), ('context', [{'role': 'assistant', 'content': 'annotation'}]),
    ('candidate_permutation', [False, True]), ('id', 'tampered'),
])
def test_persisted_view_tampering_is_rejected(field, value):
    view = mm_views(*mm(ratings=(1, 5)))[0]
    view[field] = value
    with pytest.raises(ValueError):
        adapter.validate_dimension_view(view)


def test_structural_mapping_guards_survive_rehashed_view_id():
    view = hs_view(*hs2())
    view['source_annotations'][1]['row_index'] = view['source_annotations'][0]['row_index']
    view['rating_record_ids'] = [adapter._rating_id(view['source_provenance'], annotation, candidate)
                               for annotation, candidate in zip(view['source_annotations'], view['candidates'])]
    view['id'] = adapter._identity(view)
    with pytest.raises(ValueError, match='distinct row'):
        adapter.validate_dimension_view(view)
    view = mm_views(*mm(ratings=(1, 5)))[0]
    view['question'] = 'Different task'
    view['parent_input_sha256'] = adapter._input_key(view)
    view['id'] = adapter._identity(view)
    with pytest.raises(ValueError, match='criterion'):
        adapter.validate_dimension_view(view)


def test_optional_parent_pins_are_retained_but_not_claimed_as_corpus_proof():
    view = mm_views(*mm(ratings=(1, 5)))[0]
    view['parent_preparation'] = {'path': '/original/images.jsonl', 'sha256': 'a' * 64,
                                  'manifest_sha256': 'b' * 64, 'records': 100}
    view['parent_feature_identity_sha256'] = 'c' * 64
    assert adapter.validate_dimension_view(view) is view
    for field, value in [('parent_preparation', {}), ('parent_feature_identity_sha256', 'bad')]:
        changed = dict(view, **{field: value})
        with pytest.raises(ValueError):
            adapter.validate_dimension_view(changed)


@pytest.mark.parametrize('source', adapter.SOURCES)
def test_sorted_json_roundtrip_preserves_legacy_parent_identity(source):
    view = mm_views(*mm(ratings=(1, 5)))[0] if source == 'mm-rlhf' else hs_view(*hs2())
    persisted = json.loads(json.dumps(view, sort_keys=True))
    assert adapter.validate_dimension_view(persisted) is persisted
    assert persisted == view
    parent = persisted['parent_records'][0]
    parent['candidates'][0]['unexpected'] = 'must-not-ignore-fields'
    with pytest.raises(ValueError, match='canonical source construction'):
        adapter.validate_dimension_view(persisted)


@pytest.fixture(scope='module')
def phase1_pool():
    views, owners = [], {}
    for role in adapter.ROLES:
        for source in adapter.SOURCES:
            for number in range(adapter.PHASE1_BUDGETS[role][source]):
                if source == 'mm-rlhf':
                    parent, raw = mm(number, role, ratings=((1, 5), (5, 1), (3, 3))[number % 3])
                    views.extend(mm_views(parent, raw))
                    owners[parent['group']] = role
                else:
                    parents, raws = hs2(number, role, scores=((0, 4), (4, 0), (2, 2))[number % 3])
                    views.append(hs_view(parents, raws))
                    owners[parents[0]['group']] = role
    return views, owners


def test_fixed_6144_quota_no_replay_role_leakage_or_aliases(phase1_pool):
    views, owners = phase1_pool
    selected, receipt = adapter.select_dimension_views(views, group_roles=owners)
    assert len(selected) == receipt['records'] == 6144
    assert len({view['id'] for view in selected}) == len(selected)
    assert len({view['group'] for view in selected}) == len(selected)
    counts = Counter((view['role'], view['source']) for view in selected)
    for role in adapter.ROLES:
        for source in adapter.SOURCES:
            assert counts[role, source] == adapter.PHASE1_BUDGETS[role][source]
            assert sum(receipt['selected_group_labels'][role][source].values()) == counts[role, source]
            assert receipt['selected_group_labels'][role][source] == receipt['population_group_labels'][role][source]
    assert all(owners[view['group']] == view['role'] for view in selected)
    assert receipt['dataset_scope_revision'] == adapter.REVISION
    assert 'corpus/raw-file proof remains required' in receipt['qualification']
    selected[0]['parent_records'][0]['question'] = 'changed child'
    assert views[0]['parent_records'][0]['question'] != 'changed child'
    reversed_selected, reversed_receipt = adapter.select_dimension_views(reversed(views), group_roles=owners)
    assert [view['id'] for view in reversed_selected] == [view['id'] for view in selected]
    assert receipt == reversed_receipt


def test_pair_selection_within_group_is_label_independent(phase1_pool):
    views, owners = phase1_pool
    parent, raw = mm(0, 'fit', ratings=(1, 5, 3))
    candidates = mm_views(parent, raw)
    alternate = dict(raw, faithfulness=[5, 1, 4])
    changed = mm_views(parent, alternate)
    others = [view for view in views if view['group'] != parent['group']]
    first, _ = adapter.select_dimension_views([*others, *candidates], group_roles=owners)
    second, _ = adapter.select_dimension_views([*others, *changed], group_roles=owners)
    first_pair = next(view for view in first if view['group'] == parent['group'])
    second_pair = next(view for view in second if view['group'] == parent['group'])
    assert first_pair['candidates'] == second_pair['candidates']
    assert first_pair['parent_input_sha256'] == second_pair['parent_input_sha256']
    assert first_pair['id'] != second_pair['id']


def test_quota_infeasible_reports_without_reduction_or_duplicate_inflation():
    view = mm_views(*mm(ratings=(1, 5)))[0]
    with pytest.raises(adapter.QuotaInfeasible) as error:
        adapter.select_dimension_views([view] * 10, group_roles={view['group']: 'fit'})
    assert len(error.value.deficits) == 6
    mm_fit = next(item for item in error.value.deficits if item['source'] == 'mm-rlhf' and item['role'] == 'fit')
    assert mm_fit == {'source': 'mm-rlhf', 'role': 'fit', 'required_groups': 3072, 'available_groups': 1}


def test_ambiguous_same_input_different_annotations_are_not_averaged():
    parent, raw = mm(ratings=(1, 5))
    first = mm_views(parent, raw)[0]
    raw['faithfulness'] = [5, 1]
    second = mm_views(parent, raw)[0]
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        adapter.select_dimension_views([first, second], group_roles={parent['group']: 'fit'})


@pytest.mark.parametrize('size,counts', [(20, {'first': 70, 'second': 20, 'tie': 10}),
                                        (1, {'first': 0, 'second': 0, 'tie': 5}),
                                        (3, {'first': 1, 'second': 1, 'tie': 1})])
def test_strata_are_proportional_bounded_and_cover_present_labels(size, counts):
    quota = adapter._stratum_quotas(size, counts)
    assert sum(quota.values()) == size
    assert all(0 <= quota[label] <= counts[label] for label in adapter.STRATA)
    if size >= sum(bool(count) for count in counts.values()):
        assert all(quota[label] >= 1 for label in adapter.STRATA if counts[label])
    if size == 20:
        assert quota == {'first': 13, 'second': 4, 'tie': 3}


def test_adapter_performs_no_file_or_model_access(monkeypatch):
    parent, raw = mm(ratings=(1, 5))
    parents, raws = hs2()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure adapter must not read files or acquire data')
    monkeypatch.setattr('builtins.open', forbidden)
    monkeypatch.setattr('fuzev_research.data.download', forbidden)
    assert adapter.validate_dimension_view(mm_views(parent, raw)[0])
    assert adapter.validate_dimension_view(hs_view(parents, raws))
