from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_vllm_assisted_serving import AssistedHead, assisted_service
from test_vllm_service import decision, run


@pytest.fixture
def client(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/kev_vllm_e2e_client.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('_synthetic_vllm_e2e', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def response(client):
    async def case():
        service = assisted_service()
        class RawHead(AssistedHead):
            def score(self, *args, **kwargs):
                logits, raw, _ = super().score(*args, **kwargs)
                return logits, raw, raw[:]
        service.head = RawHead()
        request = decision()
        try:
            value = await service.decide(request, 'synthetic-http')
            value.update(client.canonical.FLAGS)
            value.update(head_execution_device='cuda:0', head_sha256=client.canonical.HEAD_SHA,
                         head_manifest_sha256=client.canonical.PLAN_SHA, calibration_id='uncalibrated:T1')
            client.BINDING = deepcopy(value['provenance'])
            return value, request.model_dump()
        finally:
            await service.close()
    return run(case())


def test_wire_changes_only_model_and_keeps_canonical_body(client):
    client.MODEL = 'vllm-nvidia'
    request = {'model': 'native-4.05', 'head_id': 'reference', 'candidates': [{'id': 'a', 'text': 'A'}]}
    before = deepcopy(request)
    wire = client.wire_request(request)
    assert request == before and wire['model'] == 'vllm-nvidia'
    assert {**wire, 'model': 'native-4.05'} == before
    with pytest.raises(ValueError):
        client.wire_request(wire)


def test_real_vllm_schema_is_accepted_without_native_consumption_claims(client, response):
    value, request = response
    scores = client.response_check(value, request, 'synthetic-http')
    assert len(scores) == len(request['candidates'])
    assert value['execution']['consumed_tokens'] is None
    assert value['execution']['forwarded_tokens'] is None


@pytest.mark.parametrize('mutation', ('missing', 'projection', 'generated_bool', 'source',
                                     'query_labels', 'candidate_labels', 'totals', 'native_consumption',
                                     'provenance', 'raw', 'ordering', 'request'))
def test_bad_worker_receipt_or_response_is_rejected(client, response, mutation):
    value, request = response
    value = deepcopy(value)
    query, candidate = value['execution']['requests'][:2]
    if mutation == 'missing':
        del query['fuzev_readout_receipt']
    elif mutation == 'projection':
        candidate['fuzev_readout_receipt']['lm_projection_calls'] = 1
    elif mutation == 'generated_bool':
        query['fuzev_readout_receipt']['generated_token_count'] = False
    elif mutation == 'source':
        source = query['fuzev_readout_receipt']['worker_loaded_sources']
        source[next(iter(source))] = '4' * 64
    elif mutation == 'query_labels':
        query['fuzev_readout_receipt']['baseline_token_ids'] = [1, 1]
    elif mutation == 'candidate_labels':
        candidate['fuzev_readout_receipt']['baseline_token_ids'] = [1, 2]
    elif mutation == 'totals':
        value['execution']['submitted_tokens'] += 1
    elif mutation == 'native_consumption':
        value['execution']['consumed_tokens'] = 5
    elif mutation == 'provenance':
        value['provenance']['cell'] = 'wrong'
    elif mutation == 'raw':
        value['scores'][0]['calibrated_probability'] = .6
    elif mutation == 'ordering':
        value['scores'].reverse()
    elif mutation == 'request':
        value['id'] = 'other-http'
    with pytest.raises(ValueError):
        client.response_check(value, request, 'synthetic-http')
