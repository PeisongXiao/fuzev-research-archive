import pytest

from fuzev_research.contracts import DecisionRequest
from fuzev_research.prefix import PrefixRegistry, identity_digest
from fuzev_research.service_contracts import ServiceDecisionRequest


def request(**kwargs):
    return DecisionRequest(model="reference", primitive="choice", question="Which fits?",
                           candidates=[{"id": "a", "text": "offline"}, {"id": "b", "text": "online"}],
                           **kwargs)


def test_request_requires_exactly_one_context_source():
    assert request(messages=[{"role": "user", "content": "Compare tools"}])
    assert request(prefix_id="prefix")
    with pytest.raises(ValueError):
        request()
    with pytest.raises(ValueError):
        request(messages=[], prefix_id="prefix")


def test_noul_and_ordinal_semantics_are_not_arbitrary_slot_labels():
    with pytest.raises(ValueError, match="yes and no"):
        DecisionRequest(model="reference", primitive="noul", question="Supported?", messages=[],
                        candidates=[{"id": "a", "text": "yes"}, {"id": "b", "text": "no"}])
    with pytest.raises(ValueError, match="increasing"):
        DecisionRequest(model="reference", primitive="ordinal", question="Quality?", messages=[],
                        candidates=[{"id": "a", "text": "low", "ordinal_value": 2},
                                    {"id": "b", "text": "high", "ordinal_value": 1}])


@pytest.mark.parametrize("schema", [DecisionRequest, ServiceDecisionRequest])
@pytest.mark.parametrize("texts", [("No", "Yes"), ("Yes", "Yes"),
                                   ("unrelated", "No"), (" Yes", "No")])
def test_noul_rejects_text_that_changes_boolean_identifier_meaning(schema, texts):
    with pytest.raises(ValueError, match="text must match"):
        schema(model="reference", primitive="noul", question="Supported?", messages=[],
               candidates=[{"id": "yes", "text": texts[0]}, {"id": "no", "text": texts[1]}])


@pytest.mark.parametrize("schema", [DecisionRequest, ServiceDecisionRequest])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("texts", [("Yes", "No"), ("yes", "no"), ("YES", "NO")])
def test_noul_keeps_boolean_option_order_and_text_bytes(schema, reverse, texts):
    candidates = [{"id": "yes", "text": texts[0]}, {"id": "no", "text": texts[1]}]
    if reverse:
        candidates.reverse()
    value = schema(model="reference", primitive="noul", question="Supported?", messages=[],
                   candidates=candidates)
    assert [{"id": candidate.id, "text": candidate.text} for candidate in value.candidates] == candidates


def test_prefix_keeps_unconsumed_final_token_and_is_immutable():
    registry = PrefixRegistry("identity")
    prefix = registry.retain([1, 2, 3], consumed=2, image_hashes=["image-a"])
    branch_a = registry.get(prefix.id, "identity")
    branch_b = registry.get(prefix.id, "identity")
    assert branch_a.token_ids == branch_b.token_ids == (1, 2, 3)
    assert prefix.consumed == 2
    with pytest.raises(ValueError, match="incompatible"):
        registry.get(prefix.id, "different-template")


def test_prefix_expiry_and_capacity():
    now = [0]
    registry = PrefixRegistry("identity", capacity=1, ttl=5, clock=lambda: now[0])
    first = registry.retain([1], 1)
    second = registry.retain([2], 1)
    with pytest.raises(KeyError):
        registry.get(first.id, "identity")
    now[0] = 5
    with pytest.raises(KeyError):
        registry.get(second.id, "identity")
    assert identity_digest({"a": 1, "b": 2}) == identity_digest({"b": 2, "a": 1})
