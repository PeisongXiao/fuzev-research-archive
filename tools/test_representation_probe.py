
import torch

import representation_probe as probe
from train_shared_head import initialize


GEOMETRY = dict(hidden_size=4, projection_size=3, query_adapter_size=2, key_adapter_size=2)


def row(identifier, count=3, role="fit"):
    query = torch.tensor([0.1, -0.3, 0.2, 0.4])
    keys = torch.arange(count * 4).reshape(count, 4).float() / 20 - 0.3
    ids = [f"c{i}" for i in range(count)]
    target = torch.zeros(count)
    target[-1] = 1
    return dict(id=identifier, group=identifier, role=role, primitive="choice", modality="image_text",
                candidate_ids=ids, acceptable_candidate_ids=[ids[-1]], target=target,
                baseline=torch.linspace(0.4, -0.3, count), views={
                    view: dict(query=query.clone(), candidates=keys.clone()) for view in ("control",)})



def test_full_slate_scores_match_shared_head():
    rows = [row("short", 2), row("long", 5)]
    model = initialize(GEOMETRY, 17).eval()
    actual = probe.score_probe(model, rows)
    for record, original in zip(actual, rows):
        query, candidates = original["views"]["control"].values()
        expected = model(None, query[None], candidates[None], baseline=original["baseline"][None])[0]
        torch.testing.assert_close(torch.tensor(record["utilities"]), expected)
        torch.testing.assert_close(torch.tensor(record["probabilities"]), expected.softmax(0))


def test_bounded_scoring_preserves_complete_slates():
    rows = [row(f"test-{i}", 2 + i, "test") for i in range(5)]
    model = initialize(GEOMETRY, 17)
    actual = probe.score_probe(model, rows, batch_size=2)
    expected = probe.score_probe(model, rows, batch_size=1)
    for result, single in zip(actual, expected):
        assert result["candidate_ids"] == single["candidate_ids"]
        assert result["selected_id"] == single["selected_id"]
        torch.testing.assert_close(torch.tensor(result["utilities"]), torch.tensor(single["utilities"]))
        torch.testing.assert_close(torch.tensor(result["probabilities"]), torch.tensor(single["probabilities"]))
