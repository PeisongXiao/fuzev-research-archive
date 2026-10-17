"""Synthetic CPU behavior checks; no backbone, external data or feature extraction."""

import copy
import json
import math

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fuzev.encoding import (VERSION, LIST_AWARE_VERSION, LABEL_CAPACITY,
                            query_suffix, list_aware_query_suffix, candidate_suffix)
from fuzev.head import ServingHead, read_artifacts
from ragged_features import (RaggedFeatures, RaggedWriter, batches, collate, import_legacy,
                             main as replay_main, replay_config)
from train_shared_head import initialize, export_bundle as export_v1
from train_shared_head_v2 import (GEOMETRY, export_bundle, fit_temperature,
                                  load_model, load_rows, loss_components, main, report_scores,
                                  row_masses, utilities, write_json)


def config(maximum=27, hidden=4):
    feature = {"tap": "final-pre-lm", "pooling": "last-token"}
    return {"schema": "fuzev-shared-head-package-v2", "architecture_version": "fuzev-shared-decision-head-v1",
            "encoding_version": LIST_AWARE_VERSION, "hidden_size": hidden, "projection_size": 3,
            "query_adapter_size": 2, "key_adapter_size": 2,
            "features": {"query": feature, "candidate": feature},
            "baseline": {"labelled": "query-label-logits", "label_free": "zero"},
            "capacities": {"choice": {"text": {"implemented_max": maximum, "validated_max": 2}}},
            "residual_strength": {"text": 1.}}


def row(identifier, k, *, role="fit", group=None, graded=False):
    candidates = [{"id": f"c{i}", "text": f"Candidate {i}"} for i in range(k)]
    request = {"primitive": "choice", "messages": [{"role": "user", "content": "A synthetic context."}],
               "question": "Which candidate?", "additional_context": "", "candidates": candidates}
    suffix = query_suffix if k <= LABEL_CAPACITY else list_aware_query_suffix
    result = {"id": identifier, "group": group or identifier, "role": role, "primitive": "choice",
              "modality": "text", "source": "synthetic", "revision": "synthetic", "request": request,
              "rendering": {"prefix": "<|im_start|>user\nA synthetic context.<|im_end|>\n",
                            "query_suffix": suffix(request["question"], candidates),
                            "candidate_suffixes": [candidate_suffix(c) for c in candidates],
                            "encoding_version": LIST_AWARE_VERSION},
              "target": [1. / k] * k}
    if graded:
        result["grades"] = [2., 2.] + [1.] * (k - 3) + ([0.] if k > 2 else [])
        result["target"] = [.5, .5] + [0.] * (k - 2)
        result["acceptable_candidate_ids"] = ["c0", "c1"]
    return result


def values(item, hidden=4):
    k = len(item["target"])
    return (torch.arange(hidden).float() / 7,
            torch.arange(k * hidden).reshape(k, hidden).float().remainder(11) / 11,
            torch.linspace(-.4, .7, k) if k <= LABEL_CAPACITY else torch.zeros(k))


def store(path, cfg, rows):
    writer = RaggedWriter(path, cfg, shard_bytes=1 << 20)
    for item in rows:
        writer.add(item, *values(item, cfg["hidden_size"]))
    writer.flush()
    return path


def source_bundle(path, cfg):
    geometry = {name: cfg[name] for name in GEOMETRY}
    model = initialize(geometry, 5)
    export_v1(path, model.state_dict(), dict(geometry, primitives=["choice"], max_options=16),
              {"choice": 1.}, {"encoding_version": VERSION})
    return path


def test_ragged_resume_preserves_completed_rows_and_request_fields(tmp_path):
    cfg = config()
    first, second, third = row("a", 2), row("b", 5, graded=True), row("c", 27)
    path = store(tmp_path / "features", cfg, [first, second])
    saved = {p.name: p.read_bytes() for p in path.iterdir()}
    with safe_open(path / "shard-000000.safetensors", framework="pt") as handle:
        assert handle.get_tensor("offsets").tolist() == [0, 2, 7]
        assert handle.get_slice("candidates").get_shape() == [7, 4]
    save_file({"unfinished": torch.ones(1)}, path / "shard-000001.safetensors")
    with pytest.warns(UserWarning, match="Incomplete tail"):
        writer = RaggedWriter(path, cfg, shard_bytes=1 << 20)
    assert writer.contains(first)
    assert not writer.add(first, *values(first))
    changed = copy.deepcopy(first)
    changed["request"]["candidates"][0]["text"] = "Changed candidate"
    changed["rendering"]["query_suffix"] = query_suffix(changed["request"]["question"], changed["request"]["candidates"])
    changed["rendering"]["candidate_suffixes"][0] = candidate_suffix(changed["request"]["candidates"][0])
    with pytest.raises(ValueError, match="changed metadata/request/rendering"):
        writer.contains(changed)
    changed = copy.deepcopy(first)
    changed["rendering"]["prefix"] += "A changed prefix"
    with pytest.raises(ValueError, match="changed metadata/request/rendering"):
        writer.contains(changed)
    writer.add(third, *values(third))
    writer.flush()
    assert all((path / name).read_bytes() == content for name, content in saved.items())
    dataset = RaggedFeatures(path, cfg)
    assert [ref["k"] for ref in dataset.rows] == [2, 5, 27]
    for item in (first, second, third):
        actual, tensors = dataset.read(dataset.by_id[item["id"]])
        assert actual == item
        for name, expected in zip(("query", "candidates", "baseline"), values(item)):
            torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)


def test_weighted_gradients_ignore_shards_padding_and_microbatch_partition(tmp_path):
    cfg = config()
    old = [row("old-a", 2, group="old"), row("old-b", 2, group="old")]
    new = [row("new-a", 5, graded=True), row("new-b", 27, graded=True)]
    refs = load_rows([store(tmp_path / "new", cfg, new)], cfg, "fit", "new") + load_rows(
        [store(tmp_path / "old", cfg, old)], cfg, "fit", "replay")
    masses = row_masses(refs, ((2, 16), (17, 26), (27, 64)))
    assert sum(masses.values()) == pytest.approx(1)
    assert masses["old-a"] + masses["old-b"] == pytest.approx(.5)
    geometry = {name: cfg[name] for name in GEOMETRY}
    gradients, losses = [], []
    for budget in (10_000, 27 * 4):
        model = initialize(geometry, 9)
        total = 0.
        ordered = refs if budget == 10_000 else sorted(refs, key=lambda ref: ref["k"])
        for subset in batches(ordered, max_elements=budget, max_rows=4, hidden_size=4):
            metadata, batch = collate(subset)
            ce, bt = loss_components(utilities(model, batch, metadata, cfg), batch, metadata)
            weights = torch.tensor([len(refs) * masses[r["id"]] for r in subset])
            loss = ((ce + bt) * weights).sum() / 4
            total += float(loss.detach())
            loss.backward()
        losses.append(total)
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters()})
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=1e-6, rtol=2e-5)


def test_graded_loss_ties_permutation_and_ungraded_targets(tmp_path):
    item = row("graded", 4, graded=True)
    batch = {"mask": torch.ones(1, 4, dtype=torch.bool), "targets": torch.tensor([item["target"]])}
    scores = torch.tensor([[3., 2., 1., 0.]], requires_grad=True)
    ce, bt = loss_components(scores, batch, [item], pair_block=1)
    wrong_ce, wrong_bt = loss_components(-scores, batch, [item], pair_block=3)
    assert ce < wrong_ce and bt < wrong_bt
    fractional = dict(item, grades=[grade / 10 for grade in item["grades"]])
    torch.testing.assert_close(bt, loss_components(scores, batch, [fractional])[1])
    permutation = [2, 0, 3, 1]
    reordered = dict(item, target=[item["target"][i] for i in permutation],
                     grades=[item["grades"][i] for i in permutation])
    permuted_batch = dict(batch, targets=batch["targets"][:, permutation])
    permuted = loss_components(scores[:, permutation], permuted_batch, [reordered])
    torch.testing.assert_close(ce, permuted[0])
    torch.testing.assert_close(bt, permuted[1])
    no_grades = {k: v for k, v in item.items() if k != "grades"}
    soft = dict(batch, targets=torch.tensor([[.6, .2, .1, .1]]))
    ce_only, zero = loss_components(scores, soft, [no_grades])
    torch.testing.assert_close(ce_only, -(soft["targets"] * scores.log_softmax(-1)).sum(-1) / math.log(4))
    assert zero.item() == 0
    tied = dict(item, grades=[1.] * 4)
    assert loss_components(scores, batch, [tied])[1].item() == 0
    path = tmp_path / "scores.jsonl"
    records = []
    for view, order in enumerate((range(4), permutation)):
        records.append(dict(id=str(view), group="one", source="synthetic", lane="new", primitive="choice",
                            modality="text", target=[item["target"][i] for i in order],
                            candidate_ids=[f"c{i}" for i in order], grades=[item["grades"][i] for i in order],
                            acceptable_candidate_ids=["c0", "c1"],
                            head={"scores": [float(scores[0, i].detach()) for i in order]},
                            baseline={"scores": [0.] * 4}))
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    report = report_scores(path, ((2, 16),))
    assert report["order_agreement"] == 1
    assert report["cells"][0]["head"]["acceptable_accuracy"] == 1
    assert report["cells"][0]["head"]["ndcg"] == 1
    assert report["cells"][0]["head"]["accuracy"] is None


def test_streamed_cal_matches_full_objective_and_temperature(tmp_path):
    path = tmp_path / "cal.jsonl"
    rows = []
    for index, k in enumerate((2, 5, 3)):
        z = torch.linspace(-2, 2, k, dtype=torch.float64)
        rows.append(dict(id=str(index), group="a" if index < 2 else "b", primitive="choice", modality="text",
                         target=(z / 1.7).softmax(0).tolist(), head={"scores": z.tolist()}))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stratum = dict(primitive="choice", modality="text", k_min=2, k_max=5, temperature=1.)
    small, support = fit_temperature(path, stratum, batch_size=1)
    full, _ = fit_temperature(path, stratum, batch_size=32)
    assert small == pytest.approx(1.7, rel=1e-3)
    assert small == pytest.approx(full, rel=1e-7)
    assert support == {"records": 3, "groups": 2, "k_counts": {2: 1, 5: 1, 3: 1}, "bound": None}
    for item, temperature in zip(rows, (1.2, 2.4, .8)):
        item["target"] = (torch.tensor(item["head"]["scores"], dtype=torch.float64) / temperature).softmax(0).tolist()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    full_log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([full_log_t], max_iter=50, line_search_fn="strong_wolfe")

    def full_closure():
        optimizer.zero_grad()
        temperature = full_log_t.clamp(math.log(.03), math.log(100)).exp()
        objective = sum(-weight * (torch.tensor(item["target"], dtype=torch.float64)
                                  * (torch.tensor(item["head"]["scores"], dtype=torch.float64)
                                     / temperature).log_softmax(0)).sum() / math.log(len(item["target"]))
                        for weight, item in zip((.25, .25, .5), rows))
        objective.backward()
        return objective

    optimizer.step(full_closure)
    streamed, _ = fit_temperature(path, stratum, batch_size=1)
    assert streamed == pytest.approx(full_log_t.detach().exp().item(), rel=1e-6)


def test_bundle_geometry_initialization_and_primitive_specific_export(tmp_path):
    cfg = config(300)
    cfg["capacities"].update(ordinal={"text": {"implemented_max": 5, "validated_max": 5}},
                             noul={"text": {"implemented_max": 2, "validated_max": 2}})
    cfg["capacities"]["choice"]["image_text"] = {"implemented_max": 26, "validated_max": None}
    cfg["residual_strength"]["image_text"] = 0.
    original = source_bundle(tmp_path / "original", cfg)
    existing = load_model(original, cfg, "existing", 99)
    scratch = load_model(original, cfg, "scratch", 11)
    again = load_model(original, cfg, "scratch", 11)
    assert all(torch.equal(value, again.state_dict()[name]) for name, value in scratch.state_dict().items())
    assert any(not torch.equal(value, existing.state_dict()[name]) for name, value in scratch.state_dict().items())
    strata = [dict(primitive=p, modality=m, k_min=low, k_max=high, temperature=1.7)
              for p, m, low, high in (("choice", "text", 2, 26), ("choice", "text", 27, 300),
                                      ("choice", "image_text", 2, 26), ("ordinal", "text", 2, 5),
                                      ("noul", "text", 2, 2))]
    exported = export_bundle(tmp_path / "head", existing, cfg, strata, calibrated=True)
    metadata, _, _ = read_artifacts(exported)
    assert metadata["capacities"]["ordinal"]["text"]["implemented_max"] == 5
    assert metadata["capacities"]["noul"]["text"]["implemented_max"] == 2
    assert all(c["validated_max"] is None for supported in metadata["capacities"].values() for c in supported.values())
    with ServingHead(exported, device="cpu", calibration="calibrated") as head, ServingHead(original, device="cpu") as old:
        for k in (2, 27, 300):
            q, keys, baseline = values(row("example", k))
            logits, raw, calibrated = head.score("choice", q, keys, baseline=baseline)
            expected = existing(None, q[None], keys[None], baseline=baseline[None])[0]
            torch.testing.assert_close(torch.tensor(logits), expected)
            torch.testing.assert_close(torch.tensor(calibrated), (expected / 1.7).softmax(0))
            assert sum(raw) == pytest.approx(1)
            if k == 2:
                assert logits == old.score("choice", q, keys, baseline=baseline)[0]
                assert head.score("choice", q, keys, baseline=baseline, modality="image_text")[0] == baseline.tolist()


def test_synthetic_train_then_calibrate_commands(tmp_path, capsys):
    cfg = config(5)
    original = source_bundle(tmp_path / "original", cfg)
    cfg_path = tmp_path / "target.json"
    write_json(cfg_path, cfg)
    paths = {}
    for role in ("fit", "development", "calibration"):
        for lane in ("new", "replay"):
            name = role + "-" + lane
            paths[name] = store(tmp_path / name, cfg, [row(name, 3 if lane == "new" else 2, role=role,
                                                         graded=lane == "new")])
    output = tmp_path / "training"
    main(["train", "--bundle", str(original), "--config", str(cfg_path), "--output", str(output),
          "--fit", str(paths["fit-new"]), "--replay-fit", str(paths["fit-replay"]),
          "--development", str(paths["development-new"]), "--replay-development", str(paths["development-replay"]),
          "--initialization", "existing", "--epochs", "1", "--batch-size", "2", "--batch-elements", "12"])
    report = json.loads((output / "training.json").read_text())
    assert report["selected_epoch"] == 1
    assert report["epochs"][0]["fit_visits"] == 2
    assert sum(report["fit_row_masses"].values()) == pytest.approx(1)
    assert json.loads((output / "head/calibration.json").read_text()) == {"strata": []}
    before, after = read_artifacts(original)[1], read_artifacts(output / "head")[1]
    assert any(not torch.equal(before[key], after[key]) for key in before)
    strata = tmp_path / "strata.json"
    write_json(strata, {"strata": [dict(primitive="choice", modality="text", k_min=2, k_max=5, temperature=1.)]})
    calibrated = tmp_path / "calibrated"
    main(["calibrate", "--bundle", str(output / "head"), "--training-report", str(output / "training.json"),
          "--calibration", str(paths["calibration-new"]), "--replay-calibration", str(paths["calibration-replay"]),
          "--strata", str(strata), "--output", str(calibrated), "--batch-elements", "12", "--batch-size", "2"])
    support = json.loads((calibrated / "calibration-report.json").read_text())
    assert support["strata"][0]["k_counts"] == {"2": 1, "3": 1}
    assert support["evaluation"] == "not_run"
    assert all(torch.equal(value, read_artifacts(calibrated / "head")[1][key]) for key, value in after.items())
    assert "selected_epoch" in capsys.readouterr().out


def test_frozen_evaluate_modes_top_ids_and_observed_execution(tmp_path, capsys):
    cfg = config()
    model = initialize({name: cfg[name] for name in GEOMETRY}, 9)
    strata = [dict(primitive="choice", modality="text", k_min=low, k_max=high, temperature=t)
              for low, high, t in ((2, 26, 2.), (27, 27, 3.))]
    bundle = export_bundle(tmp_path / "head", model, cfg, strata, calibrated=True)
    first = row("ordered", 5, role="test", group="views", graded=True)
    first["execution"] = dict(submitted_tokens=100, forwarded_tokens=25, cached_tokens=75,
                              completed_root_branches=2, future_native_count=17)
    reordered = copy.deepcopy(first)
    reordered["id"] = "reordered"
    del reordered["execution"]
    order = [4, 1, 3, 0, 2]
    reordered["request"]["candidates"] = [first["request"]["candidates"][i] for i in order]
    for key in ("target", "grades"):
        reordered[key] = [first[key][i] for i in order]
    reordered["rendering"]["query_suffix"] = query_suffix(
        reordered["request"]["question"], reordered["request"]["candidates"])
    reordered["rendering"]["candidate_suffixes"] = [candidate_suffix(c) for c in reordered["request"]["candidates"]]
    large = row("label-free", 27, role="test", graded=True)
    large["execution"] = dict(forwarded_tokens=0, cached_tokens=300, completed_root_branches=0)
    binary = row("binary", 2, role="test")
    binary["target"] = [0., 1.]
    path = tmp_path / "features"
    writer = RaggedWriter(path, cfg, shard_bytes=1 << 20)
    q, keys, baseline = values(first)
    writer.add(first, q, keys, baseline)
    writer.add(reordered, q, keys[order], baseline[order])
    for item in (large, binary):
        writer.add(item, *values(item))
    writer.flush()
    originals = {p: p.read_bytes() for root in (bundle, path) for p in root.iterdir()}
    reports, scored = {}, {}
    for mode in ("raw", "calibrated"):
        output = tmp_path / mode
        main(["evaluate", "--bundle", str(bundle), "--test", str(path), "--calibration", mode,
              "--output", str(output), "--batch-size", "4", "--batch-elements", "108", "--top-k", "3",
              "--k-buckets", "2:2", "3:5", "27:27"])
        reports[mode] = json.loads((output / "evaluation.json").read_text())
        scored[mode] = [json.loads(line) for line in (output / "evaluation-scores.jsonl").read_text().splitlines()]
        assert sorted(p.name for p in output.iterdir()) == ["evaluation-scores.jsonl", "evaluation.json"]
        report = reports[mode]
        assert report["calibration"] == mode and report["records"] == 4 and report["groups"] == 3
        assert report["calibration_summary"] == dict(
            requested=mode, effective=mode, calibrated_records=4 if mode == "calibrated" else 0,
            raw_records=4 if mode == "raw" else 0)
        assert report["strata"] == (strata if mode == "calibrated" else [])
        assert report["weighted_normalized_nll"] is None
        assert report["order_agreement"] == 1
        assert report["execution"]["forwarded_tokens"] == dict(observed_sum=25, observed_records=2, missing_records=2)
        assert report["execution"]["cached_tokens"] == dict(observed_sum=375, observed_records=2, missing_records=2)
        assert report["execution"]["completed_root_branches"] == dict(observed_sum=2, observed_records=2, missing_records=2)
        assert report["execution"]["submitted_tokens"] == dict(observed_sum=100, observed_records=1, missing_records=3)
        assert scored[mode][0]["execution"] == first["execution"]
        assert scored[mode][0]["head"]["top_ids"] == scored[mode][1]["head"]["top_ids"]
        assert scored[mode][2]["baseline_kind"] == "zero"
        assert scored[mode][2]["baseline"]["top_ids"] == ["c0", "c1", "c2"]
        assert len(scored[mode][3]["head"]["top_ids"]) == 2
        cells = {cell["k_min"]: cell for cell in report["cells"]}
        assert cells[2]["execution"]["forwarded_tokens"] == dict(observed_sum=None, observed_records=0, missing_records=1)
        assert cells[3]["execution"]["completed_root_branches"]["observed_sum"] == 2
        assert cells[3]["head"]["accuracy"] is None
        for name in ("acceptable_accuracy", "ndcg", "strict_pair_agreement", "brier", "ece"):
            assert math.isfinite(cells[3]["head"][name])
        with ServingHead(bundle, device="cpu", calibration="calibrated") as serving:
            logits, raw, calibrated = serving.score("choice", q, keys, baseline=baseline)
        assert scored[mode][0]["head"]["scores"] == pytest.approx(logits, abs=1e-6)
        probabilities = raw if mode == "raw" else calibrated
        nll = -sum(y * math.log(p) for y, p in zip(first["target"], probabilities))
        brier = sum((p - y) ** 2 for p, y in zip(probabilities, first["target"]))
        assert cells[3]["head"]["nll"] == pytest.approx(nll, abs=1e-6)
        assert cells[3]["head"]["brier"] == pytest.approx(brier, abs=1e-6)
        expected_ids = [first["request"]["candidates"][i]["id"]
                        for i in sorted(range(5), key=lambda i: -logits[i])[:3]]
        assert scored[mode][0]["head"]["top_ids"] == expected_ids
        assert scored[mode][0]["head"]["selected_id"] == expected_ids[0]
    assert scored["raw"] == scored["calibrated"]
    for raw, calibrated in zip(reports["raw"]["cells"], reports["calibrated"]["cells"]):
        assert raw["baseline"] == calibrated["baseline"]
    assert originals == {p: p.read_bytes() for p in originals}
    assert "calibrated" in capsys.readouterr().out


@pytest.mark.parametrize("sparse", [False, True])
def test_evaluate_sparse_calibration_reports_effective_modes(tmp_path, sparse):
    cfg = config(5)
    cfg["capacities"]["ordinal"] = {"image_text": {"implemented_max": 5, "validated_max": None}}
    cfg["residual_strength"]["image_text"] = 0.
    model = initialize({name: cfg[name] for name in GEOMETRY}, 9)
    strata = [dict(primitive="choice", modality="text", k_min=k, k_max=k, temperature=t)
              for k, t in ((2, 1.), (4, 2.))] if sparse else []
    bundle = export_bundle(tmp_path / "head", model, cfg, strata, calibrated=True)
    items = [row(f"choice-{k}", k, role="test") for k in (2, 3, 4)]
    ordinal = row("image-ordinal", 5, role="test")
    ordinal.update(primitive="ordinal", modality="image_text", ordinal_values=list(range(5)))
    ordinal["request"].update(primitive="ordinal", image="synthetic.png")
    items.append(ordinal)
    for item in items:
        item["target"] = [0.] * (len(item["target"]) - 1) + [1.]
    path = store(tmp_path / "features", cfg, items)
    originals = {p: p.read_bytes() for root in (bundle, path) for p in root.iterdir()}
    output = tmp_path / "evaluation"
    main(["evaluate", "--bundle", str(bundle), "--test", str(path), "--calibration", "calibrated",
          "--output", str(output), "--batch-size", "4", "--batch-elements", "80",
          "--k-buckets", "2:2", "3:3", "4:4", "5:5"])
    report = json.loads((output / "evaluation.json").read_text())
    assert report["calibration"] == "calibrated"
    assert report["calibration_summary"] == dict(
        requested="calibrated", effective="mixed" if sparse else "raw",
        calibrated_records=2 if sparse else 0, raw_records=2 if sparse else 4)
    scored = [json.loads(line) for line in (output / "evaluation-scores.jsonl").read_text().splitlines()]
    assert scored[-1]["head"]["scores"] == scored[-1]["baseline"]["scores"]
    with ServingHead(bundle, device="cpu", calibration="calibrated") as head:
        for item in items:
            k = len(item["target"])
            q, keys, baseline = values(item)
            _, raw, probabilities = head.score(item["primitive"], q, keys, baseline=baseline,
                                                modality=item["modality"])
            temperature = head.temperature_for(item["primitive"], modality=item["modality"], count=k)
            cell = next(cell for cell in report["cells"] if cell["k_min"] == k)
            assert cell["calibration_summary"] == dict(
                requested="calibrated", effective="raw" if temperature is None else "calibrated",
                calibrated_records=int(temperature is not None), raw_records=int(temperature is None))
            assert cell["head"]["nll"] == pytest.approx(-math.log(probabilities[-1]), abs=1e-6)
            assert cell["head"]["brier"] == pytest.approx(
                sum((p - y) ** 2 for p, y in zip(probabilities, item["target"])), abs=1e-6)
            if k == 2:
                assert probabilities == raw
                assert temperature == (1. if sparse else None)
    assert originals == {p: p.read_bytes() for p in originals}



def test_scratch_train_without_bundle_or_replay_uses_config(tmp_path):
    cfg = config(5)
    cfg_path = tmp_path / "target.json"
    write_json(cfg_path, cfg)
    fit = store(tmp_path / "fit", cfg, [row("fit", 3, graded=True)])
    dev = store(tmp_path / "development", cfg, [row("dev", 3, role="development", graded=True)])
    output = tmp_path / "scratch"
    main(["train", "--config", str(cfg_path), "--output", str(output),
          "--fit", str(fit), "--development", str(dev), "--initialization", "scratch",
          "--batch-elements", "12", "--epochs", "1"])
    report = json.loads((output / "training.json").read_text())
    assert report["replay_mass"] == 0
    assert report["fit_row_masses"] == {"fit": 1.}
    metadata, state, _ = read_artifacts(output / "head")
    assert {name: metadata[name] for name in GEOMETRY} == {name: cfg[name] for name in GEOMETRY}
    assert all(value.isfinite().all() for value in state.values())

def test_legacy_import_preserves_original_roles_soft_targets_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("FUZEV_ROOT", str(tmp_path))
    from fuzev_research.features import FeatureStore

    cfg = config(5, hidden=2560)
    identity = {"hidden_size": 2560, "encoding": {"version": VERSION, "thinking": False,
                "readout": "last consumed token, post-collapse, FP32 owned copy"},
                "feature_tap": "final-mixer/post-collapse/before-lm-head"}
    source = FeatureStore(tmp_path / "legacy", identity)
    rows = [row("old-" + role, 2, role=role) for role in ("fit", "development", "calibration")]
    for item in rows:
        item["candidates"] = item["request"]["candidates"]
        item["target"] = [.7, .3]
        item["rendering"]["encoding_version"] = VERSION
        source.add(item, *values(item, 2560))
    source.close()
    writer = RaggedWriter(tmp_path / "ragged", cfg, shard_bytes=1 << 20)
    import_legacy(tmp_path / "legacy", rows, writer)
    dataset = RaggedFeatures(tmp_path / "ragged", cfg)
    assert [r["role"] for r in dataset.rows] == ["fit", "development", "calibration"]
    for ref in dataset.rows:
        metadata, tensors = dataset.read(ref)
        assert metadata["target"] == [.7, .3]
        torch.testing.assert_close(tensors["baseline"], values(metadata, 2560)[2])
    with pytest.raises(ValueError, match="Legacy ownership"):
        import_legacy(tmp_path / "legacy", [dict(rows[0], role="development")], writer)


def test_replay_cli_joins_original_inputs_bounds_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FUZEV_ROOT", str(tmp_path))
    from fuzev_research.features import FeatureStore

    legacy = tmp_path / "legacy"
    identity = {"hidden_size": 2560, "encoding": {"version": VERSION, "thinking": False,
                "readout": "last consumed token, post-collapse, FP32 owned copy"},
                "feature_tap": "final-mixer/post-collapse/before-lm-head", "cache_precision": "float16",
                "engine": "original-producer", "model": "original-model", "revision": "original-revision"}
    source = FeatureStore(legacy, identity)
    prepared = []
    for role in ("fit", "development", "calibration", "test"):
        for index in range(2):
            k = 5 if index else 2
            item = row(f"{role}-{index}", k, role=role)
            item["context"] = item["request"]["messages"] if index else []
            item["question"] = item["request"]["question"]
            item["candidates"] = item["request"]["candidates"]
            item["target"] = [.7, .3] + [0.] * (k - 2)
            item["primitive"] = "ordinal" if index else "noul"
            if index:
                for level, candidate in enumerate(item["candidates"]):
                    candidate["ordinal_value"] = level
            else:
                item["candidates"] = [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
            del item["request"], item["rendering"]
            source.add(item, *values(item, 2560))
            prepared.append(item)
    source.close()
    original_bytes = {p.name: p.read_bytes() for p in legacy.iterdir()}
    preparation = tmp_path / "prepared.jsonl"
    preparation.write_text("".join(json.dumps(item) + "\n" for item in prepared))
    bundle = tmp_path / "bundle.json"
    write_json(bundle, {key: config(hidden=2560)[key] for key in GEOMETRY})
    provenance = tmp_path / "published.json"
    origins = {"current_runtime_sources": {"recipes": "current-record"},
               "reused": {"runtime_sources": {"recipes": "original-record"}, "records": 8}}
    write_json(provenance, {"feature_origins": origins})
    output = tmp_path / "replay"
    args = ["--legacy", str(legacy), "--prepared", str(preparation), "--bundle-config", str(bundle),
            "--provenance", str(provenance), "--output", str(output), "--shard-bytes", str(1 << 20)]
    replay_main(args + ["--max-new-rows", "1"])
    report = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(r["records"] == r["added"] == 1 and r["limited"] and not r["complete"] for r in report)
    saved = {p: p.read_bytes() for p in output.glob("*/*")}
    replay_main(args)
    report = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(r["records"] == 2 and r["resumed"] == r["added"] == 1 and r["complete"] for r in report)
    assert all(p.read_bytes() == data for p, data in saved.items())
    cfg, counts = replay_config(bundle, legacy)
    assert counts == dict.fromkeys(("fit", "development", "calibration"), 2)
    assert cfg["capacities"] == {p: {"text": {"implemented_max": k, "validated_max": None}}
                                 for p, k in (("noul", 2), ("ordinal", 5))}
    for role in counts:
        dataset = RaggedFeatures(output / role, cfg)
        for ref in dataset.rows:
            metadata, tensors = dataset.read(ref)
            assert metadata["role"] == role
            assert metadata["rendering"]["encoding_version"] == VERSION
            assert metadata["request"]["additional_context"] == ""
            content = "A synthetic context." if ref["k"] == 5 else "Context for the decision."
            assert metadata["rendering"]["prefix"] == f"<|im_start|>user\n{content}<|im_end|>\n"
            assert metadata["producer"]["cache_precision"] == "float16"
            assert metadata["producer"]["published_cache_lineage"]["reused"] == origins["reused"]
            if ref["k"] == 5:
                assert metadata["ordinal_values"] == list(range(5))
            assert metadata["target"] == [.7, .3] + [0.] * (ref["k"] - 2)
            for name, expected in zip(("query", "candidates", "baseline"), values(metadata, 2560)):
                torch.testing.assert_close(tensors[name], expected, rtol=0, atol=0)
    assert not (output / "test").exists()
    assert {p.name: p.read_bytes() for p in legacy.iterdir()} == original_bytes
    # Completed replay can resume without reopening source feature tensors.
    shard = legacy / "shard-000000.safetensors"
    held = shard.with_suffix(".held")
    shard.rename(held)
    try:
        replay_main(args)
    finally:
        held.rename(shard)
    assert all(json.loads(line)["added"] == 0 for line in capsys.readouterr().out.splitlines())
    prepared[0]["context"] = [{"role": "user", "content": "Changed model input"}]
    preparation.write_text("".join(json.dumps(item) + "\n" for item in prepared))
    with pytest.raises(ValueError, match="changed metadata/request/rendering"):
        replay_main(args)
    preparation.write_text(json.dumps(prepared[1]) + "\n")
    with pytest.raises(ValueError, match="not fully joined"):
        replay_main([arg if arg != str(output) else str(tmp_path / "incomplete") for arg in args])
