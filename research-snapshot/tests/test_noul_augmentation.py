import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from fuzev_research import noul_augmentation as augmentation


SMALL_BUDGETS = {"fit": 24, "development": 12, "calibration": 12}
DIMENSIONS = {"min_axis_pixels": 32, "max_aspect_ratio": 200, "boundary": "inclusive",
              "resized_images": False, "raw_images_deleted": False}


def parent(source="helpsteer2", role="fit", number=0, target=None, group=None):
    image = source == "mm-rlhf"
    result = {"id": f"{source}-{role}-{number}", "source": source, "revision": f"pinned-{source}",
              "group": group or f"{'image' if image else 'prompt'}:{source}-{role}-{number}",
              "role": role, "original_split": "train", "modality": "image" if image else "text",
              "context": [] if image else [{"role": "user", "content": f"Original request {number}"}],
              "question": augmentation.MM_QUESTION_PREFIX + f"What is in image {number}?" if image
                          else "Which response better answers the user's request?",
              "candidates": [{"id": "0" if image else "a", "text": f"First\nresponse {number}"},
                             {"id": "1" if image else "b", "text": f"Second response {number}"}],
              "primitive": "choice", "target": target or [1., 0.],
              "raw_annotation": {"label": "ANNOTATION_NOT_AN_INPUT"}}
    if image:
        result.update(image_path=f"/unchanged/original/image-{number}.png", image_sha256="a" * 64,
                      source_image_id=f"raw-image-{number}", annotation_id=f"raw-annotation-{number}",
                      adaptation="pairwise_preference", official_holdout_overlap=False,
                      dataset_scope_revision="2026-10-02-v2", dataset_purpose="training",
                      evaluation_scope="training")
    else:
        result["original_preference"] = "ANNOTATION_NOT_AN_INPUT"
    return result


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def write_parents(root, rows):
    prepared, manifests = root / "data/prepared", root / "data/manifests"
    prepared.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    for modality, stem, manifest_name in (("text", "text", "text-splits.json"),
                                         ("image", "images", "image-splits.json")):
        selected = [row for row in rows if row["modality"] == modality]
        contents = b"".join(json_bytes(row) + b"\n" for row in selected)
        path = prepared / (stem + ".jsonl")
        path.write_bytes(contents)
        counts = Counter(f"{row['source']}/{row['primitive']}/{row['role']}" for row in selected)
        value = {"records": len(selected), "counts": dict(counts), "sha256": hashlib.sha256(contents).hexdigest(),
                 "path": str(path.resolve())}
        if modality == "image":
            value["dimension_eligibility"] = DIMENSIONS
        (manifests / manifest_name).write_bytes(json_bytes(value))


def corpus(budgets=SMALL_BUDGETS):
    rows = []
    for role, size in budgets.items():
        for source in augmentation.SOURCES:
            quota = size // (2 if source == "mm-rlhf" else 4)
            for number in range(quota + 9):
                target = ([1., 0.], [0., 1.], [.5, .5])[number % 3]
                rows.append(parent(source, role, number, target=target,
                                   group=f"{source}:{role}:group-{number // 2}"))
    for source in augmentation.SOURCES:
        rows.append(parent(source, "test", 999))
    ordinal = parent(number=998)
    ordinal.update(id="ordinal", primitive="ordinal", target=[0., 0., 1., 0., 0.])
    rows.append(ordinal)
    best = parent("mm-rlhf", number=998)
    best.update(id="best-response", adaptation="best_response")
    rows.append(best)
    return rows


def output_paths(root):
    return [path for filename in augmentation.OUTPUTS.values()
            for path in (root / "data/prepared" / filename,
                         root / "data/manifests" / (Path(filename).stem + "-splits.json"))]


def read_views(root):
    return [json.loads(line) for filename in augmentation.OUTPUTS.values()
            for line in (root / "data/prepared" / filename).read_bytes().splitlines()]


@pytest.mark.parametrize("source", augmentation.SOURCES)
@pytest.mark.parametrize("target", [[1., 0.], [0., 1.], [.5, .5]])
def test_view_preserves_parent_and_exact_pair_with_fixed_yes_no(source, target):
    original = parent(source, target=target)
    saved = copy.deepcopy(original)
    view = augmentation.augment_parent(original)
    assert original == saved
    permutation = view["candidate_permutation"]
    assert permutation in ([0, 1], [1, 0])
    assert view["target"] == [original["target"][index] for index in permutation]
    assert view["candidates"] == [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
    for key in ("role", "source", "revision", "group", "original_split", "context", "modality", "raw_annotation"):
        assert view[key] == original[key]
    assert view["parent_id"] == original["id"] and view["id"] != original["id"]
    assert view["parent_role"] == original["role"] and view["parent_source"] == source
    assert view["parent_primitive"] == "choice"
    assert view["parent_candidate_ids"] == [c["id"] for c in original["candidates"]]
    assert view["parent_target"] == target
    assert view["parent_candidates"] == original["candidates"]
    a, b = [original["candidates"][index]["text"] for index in permutation]
    assert f"Response A: {a}\nResponse B: {b}\n" in view["question"]
    assert "ANNOTATION_NOT_AN_INPUT" not in view["question"]
    assert view["dataset_scope_revision"] == "2026-10-05-v3-noul-preference"
    assert view["dataset_purpose"] == view["evaluation_scope"] == "training"
    assert view["adaptation"] == "pairwise_preference_yes_no"
    assert view["adapter_version"] == "noul-relative-preference-v1"
    assert view["label_semantics"] == "relative_pairwise_preference"
    assert view["orientation_seed"] == 20261005
    if source == "mm-rlhf":
        for key in ("image_path", "image_sha256", "source_image_id", "official_holdout_overlap", "annotation_id"):
            assert view[key] == original[key]
        assert view["question"].endswith("Is response A preferred to response B for this question about the image?")
    else:
        assert view["question"].endswith("Is response A preferred to response B for this request?")
    view["context"].append({"role": "assistant", "content": "Mutate only the child"})
    view["parent_candidates"][0]["text"] = "Changed child metadata"
    assert original == saved


def test_orientation_does_not_hash_labels_or_label_dependent_parent_id():
    original = parent()
    changed = {**original, "id": "different-label-dependent-id", "target": [0., 1.],
               "original_preference": 99, "raw_annotation": {"label": "different"}}
    a, b = augmentation.augment_parent(original), augmentation.augment_parent(changed)
    assert a["candidate_permutation"] == b["candidate_permutation"]
    assert a["question"] == b["question"] and a["parent_input_sha256"] == b["parent_input_sha256"]
    assert a["target"] == b["target"][::-1]
    assert {tuple(augmentation.augment_parent(parent(number=i))["candidate_permutation"])
            for i in range(30)} == {(0, 1), (1, 0)}


@pytest.mark.parametrize("fields", [
    {"role": "test"}, {"source": "qasper"}, {"source": "vl-rewardbench"}, {"source": "aokvqa"},
    {"evaluation_scope": "external"}, {"dataset_purpose": "external"},
    {"original_split": "validation"}, {"primitive": "noul"}, {"primitive": "ordinal"},
    {"role": "cal"}, {"group": ""}, {"revision": None}, {"modality": "image"},
    {"target": [True, False]}, {"target": [.6, .4]}, {"target": [float("inf"), 0.]},
    {"target": [float("nan"), 0.]}, {"target": [.5]}, {"target": [2., -1.]},
    {"candidates": [{"id": "a", "text": "Only one"}]},
    {"candidates": [{"id": "a", "text": 1}, {"id": "b", "text": "B"}]},
    {"context": [{"role": "gold", "content": "annotation"}]}, {"question": "Is this correct?"},
])
def test_forbidden_or_malformed_parent_is_denied(fields):
    with pytest.raises(ValueError):
        augmentation.augment_parent({**parent(), **fields})


@pytest.mark.parametrize("fields", [{"adaptation": "best_response"}, {"official_holdout_overlap": True},
                                     {"official_holdout_overlap": None}, {"image_path": ""},
                                     {"image_sha256": "bad"}, {"source_image_id": None},
                                     {"annotation_id": ""}, {"question": "Unknown image prompt"}])
def test_unadmitted_or_nonpairwise_image_parent_is_denied(fields):
    with pytest.raises(ValueError):
        augmentation.augment_parent({**parent("mm-rlhf"), **fields})


def test_official_empty_response_bytes_are_not_changed():
    original = parent("helpsteer3")
    original["candidates"][0]["text"] = ""
    view = augmentation.augment_parent(original)
    assert view["parent_candidates"][0]["text"] == ""
    assert any(f"Response {label}: \n" in view["question"] for label in ("A", "B"))


def test_integer_raw_annotation_id_is_preserved():
    original = parent("mm-rlhf")
    original["annotation_id"] = 0
    assert augmentation.augment_parent(original)["annotation_id"] == 0


def test_prepare_is_exact_stratified_deterministic_and_leakage_free(tmp_path):
    rows = corpus()
    write_parents(tmp_path, rows)
    parent_paths = [tmp_path / "data/prepared/text.jsonl", tmp_path / "data/prepared/images.jsonl",
                    tmp_path / "data/manifests/text-splits.json", tmp_path / "data/manifests/image-splits.json"]
    saved = {path: path.read_bytes() for path in parent_paths}
    result = augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    contents = {path: path.read_bytes() for path in output_paths(tmp_path)}
    views = read_views(tmp_path)
    assert len(views) == sum(SMALL_BUDGETS.values())
    assert len({view["id"] for view in views}) == len(views)
    by_id = {row["id"]: row for row in rows}
    counts = Counter((v["role"], v["source"]) for v in views)
    for role, size in SMALL_BUDGETS.items():
        for source in augmentation.SOURCES:
            assert counts[role, source] == size // (2 if source == "mm-rlhf" else 4)
    owners = {}
    for view in views:
        original = by_id[view["parent_id"]]
        assert view["role"] != "test" and view["parent_role"] == view["role"] == original["role"]
        assert view["group"] == original["group"]
        assert owners.setdefault(view["group"], view["role"]) == view["role"]
        assert original["primitive"] == "choice" and original.get("adaptation") != "best_response"
        assert view["parent_line_sha256"] == hashlib.sha256(json_bytes(original) + b"\n").hexdigest()
    for filename, manifest in result.items():
        path = tmp_path / "data/prepared" / filename
        assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["records"] == sum(manifest["counts"].values())
        assert manifest["dataset_scope_revision"] == augmentation.REVISION
        assert manifest["seed"] == 20261005 and manifest["purpose"] == "training"
        assert manifest["parent_preparations"][0]["sha256"] == hashlib.sha256(
            Path(manifest["parent_preparations"][0]["path"]).read_bytes()).hexdigest()
    assert any(view["target"] == [.5, .5] for view in views)
    assert {path: path.read_bytes() for path in parent_paths} == saved
    for path in output_paths(tmp_path):
        path.unlink()
    assert augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS) == result
    assert {path: path.read_bytes() for path in output_paths(tmp_path)} == contents
    # Input order may change provenance hashes, never selected identities or orientation.
    for path in output_paths(tmp_path):
        path.unlink()
    write_parents(tmp_path, list(reversed(rows)))
    augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    assert [(v["id"], v["target"], v["question"]) for v in read_views(tmp_path)] == [
        (v["id"], v["target"], v["question"]) for v in views]


@pytest.mark.parametrize("problem", ["leakage", "external", "aok", "hash", "counts", "dimensions", "insufficient", "duplicates"])
def test_preparation_fails_closed_without_outputs(tmp_path, problem):
    rows = corpus()
    if problem == "leakage":
        rows[0]["group"] = next(row["group"] for row in rows if row["role"] == "calibration")
    elif problem == "external":
        rows[0]["evaluation_scope"] = "external"
    elif problem == "aok":
        rows[0]["source"] = "aokvqa"
    elif problem == "insufficient":
        rows = [row for row in rows if row["source"] != "helpsteer2"]
    elif problem == "duplicates":
        rows.append(copy.deepcopy(rows[0]))
    write_parents(tmp_path, rows)
    if problem in ("hash", "counts", "dimensions"):
        path = tmp_path / "data/manifests" / ("image-splits.json" if problem == "dimensions" else "text-splits.json")
        value = json.loads(path.read_bytes())
        if problem == "hash":
            value["sha256"] = "0" * 64
        elif problem == "counts":
            value["records"] += 1
        else:
            value["dimension_eligibility"]["resized_images"] = True
        path.write_bytes(json_bytes(value))
    with pytest.raises(ValueError):
        augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    assert not any(path.exists() for path in output_paths(tmp_path))


def test_existing_outputs_are_never_overwritten(tmp_path):
    path = tmp_path / "data/prepared/noul-preference-text.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"Existing output must survive")
    with pytest.raises(FileExistsError):
        augmentation.prepare(tmp_path)
    assert path.read_bytes() == b"Existing output must survive"


def test_publication_race_rolls_back_only_owned_outputs(tmp_path, monkeypatch):
    write_parents(tmp_path, corpus())
    link = augmentation.os.link
    collision = tmp_path / "data/prepared/noul-preference-images.jsonl"

    def conflict(origin, destination):
        if destination == collision:
            collision.write_bytes(b"Another producer owns this output")
        link(origin, destination)

    monkeypatch.setattr(augmentation.os, "link", conflict)
    with pytest.raises(FileExistsError):
        augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    assert collision.read_bytes() == b"Another producer owns this output"
    assert not any(path.exists() for path in output_paths(tmp_path) if path != collision)


def test_semantic_duplicate_views_are_not_counted_twice(tmp_path):
    rows = corpus()
    duplicate = copy.deepcopy(rows[0])
    duplicate.update(id="a-second-parent-id", raw_annotation={"label": "a different provenance"})
    rows.append(duplicate)
    write_parents(tmp_path, rows)
    augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    selected = {view["parent_id"] for view in read_views(tmp_path)}
    assert not {rows[0]["id"], duplicate["id"]} <= selected


def test_group_round_robin_does_not_spend_quota_on_a_single_group():
    candidates = [augmentation._Candidate("helpsteer2", "fit", group, "yes", f"{group}-{i}",
                                         f"input-{group}-{i}", Path("unused"), 0, 1, "a" * 64)
                  for group in ("first", "second", "third") for i in range(5)]
    for role in ("development", "calibration"):
        for source in augmentation.SOURCES:
            candidates.extend(augmentation._Candidate(source, role, "group", "yes", f"{role}-{source}-{i}",
                f"input-{i}", Path("unused"), 0, 1, "a" * 64) for i in range(8))
    for source in ("helpsteer3", "mm-rlhf"):
        candidates.extend(augmentation._Candidate(source, "fit", f"group-{i}", "yes", f"{source}-{i}",
            f"input-{i}", Path("unused"), 0, 1, "a" * 64) for i in range(8))
    selected = augmentation._select(candidates, {"fit": 12, "development": 4, "calibration": 4}, augmentation.SEED)
    assert len({c.group for c in selected if c.role == "fit" and c.source == "helpsteer2"}) == 3


@pytest.mark.parametrize("counts,size", [({"yes": 100, "no": 100, "tie": 1}, 12),
                                       ({"yes": 10, "no": 0, "tie": 0}, 4),
                                       ({"yes": 2, "no": 2, "tie": 2}, 6)])
def test_stratum_allocation_keeps_available_ties_and_exact_budget(counts, size):
    quotas = augmentation._stratum_quotas(size, counts)
    assert sum(quotas.values()) == size
    assert all(0 <= quotas[label] <= counts[label] for label in augmentation.STRATA)
    assert all(quotas[label] >= 1 for label in augmentation.STRATA if counts[label])


def test_changed_parent_file_fails_before_publication(tmp_path, monkeypatch):
    write_parents(tmp_path, corpus())
    select = augmentation._select

    def changed(*args):
        selected = select(*args)
        path = tmp_path / "data/prepared/text.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        return selected

    monkeypatch.setattr(augmentation, "_select", changed)
    with pytest.raises(ValueError, match="Canonical parents changed"):
        augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    assert not any(path.exists() for path in output_paths(tmp_path))


def test_parent_symlink_cannot_redirect_to_another_dataset_directory(tmp_path):
    write_parents(tmp_path, corpus())
    path = tmp_path / "data/prepared/text.jsonl"
    other = tmp_path / "outside.jsonl"
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="escapes canonical"):
        augmentation.prepare(tmp_path, role_budgets=SMALL_BUDGETS)
    assert not any(path.exists() for path in output_paths(tmp_path))


def test_full_approved_5120_budget_and_cli_are_cpu_only(tmp_path):
    write_parents(tmp_path, corpus(augmentation.ROLE_BUDGETS))
    cwd = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "fuzev_research.noul_augmentation", "--root", str(tmp_path)]
    env = {**os.environ, "PYTHONPATH": str(cwd), "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    manifests = json.loads(completed.stdout)
    assert sum(value["records"] for value in manifests.values()) == 5120
    assert all(value["records"] == 2560 and value["budgets"] == augmentation.BUDGETS for value in manifests.values())
    views = read_views(tmp_path)
    counts = Counter((view["role"], view["source"]) for view in views)
    for role, shares in augmentation.BUDGETS.items():
        for source, count in shares.items():
            assert counts[role, source] == count
    assert not any(view["role"] == "test" for view in views)
    imports = subprocess.run([sys.executable, "-c", "import sys; import fuzev_research.noul_augmentation; "
        "assert not {'torch','transformers','PIL','fuzev_research.native','fuzev_research.train','fuzev_research.scope'} & sys.modules.keys()"],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=10)
    assert imports.returncode == 0, imports.stderr
