"""Offline, research-only comparison against the fixed native4 RAW T1 run.

Run from the unpacked research-snapshot directory.
Only saved E2E artifacts and the small panel manifest are read. No inference,
dataset payloads, checkpoints, calibration, retries, or qualification occur.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

from . import evaluation_metrics, metrics


ROOT = Path(os.environ["FUZEV_ROOT"]).expanduser().resolve()
REFERENCE_DIRECTORY = ROOT / "reports/2026-10-10-kev-text-evaluation/e2e-native-v2"
MANIFEST = ROOT / "data/evaluation/e2e-with-kev-v1/manifest.json"
MANIFEST_SHA = "452c065912437fddc54f333bc73d17d6d1c560ccb491dfa827adfb43494e786e"
PLAN_SHA = "252f43ac3d7eab82c3d3be8db468da17798f4766e6b083e505da72c98bffcf3e"
QUALITY_SHA = "83464ee250883e186876195c4bd5421d27bce07a1968cd13a39b5243da34eb9b"
HEAD_SHA = "f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c"
REFERENCE_PINS = {
    "binding.json": "da9409ec71c50d438fe40384151508bd666511a7d931e8884d4e0e0569d45f38",
    "summary.json": "df071f4b686189f09ce1aa82dfa4379a6e4db0c180cf960407132ef5ae375a37",
    "health-provenance.json": "31d7643e30801e763b543078a96d8241f57f4fce1afe25152b2d65ef32241fa5",
}
FLAGS = {"experimental": True, "calibrated": False, "qualified": False, "servable": False}
PANEL_COUNTS = {"original-fixed": 2000, "kev-development": 3075}
RECORD_COUNTS = {"original-fixed": 2000, "kev-development": 1990}
TOTAL = 5075
MODELS = {"native-3.05", "native-4.05", "native-5.05", "vllm-nvidia",
          "vllm-radixark", "vllm-exl3-4.05"}
WEIGHTING = "Unweighted descriptive fixed-panel means; not population estimates or a quality gate"
WEIGHTING_SCOPE = ("Conditional API-eligible represented strata; subset inverse inclusion multiplied by "
                   "original external sampling weight. No whole-population or causal claim.")
FILES = ("binding.json", "summary.json", "health-provenance.json", "http.jsonl",
         "kev-development-rows.json", "failure.json")


class ContractError(ValueError):
    """Static error codes only: never include examples or response bodies."""


def _require(value, code):
    if not value:
        raise ContractError(code)


def _encoded(value):
    # Match the clients' request-ID bytes, including their default JSON spacing.
    return (json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _flags(value):
    _require(all(value.get(key) is expected for key, expected in FLAGS.items()), "research_flags")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _manifest():
    value = _read(MANIFEST)
    _require(value["combined_records"] == 3990 and value["combined_questions_or_requests"] == TOTAL,
             "manifest_population")
    old, kev = value["panels"]["original-fixed"], value["panels"]["kev-development"]
    _require(old["records"] == old["requests"] == 2000
             and kev["records"] == 1990 and kev["questions"] == 3075, "manifest_panels")
    _require(value["external_evaluation_only"] is True and value["new_data_fitting"] is False
             and all(value[key] is False for key in
                     ("training_allowed", "calibration_allowed", "model_selection_allowed")),
             "manifest_roles")
    # Check declarations against run receipts; do not open these dataset paths.
    return value, {old["path"]: old["sha256"], kev["source"]: kev["source_sha256"],
                   kev["requests"]: kev["requests_sha256"], kev["expected"]: kev["expected_sha256"]}


def _adapter(manifest, path=None):
    path = Path(path) if path is not None else Path.cwd() / "scripts/kev_requests_adapter.py"
    spec = importlib.util.spec_from_file_location("_kev_cross_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(directory):
    return {name: _sha(directory / name) if (directory / name).is_file() else None for name in FILES}


@dataclass
class Decision:
    request_id: str
    index: int
    panel: str
    track: str
    modality: str
    primitive: str
    group: str
    record_id: str
    input_digest: str
    options: tuple
    target: np.ndarray
    logits: np.ndarray
    p: np.ndarray
    selected_id: str
    ordinal: object
    kev: object
    sampling: object = None


@dataclass
class Run:
    directory: Path
    model: object = None
    decisions: dict = field(default_factory=dict)
    issues: Counter = field(default_factory=Counter)
    observed: Counter = field(default_factory=Counter)
    identities: dict = field(default_factory=dict)
    health: list = field(default_factory=list)
    provenance: object = None
    hashes: dict = field(default_factory=dict)

    def coverage(self):
        panels = {}
        for panel, expected in PANEL_COUNTS.items():
            observed = sum(p == panel for p in self.identities.values())
            successful = sum(row.panel == panel for row in self.decisions.values())
            panels[panel] = {"expected": expected, "observed_unique": observed,
                             "successful": successful, "failed_or_invalid": observed - successful,
                             "missing": max(0, expected - observed)}
        return {"expected_requests": TOTAL, "observed_decision_entries": sum(self.observed.values()),
                "observed_unique_requests": len(self.identities), "successful": len(self.decisions),
                "missing": max(0, TOTAL - len(self.identities)), "panels": panels,
                "health_entries": len(self.health), "issue_counts": dict(sorted(self.issues.items()))}


def _binding(value, panel_pins, reference):
    _flags(value)
    _require(all(value.get(key) == expected for key, expected in
                 {"streams": 1, "retries": 0, "requests": TOTAL, "records": 3990}.items()),
             "binding_contract")
    pins = value.get("input_sha256", {})
    _require(all(pins.get(path) == expected for path, expected in panel_pins.items()), "panel_input_pins")
    if reference:
        _require(value.get("model", "native-4.05") == "native-4.05", "reference_model")
        return "native-4.05"
    _require(value.get("model") in MODELS and value.get("canonical_request_model") == "native-4.05",
             "candidate_model")
    return value["model"]


def _provenance(value):
    _flags(value)
    _require(all(value.get(key) == expected for key, expected in
                 {"head_execution_device": "cuda:0", "learned_head_count": 1,
                  "head_lane": "lm-assisted", "calibration_id": "uncalibrated:T1"}.items())
             and _number(value.get("raw_inference_temperature"))
             and value["raw_inference_temperature"] == 1., "head_provenance")
    _require(isinstance(value.get("source_pins"), dict) and value["source_pins"], "source_receipt")
    _digest(value)


def _original_sampling(item):
    subset = item["subset_inclusion_probability"]
    stratum = item["label_stratum"]
    _require(_number(subset) and 0 < subset <= 1
             and isinstance(stratum, str) and stratum, "subset_sampling_metadata")
    external = item.get("original_external_sampling")
    external_probability, external_weight = 1., 1.
    if external is not None:
        external_probability = external["inclusion_probability"]
        external_weight = external["sampling_weight"]
        _require(_number(external_probability) and 0 < external_probability <= 1
                 and _number(external_weight) and external_weight > 0
                 and math.isclose(external_probability * external_weight, 1., rel_tol=1e-9, abs_tol=1e-12),
                 "external_sampling_metadata")
    weight = (1. / subset) * external_weight
    _require(_number(weight) and weight > 0, "combined_sampling_weight")
    return {"subset_probability": subset, "external_probability": external_probability,
            "external_weight": external_weight, "combined_weight": weight, "label_stratum": stratum}


def _decision(entry, model, provenance, adapter):
    request, gold = entry["request"], entry["gold"]
    panel, index, request_id = entry["panel"], entry["index"], entry["request_id"]
    _require(panel in PANEL_COUNTS and type(index) is int and 0 <= index < TOTAL, "decision_identity")
    _require((index < 2000) == (panel == "original-fixed"), "panel_index")
    expected_id = "kev_" + hashlib.sha256(_encoded([MANIFEST_SHA, panel, index, gold["id"]])).hexdigest()
    _require(request_id == expected_id, "stable_request_id")
    _require(request["model"] == model and request["head_id"] == "reference"
             and request["primitive"] == gold["primitive"]
             and request["primitive"] in ("choice", "noul", "ordinal")
             and gold["role"] == "test" and request["candidates"] == gold["candidates"], "request_contract")
    _require(entry.get("status") == 200 and "error" not in entry, "http_or_client_failure")
    candidates = request["candidates"]
    ids = tuple(candidate["id"] for candidate in candidates)
    _require(len(ids) >= 2 and len(set(ids)) == len(ids)
             and all(isinstance(key, str) and key for key in ids), "option_ids")
    target = gold["target"]
    _require(len(target) == len(ids) and all(_number(t) and t >= 0 for t in target)
             and math.isclose(sum(target), 1., rel_tol=0, abs_tol=1e-7), "target_distribution")
    response = json.loads(entry["response_body"])
    _flags(response)
    _require(response.get("id") == request_id and all(response.get(key) == request[key]
             for key in ("model", "head_id", "primitive")), "response_identity")
    _require(all(response.get(key) == expected for key, expected in
                 {"calibration_id": "uncalibrated:T1"}.items()),
             "response_head")
    if model.startswith("vllm-"):
        _require(response.get("provenance") == provenance, "response_provenance")
    scores = response["scores"]
    _require(len(scores) == len(ids) and {score["id"] for score in scores} == set(ids), "score_ids")
    by_id = {score["id"]: score for score in scores}
    logits = [by_id[key]["logit"] for key in ids]
    p = [by_id[key]["raw_probability"] for key in ids]
    _require(all(_number(z) for z in logits) and all(_number(v) and 0 <= v <= 1 for v in p)
             and all(by_id[key]["calibrated_probability"] == by_id[key]["raw_probability"] for key in ids)
             and math.isclose(sum(p), 1., rel_tol=0, abs_tol=1e-6), "raw_probability")
    _require(np.allclose(p, metrics.probabilities([logits])[0], rtol=1e-5, atol=1e-6), "raw_T1_softmax")
    selected = response["selected_id"]
    _require(selected == ids[int(np.argmax(logits))], "selected_argmax")
    grade = target[ids.index(selected)] == max(target)
    _require(entry.get("selection_passed") is grade, "selection_grade")
    ordinal = None
    if request["primitive"] == "ordinal":
        ordinal = np.asarray([candidate["ordinal_value"] for candidate in candidates], float)
        _require(np.isfinite(ordinal).all() and (np.diff(ordinal) > 0).all(), "ordinal_order")
    image = request.get("image")
    if image is not None:
        _require(image == {"path": gold["image_path"], "sha256": gold["image_sha256"]}, "image_binding")
    else:
        _require(not gold.get("image_path"), "missing_image")
    track = entry["task_track"]
    _require(isinstance(track, str) and track and gold.get("group") is not None, "track_or_group")
    kev = None
    record_id = str(gold["id"])
    if panel == "kev-development":
        _require(image is None and track == gold["source"], "kev_text_track")
        kev = adapter.prediction_metadata(gold, p, input_order="fuzev")
        kev.update(logits=adapter.to_official_order(gold, logits), inference_temperature=1.)
        record_id = str(gold["official_record_id"])
    original = entry.get("original_item")
    _require((isinstance(original, dict) if panel == "original-fixed" else original is None), "original_item")
    normalized = {**request, "model": "native-4.05"}
    sampling = None
    if original is not None:
        _require(original["row"] == gold and original["request"] == normalized
                 and original["task_track"] == track, "original_item_alignment")
        sampling = _original_sampling(original)
    return Decision(request_id, index, panel, track, "image" if image is not None else "text",
                    request["primitive"], _digest([panel, gold["source"], gold["group"]]), record_id,
                    _digest([normalized, gold, original, track]), ids, np.asarray(target, float),
                    np.asarray(logits, float), np.asarray(p, float), selected, ordinal, kev, sampling)


def _summary(value, run, reference):
    _flags(value)
    _require(value.get("status") == "completed" and value.get("frozen_postcheck_status") == "pass"
             and value.get("production_quality_pass") is False
             and value.get("processed") == value.get("total") == TOTAL
             and value.get("model", "native-4.05" if reference else None) == run.model
             and value.get("metric_weighting") == WEIGHTING, "summary_contract")
    tracks = defaultdict(list)
    passed = 0
    for row in run.decisions.values():
        selected = row.options.index(row.selected_id)
        grade = float(row.target[selected] == row.target.max())
        passed += int(grade)
        tracks[(row.panel, row.track + ":tie_aware_accuracy")].append(grade)
        if np.count_nonzero(row.target == row.target.max()) == 1:
            tracks[(row.panel, row.track + ":accuracy")].append(grade)
        if row.ordinal is not None:
            error = abs(float(np.sum((row.p - row.target) * row.ordinal)))
            tracks[(row.panel, row.track + ":raw_probability_ordinal_mae")].append(error)
    _require(value.get("passed") == passed and _number(value.get("pass_rate"))
             and math.isclose(value["pass_rate"], passed / TOTAL, abs_tol=1e-12), "summary_totals")
    supplied = value["panels"]
    _require(set(supplied) == set(PANEL_COUNTS), "summary_panels")
    for panel in PANEL_COUNTS:
        expected = {name: observations for (p, name), observations in tracks.items() if p == panel}
        declared = supplied[panel]["metrics"]
        _require(set(declared) == set(expected), "summary_metric_keys")
        for name, observations in expected.items():
            metric = declared[name]
            _require(metric["n"] == len(observations) and _number(metric["mean"])
                     and math.isclose(metric["mean"], sum(observations) / len(observations),
                                      rel_tol=1e-10, abs_tol=1e-10), "summary_metric_value")


def _predictions(value, run):
    expected = {(row.kev["id"], row.kev["question"]): row.kev for row in run.decisions.values()
                if row.kev is not None}
    _require(isinstance(value, list) and len(value) == PANEL_COUNTS["kev-development"], "prediction_count")
    seen = set()
    for row in value:
        key = (row["id"], row["question"])
        _require(key in expected and key not in seen, "prediction_identity")
        # Option keys, official order, metadata, logits, probabilities and T1 must agree exactly.
        _require(row == expected[key], "prediction_alignment")
        seen.add(key)
    _require(seen == set(expected), "prediction_coverage")


def _load_run(directory, panel_pins, adapter, reference=False):
    run = Run(Path(directory).resolve())
    try:
        run.hashes = _snapshot(run.directory)
    except OSError:
        run.issues["unreadable_run_artifacts"] += 1
        return run
    try:
        run.model = _binding(_read(run.directory / "binding.json"), panel_pins, reference)
    except ContractError as error:
        run.issues[str(error)] += 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        run.issues["missing_or_malformed_binding"] += 1
    try:
        if run.hashes["health-provenance.json"] is not None:
            run.provenance = _read(run.directory / "health-provenance.json")
            _provenance(run.provenance)
        else:
            run.issues["missing_health_provenance"] += 1
    except ContractError as error:
        run.issues[str(error)] += 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        run.provenance = None
        run.issues["malformed_health_provenance"] += 1
    try:
        # Even invalid admission metadata must not erase failure/coverage evidence.
        first_type = last_type = None
        with (run.directory / "http.jsonl").open("rb") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    kind = entry["type"]
                    first_type = first_type or kind
                    last_type = kind
                    if kind == "health":
                        run.health.append(entry)
                        continue
                    _require(kind == "decision", "ledger_entry_type")
                    panel = entry.get("panel", "unknown")
                    run.observed[panel] += 1
                    request_id = entry["request_id"]
                    _require(isinstance(request_id, str), "request_id_type")
                    _require(request_id not in run.identities, "duplicate_request")
                    run.identities[request_id] = panel
                    row = _decision(entry, run.model, run.provenance, adapter)
                    run.decisions[request_id] = row
                except ContractError as error:
                    run.issues[str(error)] += 1
                except (ValueError, KeyError, TypeError, IndexError, OverflowError, AttributeError):
                    run.issues["malformed_ledger_entry"] += 1
        _require(len(run.health) == 2 and first_type == last_type == "health", "whole_panel_health_bookends")
        for entry in run.health:
            health = json.loads(entry["response_body"])
            _require(entry["status"] == 200 and health.get("model") == run.model
                     and health.get("status") == ("loaded" if run.model.startswith("vllm-") else "ok")
                     and health.get("provenance") == run.provenance, "health_binding_changed")
        _require(len(run.decisions) == TOTAL and len(run.identities) == TOTAL, "complete_request_coverage")
        _require({row.index for row in run.decisions.values()} == set(range(TOTAL)), "index_coverage")
        for panel, expected in PANEL_COUNTS.items():
            rows = [row for row in run.decisions.values() if row.panel == panel]
            _require(len(rows) == expected and len({row.record_id for row in rows}) == RECORD_COUNTS[panel],
                     "record_and_panel_coverage")
        _require(run.hashes["failure.json"] is None, "failure_artifact_present")
        _summary(_read(run.directory / "summary.json"), run, reference)
        _predictions(_read(run.directory / "kev-development-rows.json"), run)
    except ContractError as error:
        run.issues[str(error)] += 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        run.issues["missing_or_malformed_run_artifact"] += 1
    return run


def _values(row):
    value = metrics.per_record([row.logits], [row.target],
                               ordinal_values=None if row.ordinal is None else [row.ordinal], temperature=1.)
    return {name: float(result[0]) for name, result in value.items()}


def _interval(reference, candidate, groups, draws, seed):
    valid = np.isfinite(reference) & np.isfinite(candidate)
    n, count = int(valid.sum()), len(np.unique(groups[valid]))
    if not n:
        return {"eligible_requests": 0, "supported": False, "difference": None,
                "low": None, "high": None, "groups": 0, "draws": draws, "seed": seed}
    estimate = metrics.paired_group_interval(reference, candidate, groups, draws=draws, seed=seed)
    estimate.update(eligible_requests=n, supported=count >= 2)
    if count < 2:
        estimate.update(low=None, high=None)
    return estimate


def _section(reference, candidate, draws, seed):
    if not reference:
        return {"requests": 0, "metrics": {}, "drift": {}}
    groups = np.asarray([row.group for row in reference])
    left, right = [_values(row) for row in reference], [_values(row) for row in candidate]
    names = sorted(set().union(*(set(value) for value in left)))
    a = {name: np.asarray([value.get(name, np.nan) for value in left]) for name in names}
    b = {name: np.asarray([value.get(name, np.nan) for value in right]) for name in names}
    point_a, point_b = metrics.summarize(a), metrics.summarize(b)
    scored = {name: {"reference": point_a[name], "candidate": point_b[name],
                     "candidate_minus_reference": _interval(a[name], b[name], groups, draws, seed)}
              for name in names if name != "confidence" and name in point_a}
    scored["ece"] = {"reference": point_a["ece"], "candidate": point_b["ece"],
                     "uncertainty": "point_only; ECE is not an additive per-request metric"}
    drift = defaultdict(list)
    for old, new in zip(reference, candidate):
        drift["probability_total_variation"].append(.5 * float(np.abs(new.p - old.p).sum()))
        drift["probability_max_absolute_change"].append(float(np.abs(new.p - old.p).max()))
        drift["selection_flip_rate"].append(float(new.selected_id != old.selected_id))
        if old.ordinal is not None:
            delta = float(np.sum((new.p - old.p) * old.ordinal))
            shift = float(old.ordinal[old.options.index(new.selected_id)]
                          - old.ordinal[old.options.index(old.selected_id)])
            drift["ordinal_expected_value_change"].append(delta)
            drift["ordinal_expected_value_absolute_change"].append(abs(delta))
            drift["ordinal_selected_value_change"].append(shift)
            drift["ordinal_selected_value_absolute_change"].append(abs(shift))
        else:
            for name in ("ordinal_expected_value_change", "ordinal_expected_value_absolute_change",
                         "ordinal_selected_value_change", "ordinal_selected_value_absolute_change"):
                drift[name].append(np.nan)
    intervals = {name: _interval(np.zeros(len(reference)), np.asarray(values), groups, draws, seed)
                 for name, values in drift.items()}
    return {"requests": len(reference), "groups": len(np.unique(groups)),
            "sampling": evaluation_metrics.sampling_summary(np.ones(len(reference))),
            "metrics": scored, "drift": intervals}


def _weighted_track(reference, candidate):
    """Conditional original-panel points using the same aggregation as TEST.

    Do not coerce two research heads into compare_tracks' three fixed methods.
    Its sufficient-statistic helpers support honest reference/candidate names.
    """
    _require(reference and len(reference) == len(candidate)
             and all(old.panel == "original-fixed" and old.track == reference[0].track
                     and old.modality == reference[0].modality and old.primitive == reference[0].primitive
                     and old.request_id == new.request_id and old.input_digest == new.input_digest
                     and old.options == new.options and old.sampling == new.sampling
                     for old, new in zip(reference, candidate)), "weighted_alignment")
    n = len(reference)
    weights = np.asarray([row.sampling["combined_weight"] for row in reference])
    subset = np.asarray([row.sampling["subset_probability"] for row in reference])
    external_probability = np.asarray([row.sampling["external_probability"] for row in reference])
    external_weights = np.asarray([row.sampling["external_weight"] for row in reference])
    with np.errstate(over="ignore"):
        _require(np.isfinite(weights.sum() ** 2) and np.isfinite(np.square(weights).sum()),
                 "sampling_numeric_bounds")
    groups, inverse = np.unique([row.group for row in reference], return_inverse=True)
    indices = np.arange(n)
    points, eligible_requests, eligible_weights = [], {}, {}
    for rows in (reference, candidate):
        records = [_values(row) for row in rows]
        keys = sorted(set().union(*(set(value) for value in records)))
        values = {name: np.asarray([value.get(name, np.nan) for value in records]) for name in keys}
        names, matrix = evaluation_metrics._aggregate(values, None, indices, inverse, len(groups), weights)
        totals = matrix.sum(axis=0)
        points.append(evaluation_metrics._summarize(names, totals))
        eligible_requests = {name[:-6]: int(total) for name, total in zip(names, totals)
                             if name.endswith("/count") and not name.startswith("bin/")}
        eligible_weights = {name[:-7]: float(total) for name, total in zip(names, totals)
                            if name.endswith("/weight")}
    scored = {}
    for name, old in points[0].items():
        new = points[1][name]
        finite = np.isfinite(old) and np.isfinite(new)
        scored[name] = {"reference": float(old) if finite else None,
                        "candidate": float(new) if finite else None,
                        "candidate_minus_reference": float(new - old) if finite else None,
                        "eligible_requests": n if name == "ece" else eligible_requests[name],
                        "eligible_weight": float(weights.sum()) if name == "ece" else eligible_weights[name]}
    sampling = evaluation_metrics.sampling_summary(weights, subset * external_probability)
    sampling["subset"] = evaluation_metrics.sampling_summary(1. / subset, subset)
    sampling["original_external"] = evaluation_metrics.sampling_summary(external_weights, external_probability)
    strata = np.asarray([row.sampling["label_stratum"] for row in reference])
    sampling["represented_label_strata"] = {
        stratum: {"requests": int((strata == stratum).sum()),
                  "groups": len(np.unique(inverse[strata == stratum])),
                  **evaluation_metrics.sampling_summary(weights[strata == stratum],
                      (subset * external_probability)[strata == stratum])} for stratum in np.unique(strata)}
    row = reference[0]
    return {"n": n, "groups": len(groups), "modality": row.modality, "primitive": row.primitive,
            "status": "highly-experimental" if row.modality == "image"
                      or row.track.startswith(("qasper/", "vl-rewardbench/")) else "research-only",
            "metrics": scored, "sampling": sampling, "weighting_scope": WEIGHTING_SCOPE,
            "uncertainty": "points_only; unweighted CIs do not apply to these weighted estimates"}


def compare_runs(candidate_directory, *, draws=2000, seed=17, adapter_path=None):
    """Return aggregate-only evidence; blocked runs never get subset statistics."""
    _require(type(draws) is int and 1 <= draws <= 10000
             and type(seed) is int and 0 <= seed < 2 ** 32, "bootstrap_settings")
    report = {"schema": "kev-cross-compare-v1", "status": "blocked", **FLAGS,
              "production_quality_pass": False, "head_sha256": HEAD_SHA,
              "calibration_id": "uncalibrated:T1", "inference_temperature": 1.,
              "manifest_sha256": MANIFEST_SHA, "records": 3990, "requests": TOTAL,
              "metric_weighting": WEIGHTING, "bootstrap": {"draws": draws, "seed": seed,
              "unit": "saved gold group within panel/source; whole groups, request-weighted means",
              "confidence_level": .95, "multiplicity_adjustment": "none; descriptive, no gates"},
              "visual_status": "highly-experimental; no production qualification",
              "official_kev_index": "not recomputed; official score_system/bootstrap remain separate",
              "weighting_scope": WEIGHTING_SCOPE, "original_fixed_weighted_per_track": None,
              "metrics_source_sha256": {Path(module.__file__).name: _sha(module.__file__)
                                         for module in (metrics, evaluation_metrics)},
              "panels": None, "issues": []}
    report["recorded_reference_pins"] = REFERENCE_PINS
    try:
        manifest, panel_pins = _manifest()
        adapter = _adapter(manifest, adapter_path)
        report.update(manifest_sha256=_sha(MANIFEST), recorded_manifest_sha256=MANIFEST_SHA,
                      adapter_sha256=_sha(adapter.__file__), recorded_adapter_sha256=manifest.get("adapter_sha256"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["issues"] = [str(error) if isinstance(error, ContractError) else "manifest_or_adapter_unreadable"]
        return report
    reference = _load_run(REFERENCE_DIRECTORY, panel_pins, adapter, reference=True)
    self_check = Path(candidate_directory).resolve() == Path(REFERENCE_DIRECTORY).resolve()
    report["reference_self_check"] = self_check
    candidate = _load_run(candidate_directory, panel_pins, adapter, reference=self_check)
    for name, run in (("reference", reference), ("candidate", candidate)):
        report[name] = {"directory": str(run.directory), "model": run.model,
                        "coverage": run.coverage(), "artifact_sha256": run.hashes,
                        "provenance_sha256": _digest(run.provenance) if run.provenance else None}
    if not reference.issues and not candidate.issues:
        if set(reference.decisions) != set(candidate.decisions):
            report["issues"].append("aligned_request_set")
        else:
            for key, old in reference.decisions.items():
                new = candidate.decisions[key]
                if old.input_digest != new.input_digest or old.options != new.options:
                    report["issues"].append("aligned_input_options_gold_or_sampling_changed")
                    break
        if not report["issues"]:
            panels = {}
            for panel in PANEL_COUNTS:
                modalities = {}
                for modality in ("text", "image"):
                    old = sorted((row for row in reference.decisions.values()
                                  if row.panel == panel and row.modality == modality), key=lambda row: row.request_id)
                    new = [candidate.decisions[row.request_id] for row in old]
                    result = _section(old, new, draws, seed)
                    result["status"] = "highly-experimental" if modality == "image" else "research-only"
                    result["tracks"] = {track: {**_section([row for row in old if row.track == track],
                        [row for row in new if row.track == track], draws, seed),
                        "status": "highly-experimental" if modality == "image"
                            or track.startswith(("qasper/", "vl-rewardbench/")) else "research-only"}
                        for track in sorted({r.track for r in old})}
                    modalities[modality] = result
                panels[panel] = {"requests": PANEL_COUNTS[panel], "records": RECORD_COUNTS[panel],
                                 "modalities": modalities}
            original = sorted((row for row in reference.decisions.values() if row.panel == "original-fixed"),
                              key=lambda row: row.request_id)
            try:
                weighted = {track: _weighted_track([row for row in original if row.track == track],
                    [candidate.decisions[row.request_id] for row in original if row.track == track])
                    for track in sorted({row.track for row in original})}
                report.update(status="complete", panels=panels, original_fixed_weighted_per_track=weighted)
            except ContractError as error:
                report["issues"].append(str(error))
    for name, run in (("reference", reference), ("candidate", candidate)):
        try:
            _require(_snapshot(run.directory) == run.hashes, "artifact_bookend_changed")
        except (OSError, ContractError):
            report["issues"].append(name + ":artifact_bookend_changed")
        report["issues"].extend(name + ":" + code for code in sorted(run.issues))
    try:
        manifest_unchanged = _sha(MANIFEST) == report["manifest_sha256"]
    except OSError:
        manifest_unchanged = False
    if not manifest_unchanged:
        report["issues"].append("manifest_bookend_changed")
    if report["issues"]:
        report.update(status="blocked", panels=None, original_fixed_weighted_per_track=None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--adapter", type=Path, default=Path.cwd() / "scripts/kev_requests_adapter.py")
    args = parser.parse_args(argv)
    report = compare_runs(args.candidate, draws=args.draws, seed=args.seed, adapter_path=args.adapter)
    # Never overwrite an earlier attempt; aggregated reports are private too.
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
        handle.write(_encoded(report))
    print(json.dumps({"status": report["status"], "output": str(args.output), **FLAGS}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
