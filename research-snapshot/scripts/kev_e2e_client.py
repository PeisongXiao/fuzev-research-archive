"""Explicit research-only native HTTP E2E: the full fixed 5075 requests.

Set FUZEV_ROOT to the artifact directory and run from the snapshot:
    python scripts/kev_e2e_client.py --allow-inference

Historical lab driver; reads unpublished reports under $FUZEV_ROOT/reports, so it runs only with the lab's inputs.
Full HTTP bodies remain in a private, exclusive report directory.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from kev_e2e_resume import resume_source, run_lock


ROOT = Path(os.environ["FUZEV_ROOT"]).expanduser().resolve()
HERE = Path(__file__).resolve().parent
REPORT = ROOT / "reports/2026-10-10-kev-text-evaluation"
OUTPUT = REPORT / "e2e-native-v1"
MANIFEST = ROOT / "data/evaluation/e2e-with-kev-v1/manifest.json"
MANIFEST_SHA = "452c065912437fddc54f333bc73d17d6d1c560ccb491dfa827adfb43494e786e"
PLAN = REPORT / "frozen-plan-v1.json"
PLAN_SHA = "252f43ac3d7eab82c3d3be8db468da17798f4766e6b083e505da72c98bffcf3e"
QUALITY = REPORT / "native-results-v1/result.json"
QUALITY_SHA = "83464ee250883e186876195c4bd5421d27bce07a1968cd13a39b5243da34eb9b"
HEAD_SHA = "f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c"
MODEL, HEAD, TASK, TOTAL = "native-4.05", "reference", "validation.kev-e2e", 5075
FLAGS = {"experimental": True, "calibrated": False, "qualified": False, "servable": False}
REFERENCE_MODEL = "native-4.05"
ACQUISITION = ROOT / "data/manifests/acquisition.json"
NATIVE_REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
MODEL_REVISIONS = {
    "native-3.05": "69e33439ae950f17bcbe95c98f117d80f759ab6d",
    "native-4.05": "55a732e0c4c3d4614bc42b68493bb930d9b02c0a",
    "native-5.05": "7cef615f7fd3681295b68848018876dfabc336c7",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    with Path(path).open("rb") as handle:
        return [json.loads(line) for line in handle]


def verify(pins):
    return {str(path): sha(path) for path in pins}


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def request_for(row):
    # Frozen files always contain canonical native4 bodies, even for cross-cell runs.
    value = {"model": REFERENCE_MODEL, "head_id": HEAD, "messages": row["context"],
             "primitive": row["primitive"], "question": row["question"], "candidates": row["candidates"]}
    if row.get("image_path"):
        value["image"] = {"path": row["image_path"], "sha256": row["image_sha256"]}
    return value


def wire_request(request):
    require(request["model"] == REFERENCE_MODEL and MODEL in MODEL_REVISIONS, "Wrong canonical/native wire model")
    return request if MODEL == REFERENCE_MODEL else {**request, "model": MODEL}


def serving_identity_for(model_name, model, original):
    require(original["hidden_size"] == 2560 and original["cache_precision"] == "float16"
            and original["feature_tap"] == "final-mixer/post-collapse/before-lm-head", "Incompatible canonical reference")
    runtime = original["readout_runtime"]
    directory = Path(model["local_snapshot"])
    metadata = read(directory / "config.json")
    text = metadata.get("text_config", metadata)
    require(text.get("hidden_size") == 2560 and text.get("max_position_embeddings") == 262144
            and metadata.get("quantization_config", {}).get("quant_method") == "exl3",
            "Incompatible acquired native geometry/quantization")
    return {"cell": model_name, "model": model["repo"], "revision": model["revision"],
            "recorded_model": NATIVE_REPO, "recorded_revision": MODEL_REVISIONS.get(model_name),
            "local_snapshot": str(directory), "encoding": original["encoding"],
            "feature_tap": original["feature_tap"], "hidden_size": 2560, "cache_precision": "float16",
            "readout_runtime": dict(runtime), "ple": "NVMe mmap",
            "config_sha256": sha(directory / "config.json"),
            "quantization_config_sha256": sha(directory / "quantization_config.json")}


def selection_grade(row, selected_id):
    ids, target = [c["id"] for c in row["candidates"]], row["target"]
    require(row["role"] == "test" and len(ids) == len(target) and selected_id in ids,
            "Invalid evaluation role, target length or selected ID")
    require(all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in target)
            and math.isclose(sum(target), 1., rel_tol=0., abs_tol=1e-7), "Invalid gold target")
    return target[ids.index(selected_id)] == max(target)


def load_inputs():
    """Validate all frozen evidence and every request before even health HTTP."""
    from fuzev_research.service_contracts import ServiceDecisionRequest, TextMessage
    from kev_requests_adapter import iter_request_rows

    pins = {str(MANIFEST): MANIFEST_SHA, str(PLAN): PLAN_SHA, str(QUALITY): QUALITY_SHA}
    verify(pins)
    plan, quality, manifest = read(PLAN), read(QUALITY), read(MANIFEST)
    require(quality["status"] == "pass" and quality["primary"] == "lm-assisted"
            and quality["comparison"]["passed"] is True
            and quality["test_sha256"] == plan["test_sha256"]
            and quality["uncertainty"]["index"]["lm-assisted"][0] > plan["baseline_index_ci"][1]
            and quality["held_data_used_for_tuning"] is False and quality["calibrated"] is False
            and quality["qualified"] is False and quality["servable"] is False,
            "Exact frozen primary test quality must pass; no calibration/provenance bypass")
    require(quality["systems"]["lm-assisted"]["overall"]["answered_questions"] == 3089
            and plan["primary"] == "lm-assisted" and plan["temperature"] == 1.
            and plan["training_allowed"] is False and plan["calibration_allowed"] is False,
            "Wrong frozen quality population or inference policy")
    case = plan["candidates"]["lm-assisted"]
    pins.update(plan["input_sha256"])
    for filename, expected in case["source8"].items():
        pins[str(Path(case["tree"]) / "fuzev_research" / filename)] = expected
    pins[case["weights"]] = HEAD_SHA
    original_identity = read(plan["original_feature_identity"])
    pins[plan["original_feature_identity"]] = sha(plan["original_feature_identity"])
    pins[str(ACQUISITION)] = sha(ACQUISITION)
    model = read(ACQUISITION)["models"][MODEL]
    serving_identity_for(MODEL, model, original_identity)
    for filename in ("config.json", "quantization_config.json"):
        path = Path(model["local_snapshot"]) / filename
        pins[str(path)] = sha(path)
    panels = manifest["panels"]
    old, kev = panels["original-fixed"], panels["kev-development"]
    require(manifest["combined_records"] == 3990 and manifest["combined_questions_or_requests"] == TOTAL
            and old["records"] == old["requests"] == 2000
            and kev["records"] == 1990 and kev["questions"] == 3075
            and manifest["external_evaluation_only"] is True and manifest["new_data_fitting"] is False
            and all(manifest[k] is False for k in ("training_allowed", "calibration_allowed", "model_selection_allowed")),
            "Changed fixed evaluation population or ownership")
    pins.update({old["path"]: old["sha256"], kev["source"]: kev["source_sha256"],
                 kev["requests"]: kev["requests_sha256"], kev["expected"]: kev["expected_sha256"],
                 str(HERE / "kev_requests_adapter.py"): manifest["adapter_sha256"],
                 str(HERE / "prepare_kev_e2e_panel.py"): manifest["preparation_source_sha256"],
                 str(HERE / "kev_e2e_resume.py"): sha(HERE / "kev_e2e_resume.py"),
                 str(Path(__file__)): sha(__file__)})
    verify(pins)
    fixed = read(old["path"])
    require(len(fixed["rows"]) == 2000, "Incomplete original fixed panel")
    items = [{**item, "panel": "original-fixed"} for item in fixed["rows"]]
    require(Counter(item["task_track"] for item in items) == fixed["quotas"], "Original quotas changed")
    records, requests, expected = jsonl(kev["source"]), jsonl(kev["requests"]), jsonl(kev["expected"])
    require(len(records) == 1990 and sum(len(r["questions"]) for r in records) == 3075
            and len(requests) == len(expected) == 3075, "Incomplete Kev development panel")
    for row, supplied, gold in zip(iter_request_rows(records, split="development"), requests, expected):
        common = {"id": row["id"], "record_id": row["official_record_id"],
                  "question_id": row["official_question_id"], "source": row["source"]}
        require(supplied == {**common, "request": request_for(row)}, "Kev request/order differs from frozen source")
        require(gold == {**common, **{k: row[k] for k in
                ("target", "official_keys", "candidate_keys", "official_label_index")}}, "Kev expected row misaligned")
        items.append({"panel": "kev-development", "row": row, "request": supplied["request"],
                      "task_track": row["source"]})
    require(len(items) == TOTAL, "Require the full 5075 requests; no sampling or limit")
    for index, item in enumerate(items):
        require(item["request"] == request_for(item["row"]), "Fixed request differs from unchanged gold row")
        ServiceDecisionRequest.model_validate(item["request"])
        for message in item["request"]["messages"]:
            TextMessage.model_validate(message)
        selection_grade(item["row"], item["row"]["candidates"][0]["id"])
        identity = encoded([MANIFEST_SHA, item["panel"], index, item["row"]["id"]])
        item["request_id"] = "kev_" + hashlib.sha256(identity).hexdigest()
    require(len({item["request_id"] for item in items}) == TOTAL, "Duplicate HTTP request IDs")
    verify(pins)
    return items, pins, original_identity


def git_head(directory):
    from fuzev_research.paths import source_revision
    return source_revision(directory, "fuzev-research")


def health_check(health, original, binding=None):
    """Bind newly authorized serving sources at first health, never select a head."""
    require(health.get("status") == "ok" and health.get("model") == MODEL, "Research service not healthy/selected native cell")
    provenance = health.get("provenance")
    require(isinstance(provenance, dict), "Missing actual loaded research provenance")
    require(all(provenance.get(k) is v for k, v in FLAGS.items()), "Not explicit unqualified/raw-T1 research")
    require(provenance.get("learned_head_count") == 1
            and provenance.get("head_lane") == "lm-assisted"
            and provenance.get("calibration_id") == "uncalibrated:T1"
            and type(provenance.get("raw_inference_temperature")) in (int, float)
            and provenance["raw_inference_temperature"] == 1., "Wrong research head/T1 binding")
    acquisition_sha = sha(ACQUISITION)
    model = read(ACQUISITION)["models"][MODEL]
    caps = health.get("capabilities", {})
    require(all(caps.get(k) is True for k in ("typed_decisions", "trusted_still_images", "shared_lm_assisted_head")),
            "Required research APIs unavailable")
    source_pins = provenance.get("source_pins", {})
    require(isinstance(source_pins, dict), "Source provenance must be an object")
    by_name = {Path(path).name: Path(path) for path in source_pins}
    import fuzev_research.contracts as contracts
    import fuzev_research.service_contracts as service_contracts
    recipe_sources = provenance.get("recipe_sources", {})
    recipe_path = by_name.get("server.py")
    provenance = {**provenance, "observed_recipe_revision": git_head(
        recipe_path.parent.parent if recipe_path is not None else Path(contracts.__file__).parent.parent),
        "recorded_recipe_sources": recipe_sources, "observed_source_pins": verify(source_pins),
        "observed_client_sources": {Path(module.__file__).name: sha(module.__file__)
                                    for module in (contracts, service_contracts)},
        "observed_acquisition_manifest_sha256": acquisition_sha,
        "recorded_reference_feature_identity": original,
        "observed_serving_identity": serving_identity_for(MODEL, model, original),
        "recorded_readout_runtime": original["readout_runtime"],
        "observed_readout_runtime": health.get("readout_runtime"),
        "recorded_head_sha256": HEAD_SHA, "recorded_plan_sha256": PLAN_SHA,
        "recorded_quality_sha256": QUALITY_SHA}
    if binding is not None:
        require(provenance == binding, "Loaded provenance changed during E2E")
    return provenance


def response_check(response, request, request_id):
    from fuzev_research.contracts import DecisionResponse
    validated = DecisionResponse.model_validate(response)
    require(validated.id == request_id and all(response.get(k) == request[k] for k in ("model", "primitive", "head_id")),
            "Response request/model/head/primitive mismatch")
    require(all(response.get(k) is v for k, v in FLAGS.items())
            and validated.calibration_id == "uncalibrated:T1", "Response research head/T1 provenance differs")
    scores = [score.model_dump() for score in validated.scores]
    ids = [candidate["id"] for candidate in request["candidates"]]
    require([score["id"] for score in scores] == ids, "Response candidate order/count differs")
    logits = [score["logit"] for score in scores]
    require(all(math.isfinite(z) for z in logits), "Nonfinite response logits")
    raw, legacy = ([score[key] for score in scores] for key in ("raw_probability", "calibrated_probability"))
    require(raw == legacy and all(math.isfinite(p) and 0 <= p <= 1 for p in raw)
            and math.isclose(sum(raw), 1., rel_tol=0., abs_tol=1e-6), "Invalid or nonidentical raw/legacy T1 probabilities")
    exps = [math.exp(z - max(logits)) for z in logits]
    require(all(math.isclose(p, e / sum(exps), rel_tol=1e-5, abs_tol=1e-6) for p, e in zip(raw, exps)),
            "Probabilities do not match raw T1 logits")
    require(validated.selected_id == ids[max(range(len(ids)), key=logits.__getitem__)], "Selection disagrees with logit argmax")
    return scores


def publish_progress(store, status, processed, passed):
    from fuzev_research.acquire import write_json
    rate = passed / processed if processed else None
    write_json(ROOT / "runs/validation-progress.json", {
        "run_id": OUTPUT.name, "stage": "e2e", "status": status, "processed": processed,
        "model": MODEL, "total": TOTAL, "passed": passed, "pass_rate": rate, "updated_at": time.time(), **FLAGS})
    store.update(TASK, status, f"Research {MODEL} CUDA HTTP: {processed}/{TOTAL}; raw T1, no tuning.", processed / TOTAL)
    store.record_result(TASK, {"validation_pass_rate": rate, "validation_kind": f"Research {MODEL} HTTP E2E, raw T1",
                              "processed": processed, "total": TOTAL, "status": status, **FLAGS})


def run(timeout, *, port=8000, input_loader=load_inputs, request_mapper=wire_request,
        health_validator=health_check, response_validator=response_check,
        task_parent="validation", task_title="Research native CUDA HTTP E2E: fixed original + Kev development",
        resume_from=None):
    import httpx
    from fuzev_research.state import Store
    from kev_requests_adapter import prediction_metadata, to_official_order

    items, pins, original = input_loader()
    observed_pins = verify(pins)
    head_path = read(PLAN)["candidates"]["lm-assisted"]["weights"]
    binding_document = {"manifest_sha256": observed_pins.get(str(MANIFEST)),
                 "plan_sha256": observed_pins.get(str(PLAN)),
                 "quality_sha256": observed_pins.get(str(QUALITY)), "head_sha256": sha(head_path),
                 "input_sha256": observed_pins, "recorded_input_sha256": pins,
                 "recorded_manifest_sha256": MANIFEST_SHA, "recorded_plan_sha256": PLAN_SHA,
                 "recorded_quality_sha256": QUALITY_SHA, "recorded_head_sha256": HEAD_SHA,
                 "model": MODEL, "canonical_request_model": REFERENCE_MODEL,
                 "streams": 1, "retries": 0, "requests": TOTAL, "records": 3990, "resume_format": 1, **FLAGS}
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        require(resume_from.parent == REPORT and resume_from.name.startswith("e2e-")
                and resume_from != OUTPUT, 'Resume only a prior research attempt into a new exclusive directory')
    OUTPUT.mkdir(mode=0o700, parents=False, exist_ok=False)
    private_json(OUTPUT / "binding.json", binding_document)
    store = None
    processed = passed = reused = 0
    tracks, canonical = defaultdict(list), []
    started = time.monotonic()
    leases = ExitStack()
    source_leases = ExitStack()
    try:
        leases.enter_context(run_lock(OUTPUT))
        recovery = source_leases.enter_context(resume_source(resume_from, binding_document, verify))
        store = Store()
        store.initialize()
        with store.connect() as db:
            db.execute("INSERT OR IGNORE INTO tasks(id,parent,title,updated) VALUES(?,?,?,?)",
                       (TASK, task_parent, task_title, time.time()))
        publish_progress(store, "in_progress", processed, passed)
        with os.fdopen(os.open(OUTPUT / "http.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as journal:
            def save(value):
                journal.write(encoded(value))
                journal.flush()
                os.fsync(journal.fileno())

            def account(item, request, response, scores):
                nonlocal processed, passed
                grade = selection_grade(item["row"], response["selected_id"])
                target = item["row"]["target"]
                values = {"tie_aware_accuracy": float(grade)}
                if target.count(max(target)) == 1:
                    values["accuracy"] = float(grade)
                if item["row"]["primitive"] == "ordinal":
                    values["raw_probability_ordinal_mae"] = abs(sum(
                        (s["raw_probability"] - t) * c["ordinal_value"]
                        for s, t, c in zip(scores, target, request["candidates"])))
                for metric, value in values.items():
                    tracks[(item["panel"], item["task_track"], metric)].append(value)
                if item["panel"] == "kev-development":
                    row = prediction_metadata(item["row"], [s["raw_probability"] for s in scores], input_order="fuzev")
                    row["logits"] = to_official_order(item["row"], [s["logit"] for s in scores])
                    row["inference_temperature"] = 1.
                    canonical.append(row)
                processed += 1
                passed += int(grade)
                return grade
            with httpx.Client(base_url=f"http://localhost:{port}", timeout=timeout, trust_env=False,
                              follow_redirects=False, transport=httpx.HTTPTransport(retries=0)) as client:
                def get_health():
                    reply = client.get("/research-health")
                    if reply.status_code != 200:
                        save({"type": "health", "status": reply.status_code, "response_body": reply.text})
                    reply.raise_for_status()
                    return {"type": "health", "status": reply.status_code, "response_body": reply.text}
                first_health = get_health()
                binding = health_validator(json.loads(first_health["response_body"]), original)
                private_json(OUTPUT / "health-provenance.json", binding)
                verify(pins)
                if recovery is not None:
                    require(recovery.provenance == binding,
                            'Loaded model/runtime/source provenance changed; previous responses cannot be mixed')
                    for entry in recovery.decisions():
                        require(processed < TOTAL, 'Resume journal exceeds the fixed population')
                        if processed == 0:
                            require(health_validator(json.loads(recovery.initial_health["response_body"]), original) == binding,
                                    'Original session health does not validate against the loaded runtime')
                        item = items[processed]
                        request = request_mapper(item["request"])
                        expected = {"index": processed, "panel": item["panel"], "request_id": item["request_id"],
                                    "request": request, "gold": item["row"], "task_track": item["task_track"],
                                    "original_item": item if item["panel"] == "original-fixed" else None}
                        require(all(entry.get(key) == value for key, value in expected.items()),
                                'Resume request, gold or ordering differs from the frozen panel')
                        response = json.loads(entry["response_body"])
                        scores = response_validator(response, request, item["request_id"])
                        grade = selection_grade(item["row"], response["selected_id"])
                        require(entry.get("selection_passed") is grade
                                and type(entry.get("elapsed_seconds")) in (int, float)
                                and math.isfinite(entry["elapsed_seconds"]) and entry["elapsed_seconds"] >= 0,
                                'Resume grade or timing is invalid')
                        if processed == 0:
                            save(recovery.initial_health)
                        save(entry)
                        account(item, request, response, scores)
                    reused = processed
                    if not reused:
                        require(health_validator(json.loads(recovery.initial_health["response_body"]), original) == binding,
                                'Original session health does not validate against the loaded runtime')
                        save(first_health)
                    private_json(OUTPUT / "resume.json", {**recovery.receipt(), "session_initial_health": first_health})
                    publish_progress(store, "in_progress", processed, passed)
                else:
                    save(first_health)
                for index in range(processed, len(items)):
                    item = items[index]
                    request = request_mapper(item["request"])
                    entry = {"type": "decision", "index": index, "panel": item["panel"],
                             "request_id": item["request_id"], "request": request, "gold": item["row"],
                             "task_track": item["task_track"], "original_item": item if item["panel"] == "original-fixed" else None}
                    tick = time.monotonic()
                    try:
                        reply = client.post("/v1/decisions", content=encoded(request),
                                            headers={"Content-Type": "application/json", "x-fuzev-request-id": item["request_id"]})
                        entry.update(status=reply.status_code, response_body=reply.text)
                        reply.raise_for_status()
                        response = reply.json()
                        scores = response_validator(response, request, item["request_id"])
                        grade = selection_grade(item["row"], response["selected_id"])
                        entry["selection_passed"] = grade
                    except BaseException as error:
                        entry["error"] = f"{type(error).__name__}: {error}"
                        raise
                    finally:
                        entry["elapsed_seconds"] = time.monotonic() - tick
                        save(entry)
                    account(item, request, response, scores)
                    if processed % 25 == 0 or processed == TOTAL:
                        publish_progress(store, "in_progress", processed, passed)
                last_health = get_health()
                save(last_health)
                health_validator(json.loads(last_health["response_body"]), original, binding)
        source_leases.close()
        verify(pins)
        require(processed == TOTAL and len(canonical) == 3075, "Incomplete HTTP population; no completion claim")
        private_json(OUTPUT / "kev-development-rows.json", canonical)
        summary = {"status": "completed", "model": MODEL, "processed": processed, "total": TOTAL, "passed": passed,
                   "pass_rate": passed / processed, "elapsed_seconds": time.monotonic() - started,
                   "resumed_requests": reused, "submitted_requests": processed - reused,
                   "elapsed_scope": "This session only; original request timings retained in the journal",
                   "frozen_postcheck_status": "pass", "production_quality_pass": False, **FLAGS,
                   "metric_weighting": "Unweighted descriptive fixed-panel means; not population estimates or a quality gate",
                   "panels": {panel: {"metrics": {track + ":" + metric: {"n": len(values), "mean": sum(values) / len(values)}
                              for (p, track, metric), values in tracks.items() if p == panel}}
                              for panel in ("original-fixed", "kev-development")},
                   "kev_scoring": "Canonical official-order rows with logits; Main runs official score_system/bootstrap separately"}
        private_json(OUTPUT / "summary.json", summary)
        publish_progress(store, "completed", processed, passed)
        print(json.dumps({"status": "completed", "processed": processed, "passed": passed,
                          "pass_rate": passed / processed, "output": str(OUTPUT), **FLAGS}))
        return 0
    except BaseException as error:
        private_json(OUTPUT / "failure.json", {"status": "failed", "processed": processed, "total": TOTAL,
                     "passed": passed, "frozen_postcheck_status": "fail", "error": f"{type(error).__name__}: {error}", **FLAGS})
        if store is not None:
            publish_progress(store, "failed", processed, passed)
        raise
    finally:
        try:
            source_leases.close()
        finally:
            leases.close()


def main():
    global OUTPUT, MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-inference", action="store_true", required=True)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS), default=REFERENCE_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Reuse a validated prefix from an inactive attempt; output stays exclusive")
    args = parser.parse_args()
    try:
        MODEL = args.model
        default_output = REPORT / ("e2e-native-v1" if MODEL == REFERENCE_MODEL else f"e2e-{MODEL}-v1")
        OUTPUT = (args.output_dir or default_output).resolve()
        prefix = "e2e-native-v" if MODEL == REFERENCE_MODEL else f"e2e-{MODEL}-v"
        require(OUTPUT.parent == REPORT and OUTPUT.name.startswith(prefix),
                'Use a new private E2E attempt directory; never overwrite earlier evidence')
        require(os.environ.get("FUZEV_ROOT") == str(ROOT), "Explicit FUZEV_ROOT required")
        require(math.isfinite(args.timeout) and 0 < args.timeout <= 3600, "Invalid timeout")
        sys.path.insert(0, str(HERE.parent))
        return run(args.timeout, resume_from=args.resume_from, port=args.port)
    except BaseException as error:
        # Validation exceptions can embed examples: details belong only in private reports.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "output": str(OUTPUT)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
