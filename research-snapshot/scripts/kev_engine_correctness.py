"""Separate synthetic prefix/long-input correctness probe, never quality grading."""

import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import kev_e2e_client as canonical
from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.vllm_receipts import readout_receipt, readout_sources


REPORT = canonical.ROOT / "reports/2026-10-10-cross-cell"
MODELS = ("native-3.05", "native-4.05", "native-5.05", "vllm-nvidia",
          "vllm-radixark", "vllm-exl3-4.05")
TIERS = (2048, 8192, 16384, 32768)
MAX_NEW = 2
PROBABILITY_ATOL = 2e-6
HEADER = "Synthetic engine buffer. This text is not a quality evaluation.\n"
UNIT = "Buffer entry: code ALPHA, value 17, slot 03, marker stable.\n"
END = "End of synthetic buffer.\n"
CANDIDATES = [{"id": "alpha", "text": "Marker ALPHA."},
              {"id": "beta", "text": "Marker BETA."}]
QUESTIONS = {"A": "Which marker belongs to branch A?",
             "B": "Which marker belongs to branch B?"}
CONTRACT = {"schema_version": 1, "scope": "engine correctness only; NOT quality grading",
            "tiers": list(TIERS), "max_submitted_input_tokens": 32768,
            "generation_max_tokens": MAX_NEW, "sequence": ["A", "A", "B", "A"],
            "probability_absolute_tolerance": PROBABILITY_ATOL, "probability_relative_tolerance": 0.,
            "successful_decision_posts": 16, "deleted_prefix_404_posts": 4,
            "generation_posts": 4, "gold_labels": False, "accuracy": False,
            "cpu_parity": False, "training_or_calibration": False, **canonical.FLAGS}


def require(value, message):
    canonical.require(value, message)


def encode(tokenizer, text):
    tokens = tuple(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    require(tokens and all(type(x) is int and x >= 0 for x in tokens), "Invalid offline token IDs")
    return tokens


def fragments(tokenizer):
    suffixes = {key: encode(tokenizer, query_suffix(question, CANDIDATES))
                for key, question in QUESTIONS.items()}
    tails = [encode(tokenizer, candidate_suffix(candidate)) for candidate in CANDIDATES]
    return suffixes, tails


def fixtures(reference_tokenizer, actual_tokenizer):
    """Always construct the same strings using canonical native4 tokenizer metadata."""
    suffixes, tails = fragments(reference_tokenizer)
    margin = MAX_NEW + max(map(len, suffixes.values())) + max(map(len, tails))
    actual_suffixes, actual_tails = fragments(actual_tokenizer)
    actual_margin = MAX_NEW + max(map(len, actual_suffixes.values())) + max(map(len, actual_tails))
    result = []
    for tier in TIERS:
        low, high = 0, tier
        while low < high:
            middle = (low + high + 1) // 2
            if len(encode(reference_tokenizer, HEADER + UNIT * middle + END)) + margin <= tier:
                low = middle
            else:
                high = middle - 1
        text = HEADER + UNIT * low + END
        tokens = encode(actual_tokenizer, text)
        require(low > 0 and tier - 128 <= len(tokens) + actual_margin <= tier,
                "Canonical fixture does not fit actual tokenizer/tier; no shortening or truncation")
        result.append({"tier": tier, "text": text, "tokens": tokens, "repetitions": low,
                       "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "maximum_submitted_tokens": len(tokens) + actual_margin})
    return result


def request_id(model, tier, phase):
    value = [CONTRACT, model, tier, phase]
    return "engine_" + hashlib.sha256(canonical.encoded(value)).hexdigest()


def cached_count(value, count):
    require(value is None or type(value) is int and 0 <= value <= count, "Invalid cache count")
    return value


def generation_check(value, model, identifier, prompt_tokens):
    require(value.get("id") == identifier and value.get("model") == model
            and value.get("object") == "text_completion", "Generation request binding differs")
    choices = value.get("choices", [])
    require(isinstance(choices, list) and len(choices) == 1
            and type(choices[0].get("index")) is int and choices[0]["index"] == 0
            and choices[0].get("finish_reason") in ("stop", "length")
            and isinstance(choices[0].get("text"), str), "Invalid generation response shape")
    completion = value.get("completion_token_ids")
    minimum = 1 if model.startswith("native-") else 0
    require(isinstance(completion, list) and minimum <= len(completion) <= MAX_NEW
            and all(type(x) is int and x >= 0 for x in completion), "Invalid accepted output IDs")
    accepted = prompt_tokens + tuple(completion)
    usage = value.get("usage", {})
    expected = {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(completion),
                "total_tokens": len(accepted)}
    require(all(type(usage.get(k)) is int and usage[k] == v for k, v in expected.items()),
            "Generation token counts differ from actual offline/accepted tokens")
    require(value.get("accepted_token_sha256") == token_hash(accepted), "Accepted token hash differs")
    prefix_id = value.get("prefix_id")
    require(isinstance(prefix_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prefix_id)
            and isinstance(value.get("prefix_identity"), str) and value["prefix_identity"],
            "Missing or invalid accepted prefix identity")
    if model.startswith("native-"):
        consumed = value.get("consumed_tokens")
        require(type(consumed) is int and consumed == len(accepted) - 1
                and value.get("unconsumed_token_ids") == list(accepted[consumed:]),
                "Native checkpoint must expose its actual consumed boundary and pending token")
    else:
        receipt = value.get("execution", {})
        require(value.get("prompt_token_ids") == list(prompt_tokens)
                and value.get("consumed_tokens") is None and value.get("unconsumed_token_ids") is None
                and "no KV/recurrent checkpoint retained" in value.get("retention", "")
                and isinstance(receipt.get("request_id"), str) and receipt["request_id"]
                and type(receipt.get("submitted_tokens")) is int and receipt["submitted_tokens"] == len(prompt_tokens)
                and receipt.get("token_sha256") == token_hash(prompt_tokens),
                "vLLM generation must retain only exact logical tokens, not claim a checkpoint")
        cached_count(receipt.get("num_cached_tokens"), len(prompt_tokens))
        require(usage.get("prompt_tokens_details", {}).get("cached_tokens") == receipt.get("num_cached_tokens"),
                "Generation usage/cache receipt differs")
    return accepted


def score_check(value, request, identifier):
    require(value.get("id") == identifier
            and all(value.get(k) == request[k] for k in ("model", "primitive", "head_id")),
            "Decision request binding differs")
    require(all(value.get(k) is v for k, v in canonical.FLAGS.items())
            and value.get("head_sha256") == canonical.HEAD_SHA
            and value.get("head_manifest_sha256") == canonical.PLAN_SHA
            and value.get("head_execution_device") == "cuda:0"
            and value.get("calibration_id") == "uncalibrated:T1", "Wrong research head/T1 flags")
    scores = value.get("scores", [])
    ids = [c["id"] for c in request["candidates"]]
    require([s["id"] for s in scores] == ids, "Candidate order differs")
    logits = [s["logit"] for s in scores]
    raw = [s["raw_probability"] for s in scores]
    require(all(type(x) in (int, float) and math.isfinite(x) for x in logits + raw)
            and all(0 <= p <= 1 for p in raw) and math.isclose(sum(raw), 1., abs_tol=1e-6, rel_tol=0.)
            and raw == [s["calibrated_probability"] for s in scores], "Invalid/nonidentical T1 probabilities")
    exps = [math.exp(z - max(logits)) for z in logits]
    require(all(math.isclose(p, e / sum(exps), abs_tol=1e-6, rel_tol=1e-5)
                for p, e in zip(raw, exps)), "Raw T1 softmax differs")
    require(value.get("selected_id") == ids[max(range(len(ids)), key=logits.__getitem__)],
            "Selected ID is inconsistent with logits (no answer grading)")
    return raw


def decision_check(value, request, generation, accepted, suffix, tails, tier, labels, sources):
    query = accepted + suffix
    sequences = [query] + [query + tail for tail in tails]
    require(max(map(len, sequences)) <= tier <= 32768, "Actual submitted input exceeds tier/cap")
    require(value.get("prefix_id") == request["prefix_id"]
            and value.get("prefix_identity") == generation["prefix_identity"]
            and value.get("accepted_token_sha256") == token_hash(accepted)
            and value.get("query_token_sha256") == token_hash(query)
            and value.get("layout") == "accepted-ab-append-c", "Prefix/query token boundary differs")
    execution = value["execution"]
    require(type(execution.get("logical_prefix_tokens")) is int
            and execution["logical_prefix_tokens"] == len(accepted)
            and execution.get("feature_tap") == "final-mixer/post-collapse/before-lm-head",
            "Logical prefix/readout differs")
    if request["model"].startswith("native-"):
        consumed = generation["consumed_tokens"]
        expected = {"reused_tokens": consumed, "replayed_tokens": 0, "consumed_tokens": len(query),
                    "forwarded_tokens": len(query) - consumed + sum(map(len, tails))}
        require(execution.get("mode") == "checkpoint"
                and all(type(execution.get(k)) is int and execution[k] == v for k, v in expected.items())
                and value.get("resolved_unconsumed_tokens") == len(accepted) - consumed,
                "Native checkpoint consumption/replay evidence differs")
        batches = value.get("candidate_forward_batches", [])
        require(batches and all(type(n) is int and 1 <= n <= len(tails) for n in batches)
                and sum(batches) == len(tails), "Native candidate branch coverage differs")
        return {"mode": "checkpoint", **expected, "query_tokens": len(query),
                "maximum_branch_tokens": max(map(len, sequences))}
    require(execution.get("mode") == "full-token-replay"
            and type(execution.get("logical_query_tokens")) is int and execution["logical_query_tokens"] == len(query)
            and execution.get("checkpoint_retained") is False
            and all(execution.get(k) is None for k in ("consumed_tokens", "replayed_tokens", "reused_tokens")),
            "vLLM must honestly report logical full-token replay, not hybrid checkpoint reuse")
    receipts = execution.get("requests", [])
    require(len(receipts) == len(sequences), "Missing query/candidate engine receipts")
    seen, cache_counts = set(), []
    for index, (receipt, tokens) in enumerate(zip(receipts, sequences)):
        rid = receipt.get("request_id")
        require(isinstance(rid, str) and rid and rid not in seen
                and type(receipt.get("submitted_tokens")) is int and receipt["submitted_tokens"] == len(tokens)
                and receipt.get("token_sha256") == token_hash(tokens), "Engine request/token binding differs")
        seen.add(rid)
        readout_receipt(SimpleNamespace(fuzev_readout_receipt=receipt.get("fuzev_readout_receipt")),
                        tokens, rid, labels if index == 0 else (), sources)
        cache_counts.append(cached_count(receipt.get("num_cached_tokens"), len(tokens)))
    cached = sum(cache_counts) if None not in cache_counts else None
    cached_count(execution.get("num_cached_tokens"), sum(map(len, sequences)))
    require(type(execution.get("submitted_tokens")) is int and execution["submitted_tokens"] == sum(map(len, sequences))
            and execution.get("num_cached_tokens") == cached, "Aggregate submitted/cache counts differ")
    return {"mode": "full-token-replay", "query_tokens": len(query),
            "maximum_branch_tokens": max(map(len, sequences)), "num_cached_tokens": cached,
            "per_request_cached_tokens": cache_counts, "engine_request_ids": [r["request_id"] for r in receipts],
            "hybrid_checkpoint_proven": False}


def stable(reference, repeated):
    delta = max(abs(a - b) for a, b in zip(reference, repeated))
    require(len(reference) == len(repeated) and delta <= PROBABILITY_ATOL,
            "Repeat A probabilities exceed predeclared absolute tolerance")
    return delta


def run_suite(client, model, cases, tokenizer, health_validator, response_validator, journal):
    from fuzev_research.service_contracts import GenerationRequest, ServiceDecisionRequest

    suffixes, tails = fragments(tokenizer)
    require(suffixes["A"] != suffixes["B"], "Branch questions must have distinct actual token sequences")
    require(tuple(case["tier"] for case in cases) == TIERS, "Exactly four fixed tiers required")
    for case in cases:
        maximum = len(case["tokens"]) + MAX_NEW + max(map(len, suffixes.values())) + max(map(len, tails))
        require(maximum <= case["tier"] <= 32768, "Preflight input plus query/candidate margin exceeds tier/cap")
    labels = ()
    if not model.startswith("native-"):
        encoded_labels = [encode(tokenizer, " " + chr(65 + i)) for i in range(len(CANDIDATES))]
        require(all(len(x) == 1 for x in encoded_labels) and len(set(encoded_labels)) == len(CANDIDATES),
                "Actual tokenizer option labels are not single distinct tokens")
        labels = tuple(x[0] for x in encoded_labels)

    def exchange(method, path, body=None, identifier=None, expected=200):
        headers = {} if identifier is None else {"x-fuzev-request-id": identifier}
        response = client.request(method, path, json=body, headers=headers)
        value = response.json()
        journal({"method": method, "path": path, "request_id": identifier, "request": body,
                 "status": response.status_code, "response": value})
        require(response.status_code == expected, f"Unexpected HTTP status for {method} {path}")
        return value

    initial = exchange("GET", "/research-health")
    binding = health_validator(initial, None)
    sources = (None if model.startswith("native-") else
               readout_sources(initial["worker_readout"]["expected_loaded_sources"]))
    results, prefix_ids, engine_ids = [], set(), set()
    for case in cases:
        tier = case["tier"]
        generation_body = {"model": model, "prompt": case["text"], "max_tokens": MAX_NEW,
                           "temperature": 0., "top_p": 1., "top_k": 0, "seed": 0, "stream": False}
        gid = request_id(model, tier, "generation")
        GenerationRequest.model_validate(generation_body)
        generation = exchange("POST", "/v1/completions", generation_body, gid)
        prefix_id = generation.get("prefix_id")
        require(isinstance(prefix_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prefix_id),
                "Generation returned an invalid prefix ID")
        try:
            accepted = generation_check(generation, model, gid, case["tokens"])
            require(prefix_id not in prefix_ids, "Generation reused a previous prefix ID")
            prefix_ids.add(prefix_id)
            if sources is not None:
                rid = generation["execution"]["request_id"]
                require(rid not in engine_ids, "Generation reused a previous engine request ID")
                engine_ids.add(rid)
            probabilities, evidence = [], []
            for position, branch in enumerate(CONTRACT["sequence"]):
                body = {"model": model, "primitive": "choice", "head_id": "reference",
                        "prefix_id": prefix_id, "question": QUESTIONS[branch], "candidates": CANDIDATES}
                ServiceDecisionRequest.model_validate(body)
                identifier = request_id(model, tier, f"decision-{position}-{branch}")
                value = exchange("POST", "/v1/decisions", body, identifier)
                response_validator(value, body, identifier, binding)
                probabilities.append(score_check(value, body, identifier))
                proof = decision_check(value, body, generation, accepted,
                                       suffixes[branch], tails, tier, labels, sources)
                if sources is not None:
                    ids = set(proof["engine_request_ids"])
                    require(not ids & engine_ids, "Decision reused a previous engine request ID")
                    engine_ids.update(ids)
                evidence.append(proof)
            deltas = [stable(probabilities[0], probabilities[i]) for i in (1, 3)]
            if model.startswith('native-'):
                caching = {'status': 'checkpoint-reuse-observed', 'mechanism': 'native immutable checkpoint',
                           'evidence': 'exact consumed/reused counts, zero replay and repeated-branch stability'}
            else:
                hit = any(proof['num_cached_tokens'] is not None and proof['num_cached_tokens'] > 0
                          for proof in evidence)
                caching = {'status': 'cache-hit-observed' if hit else 'inconclusive-no-hit-evidence',
                           'mechanism': 'engine-owned prefix cache; not a hybrid checkpoint',
                           'evidence': 'per-request cached-token receipts; zero/null is not a cache-reuse pass'}
            results.append({"tier": tier, "fixture_repetitions": case["repetitions"],
                            "fixture_text_sha256": case["text_sha256"], "prompt_tokens": len(case["tokens"]),
                            "prompt_token_sha256": token_hash(case["tokens"]), "accepted_tokens": len(accepted),
                            "accepted_token_sha256": token_hash(accepted), "repeat_A_absolute_deltas": deltas,
                            "execution": evidence, "prefix_cache": caching, "correctness_passed": True})
        finally:
            released = exchange("DELETE", "/v1/prefixes/" + prefix_id,
                                identifier=request_id(model, tier, "delete"))
            require(released.get("id") == prefix_id and released.get("released") is True,
                    "Prefix deletion was not acknowledged")
        exchange("POST", "/v1/decisions", body, request_id(model, tier, "deleted-404"), expected=404)
        results[-1]["deleted_prefix_404"] = True
    health_validator(exchange("GET", "/research-health"), binding)
    return {"contract": CONTRACT, "model": model, "status": "passed", "tiers": results,
            "provenance": binding, "deleted_prefix_404": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-inference", action="store_true")
    parser.add_argument("--model", choices=MODELS, default="native-4.05")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(args.allow_inference, "Explicit --allow-inference required; run only after the 5075 panel completes")
    url = urlsplit(args.base_url)
    require(url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
            and url.port is not None and url.path in ("", "/") and not url.query and not url.fragment
            and url.username is None and url.password is None, "Use an explicit loopback host and port")
    output = (args.output or REPORT / ("engine-correctness-" + args.model + "-v1")).resolve()
    require(output.parent == REPORT.resolve(), "Output must be an exclusive child of the cross-cell report directory")
    canonical.verify({str(canonical.PLAN): canonical.PLAN_SHA, str(canonical.QUALITY): canonical.QUALITY_SHA})
    identity_path = Path(canonical.read(canonical.PLAN)["original_feature_identity"])
    original = canonical.read(identity_path)
    acquisition = canonical.read(canonical.ACQUISITION)["models"]
    key = "native-4.05" if args.model == "vllm-exl3-4.05" else args.model
    snapshot = Path(acquisition[key]["local_snapshot"])
    reference_snapshot = Path(acquisition["native-4.05"]["local_snapshot"])
    from transformers import AutoTokenizer
    import httpx
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    reference = AutoTokenizer.from_pretrained(reference_snapshot, local_files_only=True, trust_remote_code=False)
    cases = fixtures(reference, tokenizer)
    canonical.MODEL = args.model
    if args.model.startswith("native-"):
        health = lambda value, binding: canonical.health_check(value, original, binding)
        validate = lambda value, body, rid, binding: canonical.response_check(value, body, rid)
    else:
        import kev_vllm_e2e_client as vllm
        from fuzev_research.vllm_service import VllmDecisionResponse
        vllm.MODEL = args.model
        health = lambda value, binding: vllm.health_check(value, original, binding)

        def validate(value, body, rid, binding):
            VllmDecisionResponse.model_validate(value)
            require(value.get("provenance") == binding, "Actual vLLM serving provenance changed")

    pins = {str(canonical.PLAN): canonical.PLAN_SHA, str(canonical.QUALITY): canonical.QUALITY_SHA,
            str(identity_path): canonical.sha(identity_path),
            str(canonical.ACQUISITION): canonical.sha(canonical.ACQUISITION)}
    for source in (__file__, canonical.__file__, HERE / "kev_vllm_e2e_client.py",
                   HERE.parent / "fuzev_research/encoding.py", HERE.parent / "fuzev_research/vllm_receipts.py"):
        pins[str(source)] = canonical.sha(source)
    for directory in {snapshot, reference_snapshot}:
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"):
            path = directory / name
            if path.is_file():
                pins[str(path)] = canonical.sha(path)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    canonical.private_json(output / "contract.json", {**CONTRACT, "model": args.model, "input_source_pins": pins})
    try:
        with os.fdopen(os.open(output / "http.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="ascii") as log:

            def journal(value):
                log.write(canonical.encoded(value).decode("ascii"))
                log.flush()

            with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=3600.,
                              transport=httpx.HTTPTransport(retries=0), trust_env=False) as client:
                result = run_suite(client, args.model, cases, tokenizer, health, validate, journal)
        canonical.verify(pins)
        canonical.private_json(output / "result.json", result)
    except Exception as error:
        canonical.private_json(output / "result.json", {"contract": CONTRACT, "model": args.model,
                               "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    print(f"Engine correctness only: passed; private report {output}")


if __name__ == "__main__":
    main()
