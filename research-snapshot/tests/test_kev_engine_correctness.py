"""CPU-only synthetic HTTP/token fixtures; no live service or model execution."""

import copy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kev_engine_correctness.py"
spec = importlib.util.spec_from_file_location("kev_engine_correctness_test_module", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
from fuzev_research.vllm_receipts import READOUT_SOURCE_FILES


class Tokenizer:
    @staticmethod
    @lru_cache(maxsize=None)
    def token(word):
        return int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return [self.token(word) for word in re.findall(r"<\|[^>]+\|>|[A-Za-z]+|\d+|[^\s]", text)]


SOURCES = {name: "a" * 64 for name in READOUT_SOURCE_FILES}
BINDING = {"test_fixture_only": True}


class Client:
    def __init__(self, model, tokenizer, mutate=None, cached=7, completion_count=2):
        self.model, self.tokenizer, self.mutate = model, tokenizer, mutate
        self.cached, self.completion_count = cached, completion_count
        self.calls, self.prefixes, self.generated, self.decisions = [], {}, 0, 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def request(self, method, path, *, json, headers):
        self.calls.append((method, path, copy.deepcopy(json), headers))
        identifier = headers.get("x-fuzev-request-id")
        if method == "GET":
            value = {"model": self.model, "provenance": BINDING,
                     "worker_readout": {"expected_loaded_sources": SOURCES}}
        elif method == "DELETE":
            pid = path.rsplit("/", 1)[1]
            value = {"id": pid, "released": self.prefixes.pop(pid, None) is not None}
        elif path == "/v1/completions":
            self.generated += 1
            prompt = probe.encode(self.tokenizer, json["prompt"])
            completion = [71, 72][:self.completion_count]
            accepted = prompt + tuple(completion)
            pid = "prefix_" + str(self.generated)
            value = {"id": identifier, "model": self.model, "object": "text_completion",
                     "choices": [{"index": 0, "finish_reason": "length", "text": ""}],
                     "prefix_id": pid, "prefix_identity": "fixture-identity",
                     "completion_token_ids": completion, "accepted_token_sha256": probe.token_hash(accepted),
                     "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                               "total_tokens": len(accepted)}}
            if self.model.startswith("native-"):
                value.update(consumed_tokens=len(accepted) - 1, unconsumed_token_ids=completion[-1:])
            else:
                value["usage"]["prompt_tokens_details"] = {"cached_tokens": self.cached}
                value.update(prompt_token_ids=list(prompt), consumed_tokens=None, unconsumed_token_ids=None,
                             retention="exact API-accepted tokens only; no KV/recurrent checkpoint retained",
                             execution={"request_id": "generation-engine-" + str(self.generated), "submitted_tokens": len(prompt),
                                        "token_sha256": probe.token_hash(prompt), "num_cached_tokens": self.cached})
            self.prefixes[pid] = (accepted, copy.deepcopy(value))
        else:
            if json["prefix_id"] not in self.prefixes:
                return SimpleNamespace(status_code=404, json=lambda: {"detail": "Unknown prefix"})
            self.decisions += 1
            accepted, generation = self.prefixes[json["prefix_id"]]
            query = accepted + probe.encode(self.tokenizer, probe.query_suffix(json["question"], json["candidates"]))
            tails = [probe.encode(self.tokenizer, probe.candidate_suffix(c)) for c in json["candidates"]]
            raw = [.4, .6] if json["question"] == probe.QUESTIONS["A"] else [.7, .3]
            value = {"id": identifier, "model": self.model, "primitive": "choice", "head_id": "reference",
                     "head_sha256": probe.canonical.HEAD_SHA, "head_manifest_sha256": probe.canonical.PLAN_SHA,
                     "head_execution_device": "cuda:0", "calibration_id": "uncalibrated:T1",
                     "prefix_id": json["prefix_id"], "prefix_identity": generation["prefix_identity"],
                     "accepted_token_sha256": probe.token_hash(accepted), "query_token_sha256": probe.token_hash(query),
                     "layout": "accepted-ab-append-c", "selected_id": "beta" if raw[1] > raw[0] else "alpha",
                     "scores": [{"id": c["id"], "logit": math.log(p), "raw_probability": p,
                                 "calibrated_probability": p} for c, p in zip(json["candidates"], raw)],
                     "provenance": BINDING, **probe.canonical.FLAGS}
            execution = {"logical_prefix_tokens": len(accepted),
                         "feature_tap": "final-mixer/post-collapse/before-lm-head"}
            if self.model.startswith("native-"):
                consumed = generation["consumed_tokens"]
                execution.update(mode="checkpoint", reused_tokens=consumed, replayed_tokens=0,
                                 consumed_tokens=len(query), forwarded_tokens=len(query) - consumed + sum(map(len, tails)))
                value.update(resolved_unconsumed_tokens=len(accepted) - consumed, candidate_forward_batches=[2])
            else:
                receipts = []
                for index, tokens in enumerate([query] + [query + tail for tail in tails]):
                    rid = f"engine_{self.decisions}_{index}"
                    labels = [probe.encode(self.tokenizer, " A")[0], probe.encode(self.tokenizer, " B")[0]] if index == 0 else []
                    proof = {"schema_version": 1, "worker_request_id": "worker_" + rid, "request_id": rid,
                             "prompt_tokens": len(tokens), "prompt_token_sha256": probe.token_hash(tokens),
                             "feature_readout_position": len(tokens) - 1, "feature_width": 2560,
                             "baseline_token_ids": labels, "lm_projection_calls": 1 if labels else 0,
                             "generated_token_count": 0, "worker_loaded_sources": SOURCES}
                    receipts.append({"request_id": rid, "submitted_tokens": len(tokens),
                                     "token_sha256": probe.token_hash(tokens), "num_cached_tokens": self.cached,
                                     "fuzev_readout_receipt": proof})
                execution.update(mode="full-token-replay", logical_query_tokens=len(query),
                                 submitted_tokens=sum(r["submitted_tokens"] for r in receipts),
                                 num_cached_tokens=None if self.cached is None else 3 * self.cached,
                                 consumed_tokens=None, reused_tokens=None, replayed_tokens=None,
                                 checkpoint_retained=False, requests=receipts)
            value["execution"] = execution
            if self.mutate:
                self.mutate(value, self.decisions)
        return SimpleNamespace(status_code=200, json=lambda: value)


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer()
        cls.cases = probe.fixtures(cls.tokenizer, cls.tokenizer)

    def suite(self, model="native-4.05", mutate=None, cached=7, completion_count=2):
        client = Client(model, self.tokenizer, mutate, cached, completion_count)
        records = []

        def health(value, binding):
            self.assertEqual(value["model"], model)
            if binding is not None:
                self.assertEqual(binding, value["provenance"])
            return value["provenance"]

        def response(value, body, rid, binding):
            if model.startswith("native-"):
                probe.canonical.response_check(value, body, rid)
            else:
                self.assertEqual(value["provenance"], binding)

        result = probe.run_suite(client, model, self.cases, self.tokenizer, health, response, records.append)
        return client, result, records

    def test_native_four_tiers_budget_greedy_and_actual_checkpoint(self):
        client, result, records = self.suite()
        decisions = [c for c in client.calls if c[1] == "/v1/decisions"]
        generations = [c for c in client.calls if c[1] == "/v1/completions"]
        self.assertEqual((len(decisions), len(generations), len(records)), (20, 4, 30))
        self.assertEqual(sum(r["status"] == 404 for r in records), 4)
        self.assertEqual(client.prefixes, {})
        self.assertEqual(result["status"], "passed")
        identifiers = [c[3]["x-fuzev-request-id"] for c in client.calls if c[3]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for call in generations:
            self.assertEqual((call[2]["max_tokens"], call[2]["temperature"], call[2]["top_k"]), (2, 0., 0))
        for tier in result["tiers"]:
            self.assertEqual(tier["repeat_A_absolute_deltas"], [0., 0.])
            self.assertEqual(tier["prefix_cache"]["status"], "checkpoint-reuse-observed")
            self.assertEqual(tier["prefix_cache"]["mechanism"], "native immutable checkpoint")
            self.assertTrue(tier["deleted_prefix_404"])
            self.assertTrue(all(e["replayed_tokens"] == 0 and e["reused_tokens"] > 0 for e in tier["execution"]))
            self.assertTrue(all(e["maximum_branch_tokens"] <= tier["tier"] <= 32768 for e in tier["execution"]))
        self.assertEqual(records[-2]["status"], 404)
        self.assertFalse(result["contract"]["accuracy"])
        self.assertNotIn("gold", result)

    def test_all_six_cells_share_fixed_prefix_only_fixture_contract(self):
        self.assertEqual(set(probe.MODELS), {"native-3.05", "native-4.05", "native-5.05",
                                            "vllm-nvidia", "vllm-radixark", "vllm-exl3-4.05"})
        for model in probe.MODELS:
            with self.subTest(model=model):
                client, result, records = self.suite(model)
                self.assertEqual(result["model"], model)
                self.assertEqual([t["fixture_text_sha256"] for t in result["tiers"]],
                                 [c["text_sha256"] for c in self.cases])
                self.assertEqual(sum(r["status"] == 404 for r in records), 4)
                successful = [r for r in records if r["path"] == "/v1/decisions" and r["status"] == 200]
                self.assertEqual(len(successful), 16)
                self.assertTrue(all(r["request"]["model"] == model and "prefix_id" in r["request"]
                                    and "messages" not in r["request"] for r in successful))
                self.assertEqual(client.prefixes, {})

    def test_vllm_full_token_replay_with_available_or_unavailable_cache_counts(self):
        for cached in (7, None, 0):
            with self.subTest(cached=cached):
                _, result, _ = self.suite("vllm-exl3-4.05", cached=cached)
                execution = result["tiers"][-1]["execution"][-1]
                self.assertEqual(execution["num_cached_tokens"], None if cached is None else 3 * cached)
                self.assertFalse(execution["hybrid_checkpoint_proven"])
                expected = "cache-hit-observed" if cached == 7 else "inconclusive-no-hit-evidence"
                for tier in result["tiers"]:
                    self.assertEqual(tier["prefix_cache"]["status"], expected)
                    self.assertEqual(tier["prefix_cache"]["mechanism"],
                                     "engine-owned prefix cache; not a hybrid checkpoint")

    def test_early_eos_boundary_keeps_actual_accepted_ids(self):
        _, result, _ = self.suite(completion_count=1)
        self.assertTrue(all(t["accepted_tokens"] == t["prompt_tokens"] + 1 for t in result["tiers"]))
        _, result, _ = self.suite("vllm-radixark", completion_count=0)
        self.assertTrue(all(t["accepted_tokens"] == t["prompt_tokens"] for t in result["tiers"]))

    def test_repeated_A_drift_is_failure_and_prefix_is_deleted(self):
        def corrupt(value, number):
            if number == 4:
                for score, p in zip(value["scores"], [.4001, .5999]):
                    score.update(logit=math.log(p), raw_probability=p, calibrated_probability=p)

        client = Client("native-4.05", self.tokenizer, corrupt)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            probe.run_suite(client, client.model, self.cases, self.tokenizer,
                            lambda h, b: BINDING, lambda *a: None, lambda *a: None)
        self.assertEqual(client.prefixes, {})

    def test_receipt_corruption_rejected(self):
        corruptions = {
            "accepted hash": lambda v: v.update(accepted_token_sha256="b" * 64),
            "branch query hash": lambda v: v.update(query_token_sha256="b" * 64),
            "native replay": lambda v: v["execution"].update(replayed_tokens=1),
            "native consumed": lambda v: v["execution"].update(consumed_tokens=0),
            "native unresolved": lambda v: v.update(resolved_unconsumed_tokens=0),
            "candidate order": lambda v: v["scores"].reverse(),
            "nonfinite": lambda v: v["scores"][0].update(logit=float("nan")),
            "raw calibration": lambda v: v["scores"][0].update(calibrated_probability=.5),
            "request ID": lambda v: v.update(id="incorrect"),
            "head provenance": lambda v: v.update(head_sha256="b" * 64),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.suite(mutate=lambda v, n: corrupt(v))

    def test_vllm_worker_proof_and_no_checkpoint_claim(self):
        corruptions = [lambda v: v["execution"].update(checkpoint_retained=True),
                       lambda v: v["execution"].update(reused_tokens=1),
                       lambda v: v["execution"]["requests"][0].update(token_sha256="b" * 64),
                       lambda v: v["execution"]["requests"][0]["fuzev_readout_receipt"].update(lm_projection_calls=0),
                       lambda v: v["execution"]["requests"][1]["fuzev_readout_receipt"].update(worker_loaded_sources={}),
                       lambda v: v["execution"].update(num_cached_tokens=-1),
                       lambda v: v["execution"].update(num_cached_tokens=21.0)]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.suite("vllm-nvidia", mutate=lambda v, n: corrupt(v))

    def test_generation_boundaries_hashes_and_no_decode(self):
        client = Client("native-4.05", self.tokenizer)
        case = self.cases[0]
        value = client.request("POST", "/v1/completions", json={"prompt": case["text"]},
                               headers={"x-fuzev-request-id": "gen"}).json()
        accepted = probe.generation_check(value, client.model, "gen", case["tokens"])
        self.assertEqual(accepted[-2:], (71, 72))
        for key, bad in (("consumed_tokens", len(accepted)), ("completion_token_ids", [1, 2, 3]),
                         ("accepted_token_sha256", "b" * 64), ("unconsumed_token_ids", []), ("choices", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.generation_check({**value, key: bad}, client.model, "gen", case["tokens"])

    def test_same_fixed_strings_across_cells_no_adaptive_shortening(self):
        other = probe.fixtures(Tokenizer(), Tokenizer())
        self.assertEqual([x["text_sha256"] for x in self.cases], [x["text_sha256"] for x in other])
        doubled = SimpleNamespace(encode=lambda text, **kw: self.tokenizer.encode(text, **kw) * 2)
        with self.assertRaisesRegex(ValueError, "no shortening"):
            probe.fixtures(self.tokenizer, doubled)

    def test_predeclared_tolerance_and_cache_scalar_types(self):
        self.assertLessEqual(probe.stable([.5, .5], [.500001, .499999]), probe.PROBABILITY_ATOL)
        with self.assertRaises(ValueError):
            probe.stable([.5, .5], [.50001, .49999])
        for value in (True, -1, 100, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.cached_count(value, 20)

    def test_margin_preflight_rejects_oversize_before_http(self):
        cases = copy.deepcopy(self.cases)
        cases[-1]["tokens"] = (0,) * 32768
        client = Client("native-4.05", self.tokenizer)
        with self.assertRaisesRegex(ValueError, "margin"):
            probe.run_suite(client, client.model, cases, self.tokenizer,
                            lambda *a: None, lambda *a: None, lambda *a: None)
        self.assertEqual(client.calls, [])

    def test_cli_requires_explicit_permission_before_import_or_http(self):
        with self.assertRaisesRegex(ValueError, "allow-inference"):
            probe.main([])

    def test_private_exclusive_outputs_with_only_mocked_runtime(self):
        client = Client("native-4.05", self.tokenizer)
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp)
            identity = report / "identity.json"
            def read(path):
                if Path(path) == probe.canonical.PLAN:
                    return {"original_feature_identity": str(identity)}
                if Path(path) == probe.canonical.ACQUISITION:
                    return {"models": {"native-4.05": {"local_snapshot": str(report / "snapshot")}}}
                return {"test_fixture_only": True}

            modules = {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: self.tokenizer)),
                       "httpx": SimpleNamespace(Client=lambda **kw: client, HTTPTransport=lambda **kw: None)}
            with patch.object(probe, "REPORT", report), patch.dict(sys.modules, modules), \
                 patch.object(probe.canonical, "verify"), patch.object(probe.canonical, "read", side_effect=read), \
                 patch.object(probe.canonical, "sha", return_value="a" * 64), \
                 patch.object(probe.canonical, "health_check", side_effect=lambda h, o, b: BINDING), patch("builtins.print"):
                probe.main(["--allow-inference"])
                output = report / "engine-correctness-native-4.05-v1"
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "passed")
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)
                for path in output.iterdir():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                before = (output / "result.json").read_bytes()
                calls = len(client.calls)
                with self.assertRaises(FileExistsError):
                    probe.main(["--allow-inference"])
                self.assertEqual((output / "result.json").read_bytes(), before)
                self.assertEqual(len(client.calls), calls)
                with self.assertRaisesRegex(ValueError, "exclusive child"):
                    probe.main(["--allow-inference", "--output", str(report.parent / "outside")])


if __name__ == "__main__":
    unittest.main()
