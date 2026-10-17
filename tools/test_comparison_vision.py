"""CPU behavior tests for the shared image comparison runner."""

import base64
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import comparison_vision as comparison


def record(identifier="sample", *, context=None, target=None):
    return {"id": identifier, "source": "fixture", "primitive": "choice", "modality": "image",
            "context": [] if context is None else context, "question": "Which caption matches?",
            "candidates": [{"id": "b", "text": "Blue."}, {"id": "r", "text": "Red."}],
            "target": [1., 0.] if target is None else target, "role": "test",
            "image_path": "fixture.png", "group": "image:" + identifier,
            "adaptation": "caption"}


def response(probabilities=(.8, .2), *, calibrated=None):
    selected = ("b", "r")[max(range(len(probabilities)), key=probabilities.__getitem__)]
    return {"calibration_id": "raw" if calibrated is None else "calibrated", "selected_id": selected,
            "scores": [{"id": identifier, "logit": math.log(probability),
                        "raw_probability": probability,
                        "calibrated_probability": calibrated[index] if calibrated else probability}
                       for index, (identifier, probability) in enumerate(zip(("b", "r"), probabilities))]}


def measured(source=None, reply=None):
    source = record() if source is None else source
    return {"arm": "fixture", "set": "sample", "track": "fixture/choice/image/caption",
            "index": 0, "record": source, "status_code": 200, "error": None,
            "response": response() if reply is None else reply}


class ComparisonVisionTests(unittest.TestCase):
    def test_binary_answer_order_counts_include_failed_requests(self):
        first = measured(record("first"))
        second = measured(record("second", target=[0., 1.]), response((.2, .8)))
        wrong = measured(record("wrong", target=[0., 1.]))
        failed = {**measured(record("failed")), "status_code": 413, "error": "HTTP 413"}
        result = next(row for row in comparison.summarize_rows([first, second, wrong, failed])
                      if row["track"] == "all")
        self.assertEqual(result["gold_first"], 2)
        self.assertEqual(result["gold_second"], 2)
        self.assertEqual(result["first_pick"], 2)
        self.assertEqual(result["gold_second_scored"], 2)
        self.assertEqual(result["gold_second_correct"], 1)

    def test_original_mix_order_and_jsonl_views_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = {"row": record("first"), "task_track": "original/track",
                        "request": {"messages": [], "question": "Which caption matches?"}}
            second = {"row": record("second")}
            text = {"row": {**record("text"), "modality": "text"}}
            mix, other = directory / "mix.json", directory / "other.jsonl"
            mix.write_text(json.dumps({"rows": [original, text, second]}))
            other.write_text(json.dumps(record("other")) + "\n")
            cases = comparison.load_cases({"eval-mix": mix, "vl-choice": other})
        self.assertEqual([case["record"]["id"] for case in cases], ["first", "second", "other"])
        self.assertEqual([case["source"]["position"] for case in cases], [0, 2, 0])
        self.assertEqual(cases[0]["track"], "original/track")
        self.assertEqual(cases[0]["original_request"], original["request"])
        self.assertEqual(cases[0]["record"], original["row"])

    def test_public_image_transport_keeps_questions_options_and_context(self):
        row = record(context=[{"role": "user", "content": "A reference."}])
        row["additional_context"] = "Use visible color."
        before = deepcopy(row)
        with patch.object(comparison, "image_url", return_value="data:image/png;base64,AA=="):
            request = comparison.request_for({"record": row}, "served-model")
        self.assertEqual(request, {"primitive": "choice", "question": row["question"],
            "candidates": row["candidates"], "messages": row["context"],
            "additional_context": "Use visible color.", "image": "data:image/png;base64,AA==",
            "model": "served-model"})
        self.assertEqual(row, before)
        with patch.object(comparison, "image_url", return_value="image"):
            request = comparison.request_for({"record": record()})
        self.assertEqual(request["messages"], [{"role": "user", "content": "Consider the supplied image."}])

    def test_every_http_attempt_including_errors_is_written(self):
        replies = iter([SimpleNamespace(status_code=400, text='{"detail":"image too large"}',
                                        json=lambda: {"detail": "image too large"}),
                        RuntimeError("connection lost"),
                        SimpleNamespace(status_code=200, text=json.dumps(response()), json=response)])

        def post(path, json):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        cases = [{"set": "sample", "track": "caption", "record": record(str(index)),
                  "source": {"path": "input", "position": index}} for index in range(3)]
        output = io.StringIO()
        with patch.object(comparison, "image_url", return_value="image"):
            counts = comparison.run_cases(cases, SimpleNamespace(post=post), output, arm="example")
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(counts, {"requests": 3, "errors": 2})
        self.assertEqual([row["record"]["id"] for row in rows], ["0", "1", "2"])
        self.assertEqual([row["status_code"] for row in rows], [400, None, 200])
        self.assertIn("image too large", rows[0]["error"])
        self.assertIn("connection lost", rows[1]["error"])
        self.assertEqual(rows[2]["response"], response())

    def test_lossless_webp_preserves_native_rgb_pixels_and_shared_bytes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "canonical.png"
            image = Image.new("RGB", (13, 17))
            image.putdata([((index * 17) % 256, (index * 31) % 256, (index * 43) % 256)
                           for index in range(13 * 17)])
            image.save(source)
            cached = comparison.lossless_webp(source, directory / "shared-cache")
            first = comparison.image_url(cached)
            second = comparison.image_url(comparison.lossless_webp(source, directory / "shared-cache"))
            self.assertEqual(first, second)
            self.assertTrue(first.startswith("data:image/webp;base64,"))
            with Image.open(io.BytesIO(base64.b64decode(first.split(",", 1)[1]))) as decoded:
                self.assertEqual(decoded.size, image.size)
                self.assertEqual(decoded.convert("RGB").tobytes(), image.tobytes())

    def test_retry_writes_separate_attempts_and_preserves_nonimage_request_fields(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, raw = directory / "canonical.png", directory / "original.jsonl"
            Image.new("RGB", (3, 2), (12, 34, 56)).save(source)
            rows = []
            for index, status in enumerate((200, 413, 500, 413)):
                row = measured({**record(str(index)), "image_path": str(source)})
                row.update(index=index, status_code=status, error=None if status == 200 else f"HTTP {status}")
                row["request"] = comparison.request_for(row, "shipped-model")
                rows.append(row)
            original = "".join(json.dumps(row) + "\n" for row in rows)
            raw.write_text(original)
            requests = []
            success_text = json.dumps(response())

            def post(path, json):
                requests.append(json)
                status = 200 if len(requests) == 1 else 413
                return SimpleNamespace(status_code=status, text=success_text if status == 200
                                       else '{"detail":"still too large"}')

            output = io.StringIO()
            counts = comparison.retry_cases(raw, SimpleNamespace(post=post), output,
                                            image_cache=directory / "cache")
            attempts = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(counts, {"requests": 2, "errors": 1})
            self.assertEqual([row["index"] for row in attempts], [1, 3])
            self.assertEqual([row["status_code"] for row in attempts], [200, 413])
            self.assertEqual(requests[0]["image"], requests[1]["image"])
            for attempt in attempts:
                original_row = rows[attempt["index"]]
                self.assertEqual(attempt["retry_of"], {"path": raw.name,
                    "index": original_row["index"], "id": original_row["record"]["id"], "arm": "fixture"})
                self.assertEqual({k: v for k, v in attempt["request"].items() if k != "image"},
                                 {k: v for k, v in original_row["request"].items() if k != "image"})
            self.assertEqual(raw.read_text(), original)

    def test_retry_completed_summary_replaces_success_once_and_keeps_initial_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw, retries = directory / "original.jsonl", directory / "retry.jsonl"
            rows = [measured(record(str(index))) for index in range(4)]
            for index, status in enumerate((200, 413, 413, 500)):
                rows[index].update(index=index, status_code=status,
                                   error=None if status == 200 else f"HTTP {status}")
                rows[index]["request"] = {"image": "large body discarded by scorer"}
            good = {**rows[1], "status_code": 200, "error": None,
                    "retry_of": {"path": str(raw), "index": 1, "id": "1", "arm": "fixture"}}
            failed = {**rows[2], "retry_of": {"path": str(raw), "index": 2, "id": "2", "arm": "fixture"}}
            original = "".join(json.dumps(row) + "\n" for row in rows)
            raw.write_text(original)
            retries.write_text("".join(json.dumps(row) + "\n" for row in (good, failed, good)))
            result = comparison.score_files([raw], retries=[retries])
            initial = next(row for row in result["initial_pass"] if row["track"] == "all")
            completed = next(row for row in result["retry_completed"] if row["track"] == "all")
            self.assertEqual(initial["requests"], 4)
            self.assertEqual(initial["scored"], 1)
            self.assertEqual([row["index"] for row in initial["errors"]], [1, 2, 3])
            self.assertEqual(completed["requests"], 4)
            self.assertEqual(completed["scored"], 2)
            self.assertEqual([row["index"] for row in completed["errors"]], [2, 3])
            self.assertAlmostEqual(completed["metrics"]["nll"], -math.log(.8))
            self.assertEqual(completed["metrics"]["accuracy"], 1.)
            self.assertEqual(len(result["retry_attempts"]), 3)
            self.assertEqual(raw.read_text(), original)

    def test_metric_golden_and_calibrated_probability_selection(self):
        raw = comparison.score_row(measured(), "head")
        calibrated = comparison.score_row(measured(reply=response(calibrated=(.9, .1))), "head")
        self.assertAlmostEqual(raw["nll"], -math.log(.8))
        self.assertAlmostEqual(raw["brier"], .08)
        self.assertEqual(raw["accuracy"], 1.)
        self.assertAlmostEqual(calibrated["nll"], -math.log(.9))
        self.assertAlmostEqual(calibrated["brier"], .02)

    def test_candidate_ids_align_even_when_response_order_changes(self):
        reply = response()
        reply["scores"].reverse()
        self.assertEqual(comparison.score_row(measured(reply=reply), "head"),
                         comparison.score_row(measured(), "head"))

    def test_accuracy_uses_the_api_selected_candidate(self):
        reply = response()
        reply["selected_id"] = "r"
        result = comparison.score_row(measured(reply=reply), "head")
        self.assertEqual(result["accuracy"], 0.)
        self.assertEqual(result["expected_accuracy"], 0.)
        self.assertAlmostEqual(result["confidence"], .2)
        self.assertAlmostEqual(result["nll"], -math.log(.8))

    def test_ties_and_failed_rows_keep_their_metric_denominators(self):
        tied = measured(record("tie", target=[.5, .5]))
        tied["index"] = 1
        failed = {**measured(), "index": 2, "status_code": 500, "error": "server error"}
        summaries = comparison.summarize_rows([measured(), tied, failed])
        summary = next(item for item in summaries if item["track"] == "all")
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["scored"], 2)
        self.assertEqual(summary["errors"], [{"index": 2, "id": "sample", "error": "ValueError: server error"}])
        self.assertEqual(summary["metrics"]["accuracy"], 1.)
        self.assertEqual(summary["metrics"]["unique_gold_records"], 1)
        self.assertEqual(summary["metrics"]["expected_accuracy"], .75)
        self.assertAlmostEqual(summary["metrics"]["ece"], .05)

    def test_backbone_is_scored_from_its_own_probabilities(self):
        row = measured()
        row["response"]["backbone_control"] = response((.25, .75))
        result = comparison.score_row(row, "backbone")
        self.assertEqual(result["accuracy"], 0.)
        self.assertAlmostEqual(result["nll"], -math.log(.25))



if __name__ == "__main__":
    unittest.main()
