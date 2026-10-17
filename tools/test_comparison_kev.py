"""CPU checks of the comparison runner's saved outputs and published metrics."""

import json
from pathlib import Path
import tempfile
import unittest

from comparison_kev import run_records, score_results


def record(rid, source, labels):
    return {"_meta": {"id": rid, "group_id": rid, "source": source, "variant": "clean"},
            "state": rid,
            "questions": {f"q{i}": {"type": "noul", "label": label, "src": source}
                          for i, label in enumerate(labels)}}


def prediction(record, answers):
    return {"probabilities": {qid: {"true": 0.9 if answer else 0.1,
                                    "false": 0.1 if answer else 0.9}
                              for qid, answer in zip(record["questions"], answers)},
            "latency_ms": 1, "input_tokens": 1}


class ComparisonKevTests(unittest.TestCase):
    def test_case_exact_and_question_accuracy_golden(self):
        records = [record("a", "bfcl", [True, True]),
                   record("b", "bfcl", [True, True]),
                   record("c", "ordinary", [True])]
        answers = {"a": [True, False], "b": [True, True], "c": [True]}
        manifest = {"datasets": {"bfcl": {"metric": "case_exact", "area": "tools"},
                                 "ordinary": {"metric": "accuracy", "area": "tools"}},
                    "areas": {"tools": {"label": "Tools", "datasets": ["bfcl", "ordinary"]}}}
        with tempfile.TemporaryDirectory() as directory:
            rows, coverage, interrupted = run_records(
                records, lambda r: prediction(r, answers[r["_meta"]["id"]]),
                directory, system="test", split="development")
            result = score_results(records, {"test": rows}, manifest)
            system = result["systems"]["test"]
            self.assertFalse(interrupted)
            self.assertEqual(coverage["answered_questions"], 5)
            self.assertEqual(system["datasets"]["bfcl"]["score"], 0.5)
            self.assertEqual(system["datasets"]["bfcl"]["question_accuracy"], 0.75)
            self.assertEqual(system["overall"]["question_accuracy"], 0.8)
            self.assertAlmostEqual(system["overall"]["index"], 200 / 3)
            self.assertEqual(result["uncertainty"]["index"]["test"], [50.0, 100.0])
            paired = score_results(records, {"reference": [], "candidate": rows}, manifest)
            self.assertEqual(paired["uncertainty"]["difference_from"], "reference")
            self.assertEqual(paired["uncertainty"]["dataset_difference"]["ordinary"]["candidate"], [100., 100.])
            self.assertEqual(paired["uncertainty"]["dataset_difference"]["bfcl"]["candidate"], [0., 100.])

    def test_resume_preserves_error_and_completes_only_unattempted_records(self):
        records = [record(str(i), "ordinary", [True]) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            def first(r):
                if r["_meta"]["id"] == "1":
                    raise TimeoutError("response lost")
                return prediction(r, [True])

            _, coverage, interrupted = run_records(records, first, directory,
                                                   system="test", split="test")
            self.assertTrue(interrupted)
            self.assertEqual(coverage["errors"][0]["error"], "response lost")
            self.assertEqual(coverage["unattempted_records"], ["2"])
            seen = []

            def resume(r):
                seen.append(r["_meta"]["id"])
                return prediction(r, [True])

            rows, coverage, interrupted = run_records(records, resume, directory,
                                                      system="test", split="test")
            self.assertFalse(interrupted)
            self.assertEqual(seen, ["2"])
            self.assertEqual([row["id"] for row in rows], ["0", "2"])
            self.assertEqual(coverage["attempted_records"], 3)
            self.assertEqual(len(coverage["errors"]), 1)
            self.assertEqual(json.loads((Path(directory) / "rows.json").read_text()), rows)
            run_records(records, resume, directory, system="test", split="test")
            self.assertEqual(seen, ["2"])

    def test_interrupted_request_is_reported_without_replay(self):
        records = [record("0", "ordinary", [True]), record("1", "ordinary", [True])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text(json.dumps({"event": "attempt", "id": "0"}) + "\n")
            seen = []

            def predictor(r):
                seen.append(r["_meta"]["id"])
                return prediction(r, [True])

            _, coverage, _ = run_records(records, predictor, directory,
                                          system="test", split="test")
            self.assertEqual(seen, ["1"])
            self.assertEqual(coverage["unresolved_records"], ["0"])

    def test_missing_response_counts_wrong(self):
        records = [record("0", "ordinary", [True])]
        manifest = {"datasets": {"ordinary": {"metric": "accuracy", "area": "a"}},
                    "areas": {"a": {"label": "Area", "datasets": ["ordinary"]}}}
        result = score_results(records, {"test": []}, manifest)
        self.assertEqual(result["systems"]["test"]["overall"]["question_accuracy"], 0.0)
        self.assertEqual(result["systems"]["test"]["datasets"]["ordinary"]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
