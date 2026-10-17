from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import residual_blend as blend


def fixture(role, margins):
    records, scores = [], []
    for index, (baseline, head) in enumerate(margins):
        group = f"{role}:image:{index}"
        gold = str(index % 2)
        for view, order in enumerate(("ab", "ba")):
            ids = ["0", "1"] if order == "ab" else ["1", "0"]
            record_id = f"{group}:{order}"
            records.append({"id": record_id, "group": group, "role": role, "order": order,
                            "source_image_id": group, "source_example_id": str(index),
                            "question": "Which caption matches?",
                            "candidates": [{"id": id_, "text": f"Caption {id_}."} for id_ in ids],
                            "target": [float(id_ == gold) for id_ in ids]})
            score = {"id": record_id, "candidate_ids": ids}
            for name, values in (("baseline", baseline), ("head", head)):
                score[name] = {"scores": [{"id": id_, "logit": values[view] if id_ == gold else 0.}
                                           for id_ in ids[::-1]]}
            scores.append(score)
    return records[::-1], scores


class ResidualTests(unittest.TestCase):
    def test_endpoints_alignment_and_image_unit_flip_report(self):
        records, scores = fixture("development", [((2, 2), (-2, -2)),
                                                  ((-2, -2), (2, -2)),
                                                  ((2, -2), (2, 2))])
        data = blend.align(records, scores, "development")
        np.testing.assert_array_equal(blend.blend(data, 0.), data["baseline"])
        np.testing.assert_array_equal(blend.blend(data, 1.), data["v1.1_head"])
        offset = {**data, "baseline": data["baseline"] + 1.e16}
        np.testing.assert_array_equal(blend.blend(offset, 1.), data["v1.1_head"])
        result = blend.report(data, {"name": "v1.1_head", "lambda": 1., "temperature": 2.}, 1024, 17)
        self.assertEqual((result["records"], result["image_groups"]), (6, 3))
        head = result["selected"]
        self.assertEqual(head["metrics"]["accuracy"], .5)
        self.assertAlmostEqual(head["order_agreement"]["mean"], 2 / 3)
        for event in ("harmful_flips", "repairs"):
            self.assertEqual(head["vs_baseline"][event]["count"], 2)
            self.assertAlmostEqual(head["vs_baseline"][event]["mean"], 1 / 3)
            self.assertEqual(head["vs_baseline"][event]["groups"], 3)
        delta = head["vs_baseline"]["accuracy_delta"]
        self.assertEqual((delta["mean"], delta["low"], delta["high"]), (0., -1., .5))
        self.assertNotEqual(result["v1.1_head"]["metrics"]["nll"], head["metrics"]["nll"])
        # A tied positional argmax disagrees across AB/BA semantic candidate IDs.
        tied = blend.align(*fixture("fit", [((0, 0), (0, 0))]), "fit")
        tied_report = blend.report(tied, {"name": "baseline", "lambda": 0., "temperature": 1.}, 8, 1)
        self.assertEqual(tied_report["baseline"]["order_agreement"]["mean"], 0.)

    def test_fit_is_independent_of_dev_which_only_selects(self):
        fit = blend.align(*fixture("fit", [((3, -1), (-1, 3))]), "fit")
        dev_a = blend.align(*fixture("development", [((3, -1), (-1, 3))]), "development")
        dev_b = blend.align(*fixture("development", [((3, 3), (-20, -20))]), "development")
        a = blend.select(fit, dev_a, (.1, 10.), 16, 1)
        b = blend.select(fit, dev_b, (.1, 10.), 16, 1)
        self.assertEqual(a["selected"]["name"], "fit_residual")
        self.assertAlmostEqual(a["selected"]["lambda"], .5, places=5)
        self.assertEqual(b["selected"]["name"], "baseline")
        for first, second in zip(a["candidates"], b["candidates"]):
            self.assertEqual({k: v for k, v in first.items() if k != "dev_normalized_nll"},
                             {k: v for k, v in second.items() if k != "dev_normalized_nll"})
        # Stable NLL remains informative when the wrong-class probability underflows.
        np.testing.assert_allclose(blend.losses(np.array([[1000., 0.]]), np.array([[0., 1.]])),
                                   [1000. / np.log(2.)])

    def test_alignment_rejects_missing_scores_wrong_order_gold_and_nonfinite(self):
        records, scores = fixture("fit", [((1, 1), (2, 2))])
        bad_order = deepcopy(scores)
        bad_order[0]["candidate_ids"].reverse()
        bad_logits = deepcopy(scores)
        bad_logits[0]["head"]["scores"][0]["logit"] = float("nan")
        bad_gold = deepcopy(records)
        bad_gold[0]["target"].reverse()
        for rows, values in ((records, scores[:-1]), (records, scores + scores[:1]),
                             (records, bad_order), (records, bad_logits), (bad_gold, scores)):
            with self.subTest(rows=rows, scores=values), self.assertRaises(ValueError):
                blend.align(rows, values, "fit")
        with self.assertRaisesRegex(ValueError, "role"):
            blend.align(records, scores, "development")

    def test_cli_selection_and_frozen_reporting_on_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ("fit", "development", "test"):
                for kind, rows in zip(("records", "scores"), fixture(role, [((3, -1), (-1, 3))])):
                    (root / f"{role}-{kind}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            common = ["--bootstrap-draws", "16", "--seed", "5"]
            selection_args = ["select", "--temperature-min", ".1", "--temperature-max", "10"]
            for short, role in (("fit", "fit"), ("dev", "development")):
                for kind in ("records", "scores"):
                    selection_args.extend([f"--{short}-{kind}", str(root / f"{role}-{kind}.jsonl")])
            output = io.StringIO()
            with redirect_stdout(output):
                blend.main(selection_args + common)
            selection = json.loads(output.getvalue())
            frozen = root / "selection.json"
            frozen.write_text(output.getvalue())
            output = io.StringIO()
            with redirect_stdout(output):
                blend.main(["report", "--selection", str(frozen),
                            "--eval-records", str(root / "test-records.jsonl"),
                            "--eval-scores", str(root / "test-scores.jsonl")] + common)
            result = json.loads(output.getvalue())
            self.assertEqual(result["role"], "test")
            for key, value in selection["selected"].items():
                self.assertEqual(result["selected"][key], value)
            self.assertEqual(result["selected"]["metrics"]["accuracy"], 1.)
            data = blend.align(*fixture("test", [((3, -1), (-1, 3))]), "test")
            data["image_ids"] = selection["used_images"]["image_ids"][:1]
            with self.assertRaisesRegex(ValueError, "overlap"):
                blend.evaluate(data, selection, 16, 5)


if __name__ == "__main__":
    unittest.main()
