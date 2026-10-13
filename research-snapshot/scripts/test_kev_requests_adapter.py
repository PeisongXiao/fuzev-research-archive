"""Run: python3 -B -S -m unittest discover -s scripts -p test_kev_requests_adapter.py -v"""

import copy
import json
import unittest

import kev_requests_adapter as adapter
from fuzev_research.encoding import candidate_suffix, query_suffix
from fuzev_research.preparation import ReadoutPreparer


def record(questions, state="Shared state"):
    return {"state": state, "questions": questions,
            "_meta": {"id": "dataset/record", "source": "dataset", "group_id": "dataset/group",
                      "variant": "clean", "private_annotation": "TEACHER_ONLY"}}


class Tokens:
    def __init__(self, values):
        self.values = values

    def reshape(self, *_):
        return self

    def tolist(self):
        return self.values[:]


class FakeTokenizer:
    """Exercise the real stdlib preparer, not a tokenizer/model/head."""
    def __init__(self):
        self.inputs = []

    def encode(self, text, **_):
        self.inputs.append(text)
        values = [ord(text[1])] if len(text) == 2 and text[0] == " " else list(text.encode("utf-8"))
        return Tokens(values)

    def hf_render_chat_template(self, messages, **_):
        return json.dumps(messages, ensure_ascii=False)


class AdapterTests(unittest.TestCase):
    def test_official_structured_render(self):
        examples = [
            ("  plain\n", "  plain\n"), (None, ""), (True, "True"),
            ({"what": "A", "not_for": "B"}, "what: A\nnot_for: B"),
            (["x", "y"], "- x\n- y"),
            ({"ticket": {"channel": "email", "body": "hi"}}, "ticket:\n  channel: email\n  body: hi"),
            ({"examples": ["a", "b"]}, "examples:\n  - a\n  - b"),
        ]
        for value, expected in examples:
            with self.subTest(value=value):
                self.assertEqual(adapter.state_text(value), expected)

    def test_choice_full_criteria_and_order(self):
        long_text = "Full answer\n" + "detail " * 2000
        criteria = {"z_key": long_text, "intent_key": None, "empty": "", "object": {"body": ["a", "b"]}}
        instructions = {"request": "Which option?", "requirements": ["Use all details"]}
        source = record({"qID": {"type": "choice", "instructions": instructions,
                                 "criteria": criteria, "label": "intent_key", "src": "task"}})
        row = adapter.adapt_record(source)[0]
        self.assertEqual(row["context"], [{"role": "user", "content": "Shared state"}])
        self.assertEqual(row["question"], "request: Which option?\nrequirements:\n  - Use all details")
        self.assertEqual([c["id"] for c in row["candidates"]], list(criteria))
        self.assertEqual([c["text"] for c in row["candidates"]],
                         ["z_key: " + long_text, "intent_key", "empty", "object: body:\n  - a\n  - b"])
        self.assertEqual(row["target"], [0., 1., 0., 0.])
        self.assertEqual(row["official_criteria"], criteria)
        self.assertEqual(row["official_instructions"], instructions)
        self.assertEqual(row["official_label"], "intent_key")
        self.assertEqual(row["official_record_id"], source["_meta"]["id"])
        self.assertEqual(row["official_question_id"], "qID")
        self.assertEqual(row["_meta"], source["_meta"])
        self.assertEqual(adapter.to_official_order(row, [1, 2, 3, 4]), [1, 2, 3, 4])

    def test_noul_targets_and_explicit_permutation(self):
        for label in (False, True):
            with self.subTest(label=label):
                row = adapter.adapt_record(record({"n": {"type": "noul", "label": label}}))[0]
                self.assertEqual(row["candidates"], [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}])
                self.assertEqual(row["official_keys"], ["false", "true"])
                self.assertEqual(row["candidate_keys"], ["true", "false"])
                self.assertEqual(row["target"], [float(label), float(not label)])
                self.assertEqual(row["official_label_index"], int(label))
                self.assertEqual(row["official_from_fuzev"], [1, 0])
                self.assertEqual(row["fuzev_from_official"], [1, 0])
                self.assertEqual(adapter.to_fuzev_order(row, [.8, .2]), [.2, .8])
                self.assertEqual(adapter.to_official_order(row, [.2, .8]), [.8, .2])
                native = adapter.prediction_metadata(row, [.2, .8], input_order="fuzev")
                baseline = adapter.prediction_metadata(row, [.8, .2], input_order="official")
                self.assertEqual(native, baseline)
                self.assertEqual(native["p"], [.8, .2])
                self.assertEqual(native["label"], int(label))
                self.assertEqual(native["question"], "n")
                self.assertEqual(native["id"], "dataset/record")

    def test_noul_descriptions_preserved(self):
        criteria = {"true": {"meaning": "Call the tool"}, "false": "Do not call\nany tool"}
        row = adapter.adapt_record(record({"n": {"type": "noul", "criteria": criteria, "label": True}}))[0]
        self.assertEqual([c["text"] for c in row["candidates"]],
                         ["Yes: meaning: Call the tool", "No: Do not call\nany tool"])
        self.assertEqual(row["official_criteria"], criteria)

    def test_score_order_and_index_target(self):
        levels = ["low", {"name": "medium", "details": "Full middle level"}, "high"]
        row = adapter.adapt_record(record({"s": {"type": "score", "instructions": "Rate", "criteria": levels, "label": 1}}))[0]
        self.assertEqual(row["primitive"], "ordinal")
        self.assertEqual(row["candidates"], [{"id": "0", "text": "low", "ordinal_value": 0},
                         {"id": "1", "text": "name: medium\ndetails: Full middle level", "ordinal_value": 1},
                         {"id": "2", "text": "high", "ordinal_value": 2}])
        self.assertEqual(row["target"], [0., 1., 0.])
        self.assertEqual(row["official_criteria"], levels)
        self.assertEqual(adapter.to_official_order(row, [.1, .7, .2]), [.1, .7, .2])
        self.assertEqual(adapter.to_fuzev_order(row, [.1, .7, .2]), [.1, .7, .2])

    def test_real_preparer_inputs_do_not_depend_on_labels_or_siblings(self):
        source = record({"own": {"type": "choice", "instructions": "Pick", "criteria": {"a": "ONE", "b": "TWO"}, "label": "a"},
                         "sibling": {"type": "noul", "instructions": "SIBLING_INSTRUCTION", "label": True}})
        original = copy.deepcopy(source)
        rows = adapter.adapt_record(source)
        changed = copy.deepcopy(source)
        changed["questions"]["own"]["label"] = "b"
        changed["questions"]["sibling"]["instructions"] = "DIFFERENT_SIBLING"
        other = adapter.adapt_record(changed)[0]
        tokenizer = FakeTokenizer()
        preparer = ReadoutPreparer(tokenizer)
        self.assertEqual(preparer.prepare(rows[0]), preparer.prepare(other))
        self.assertEqual(rows[0]["id"], other["id"])
        self.assertEqual(rows[0]["context"], rows[1]["context"])
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])
        self.assertIn(query_suffix(rows[0]["question"], rows[0]["candidates"]), tokenizer.inputs)
        self.assertIn(candidate_suffix(rows[0]["candidates"][0]), tokenizer.inputs)
        for marker in ("TEACHER_ONLY", "SIBLING_INSTRUCTION", "DIFFERENT_SIBLING"):
            self.assertNotIn(marker, "\n".join(tokenizer.inputs))
        rows[0]["_meta"]["source"] = "changed"
        rows[0]["official_criteria"]["a"] = "changed"
        rows[0]["context"][0]["content"] = "changed"
        self.assertEqual(source, original)
        self.assertEqual(rows[1]["context"][0]["content"], "Shared state")

    def test_eval_only_and_record_question_order(self):
        source = record({"second": {"type": "noul", "label": False}, "first": {"type": "noul", "label": True}})
        rows = list(adapter.iter_request_rows([source, source], split="development"))
        self.assertEqual([r["official_question_id"] for r in rows], ["second", "first"] * 2)
        for row in rows:
            self.assertEqual(row["role"], "test")
            self.assertEqual(row["original_split"], "development")
            self.assertEqual(row["evaluation_scope"], "external")
        for split in ("fit", "train", "calibration"):
            with self.assertRaises(ValueError):
                adapter.adapt_record(source, split=split)

    def test_ambiguous_labels_and_bad_prediction_vectors_fail(self):
        for label in ("false", "true", "yes", "no", 0, 1, None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                adapter.adapt_record(record({"n": {"type": "noul", "label": label}}))
        for label in (-1, 2, True, "1", 1.0):
            with self.subTest(label=label), self.assertRaises(ValueError):
                adapter.adapt_record(record({"s": {"type": "score", "criteria": ["low", "high"], "label": label}}))
        row = adapter.adapt_record(record({"n": {"type": "noul", "label": False}}))[0]
        for values in ([1.], [1., 0., 0.], {"false": .5, "true": .5}, "ab"):
            with self.subTest(values=values), self.assertRaises(ValueError):
                adapter.to_official_order(row, values)
        with self.assertRaises(ValueError):
            adapter.prediction_metadata(row, [.5, .5], input_order="implicit")
        with self.assertRaises(TypeError):
            adapter.prediction_metadata(row, [.5, .5])


if __name__ == "__main__":
    unittest.main()
