import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import prepare_wands as wands


def write_csv(root, name, rows):
    with (root / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def pool(count=270):
    return [{"id": str(i), "text": f"product {i}", "label":
             "Exact" if i in (0, 7, 269) else "Partial" if i % 2 else "Irrelevant",
             "has_description": True} for i in range(count)]


class WandsPreparationTests(unittest.TestCase):
    def test_nested_slates_keep_grades_and_acceptable_ties(self):
        queries = {"1": "Green tables"}
        rows = list(wands.records(queries, {"1": pool()}, wands.split_groups(queries, 20261015), 20261015))
        self.assertEqual([r["k"] for r in rows], [26, 64, 128, 255])
        previous = set()
        labels = {c["id"]: c["label"] for c in pool()}
        for row in rows:
            ids = [c["id"] for c in row["candidates"]]
            self.assertTrue(previous <= set(ids))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(row["grade_labels"], [labels[i] for i in ids])
            self.assertEqual(row["grades"], [wands.GRADES[labels[i]] for i in ids])
            exact = [i for i in ids if labels[i] == "Exact"]
            self.assertEqual(row["acceptable_candidate_ids"], exact)
            self.assertEqual(row["target"], [1 / len(exact) if i in exact else 0 for i in ids])
            self.assertTrue(any(labels[i] != "Exact" for i in ids))
            self.assertTrue(all(set(c) == {"id", "text"} for c in row["candidates"]))
            previous = set(ids)
        self.assertEqual(rows, list(wands.records(queries, {"1": pool()}, wands.split_groups(queries, 20261015), 20261015)))

    def test_pooled_roles_group_normalized_queries_before_k_filter(self):
        queries = {str(i): f"Query {i}" for i in range(100)}
        queries["100"] = "QUERY-0"
        roles = wands.split_groups(queries, 20261015)
        self.assertEqual([list(roles.values()).count(r) for r in wands.ROLES], [60, 15, 10, 15])
        pools = {qid: pool(270 if qid == "0" else 26) for qid in queries}
        rows = list(wands.records(queries, pools, roles, 20261015))
        grouped = {r["role"] for r in rows if r["source_question_id"] in ("0", "100")}
        self.assertEqual(len(grouped), 1)
        self.assertEqual({r["source_question_id"] for r in rows if r["k"] == 255}, {"0"})

    def test_conflicts_duplicate_text_and_unjudged_products(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "query.csv", [{"query_id": "1", "query": "a query"},
                                          {"query_id": "2", "query": "A-QUERY"}])
            names = ["alpha", "ALPHA", "beta", "BETA", "delta", "unjudged", "epsilon"]
            write_csv(root, "product.csv", [{"product_id": str(i), "product_name": name,
                       "product_description": "full description"} for i, name in enumerate(names)])
            grades = [(0, "Exact"), (1, "Partial"), (2, "Exact"), (2, "Exact"),
                      (3, "Exact"), (4, "Partial"), (4, "Irrelevant"), (6, "Partial")]
            write_csv(root, "label.csv", [{"query_id": "1", "product_id": str(i), "label": label}
                                           for i, label in grades])
            _, pools, stats = wands.load_pools(root, set(), set())
            self.assertEqual([(c["id"], c["label"]) for c in pools["1"]], [("2", "Exact"), ("6", "Partial")])
            self.assertEqual(stats["pair_grade_conflicts_excluded"], 1)
            self.assertEqual(stats["rendered_grade_conflict_groups_excluded"], 1)
            self.assertEqual(stats["rendered_grade_conflict_product_pairs_excluded"], 2)
            self.assertEqual(stats["same_grade_duplicate_judgment_extras"], 1)
            self.assertEqual(stats["same_grade_rendered_duplicate_extras"], 1)
            _, pools, _ = wands.load_pools(root, {"full description"}, set())
            self.assertEqual(pools["1"], [])
            queries, pools, stats = wands.load_pools(root, set(), {"1"})
            self.assertEqual(queries, {})
            self.assertEqual(pools, {})
            self.assertEqual(stats["excluded_query_ids"], ["1", "2"])

    def test_exclusion_uses_inputs_not_quality_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            rows = [{"state": {"query": "Black sofa"}, "questions": {"q": {
                "instructions": "Select a product", "criteria": {"desk": "full description"},
                "answer": "must not enter"}}, "_meta": {"answer": "also not an input"}},
                {"source": "wands", "source_question_id": "8", "question": "Other input",
                 "candidates": [{"id": "x", "text": "candidate body"}], "target": [1.0]}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            texts, ids, counts = wands.historical_inputs([path])
            self.assertTrue({"black sofa", "desk", "full description", "candidate body"} <= texts)
            self.assertNotIn("must not enter", texts)
            self.assertNotIn("also not an input", texts)
            self.assertEqual(ids, {"8"})
            self.assertEqual(counts, [2])

    def test_full_source_text_and_repeatable_role_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root, "query.csv", [{"query_id": "1", "query": "Green tables"}])
            description = " full attribute and rating discussion " * 200
            write_csv(root, "product.csv", [{"product_id": str(i), "product_name": f"Product {i}",
                       "product_description": description} for i in range(26)])
            write_csv(root, "label.csv", [{"query_id": "1", "product_id": str(i),
                       "label": "Exact" if i in (0, 1) else "Partial"} for i in range(26)])
            history = root / "history.jsonl"
            history.write_text(json.dumps({"question": "unrelated existing input"}) + "\n")
            for name in ("first", "second"):
                args = argparse.Namespace(raw=root, output=root / name, seed=20261015, exclude_records=[history])
                with redirect_stdout(io.StringIO()):
                    wands.prepare(args)
            for role in wands.ROLES:
                first = (root / "first" / f"{role}.jsonl").read_bytes()
                self.assertEqual(first, (root / "second" / f"{role}.jsonl").read_bytes())
                for row in map(json.loads, first.splitlines()):
                    self.assertEqual(row["target"].count(0.5), 2)
                    self.assertTrue(all(c["text"].endswith(description) for c in row["candidates"]))


if __name__ == "__main__":
    unittest.main()
