"""Prepare judged-only, grouped WANDS role JSONL without model execution."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import unicodedata


KS = (26, 64, 128, 255)
ROLES = ("fit", "development", "calibration", "test")
GRADES = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
CONTEXT = [{"role": "user", "content": "Find products matching the search query."}]


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def input_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from input_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from input_strings(child)


def historical_inputs(paths):
    texts, ids, counts = set(), set(), []
    for path in paths:
        count = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if "state" in row:
                    values = [row["state"]]
                    for question in row["questions"].values():
                        criteria = question.get("criteria")
                        values.extend((question.get("instructions"), criteria))
                        if isinstance(criteria, dict):
                            values.append(list(criteria))
                else:
                    values = [row.get("question"), [m["content"] for m in row.get("context", [])],
                              [c["text"] for c in row.get("candidates", [])]]
                texts.update(value for text in input_strings(values) if (value := normalized(text)))
                if row.get("source") == "wands" and "source_question_id" in row:
                    ids.add(str(row["source_question_id"]))
                count += 1
        counts.append(count)
    return texts, ids, counts


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_pools(raw, excluded_texts, excluded_ids):
    query_rows, product_rows = csv_rows(raw / "query.csv"), csv_rows(raw / "product.csv")
    queries = {r["query_id"]: r["query"] for r in query_rows}
    query_ids = set(queries)
    products = {r["product_id"]: r for r in product_rows}
    if len(queries) != len(query_rows) or len(products) != len(product_rows):
        raise ValueError("Duplicate source query/product IDs")
    blocked_groups = {normalized(text) for qid, text in queries.items() if qid in excluded_ids}
    blocked_queries = {qid for qid, text in queries.items()
                       if normalized(text) in blocked_groups or normalized(text) in excluded_texts}
    if any(not normalized(text) for text in queries.values()):
        raise ValueError("Empty normalized query")
    queries = {qid: text for qid, text in queries.items() if qid not in blocked_queries}
    texts = {pid: "product_name: " + r["product_name"] + "\nproduct_description: "
             + r["product_description"] for pid, r in products.items()}
    blocked_products = {pid for pid, r in products.items()
                        if any(value and value in excluded_texts for value in
                               (normalized(texts[pid]), normalized(r["product_description"]),
                                normalized(r["product_name"] + "\n" + r["product_description"])))}
    pairs = defaultdict(Counter)
    labels = csv_rows(raw / "label.csv")
    for row in labels:
        qid, pid, label = row["query_id"], row["product_id"], row["label"]
        if qid not in query_ids or pid not in products or label not in GRADES:
            raise ValueError("Unknown source query, product, or grade")
        pairs[qid, pid][label] += 1
    grouped = defaultdict(lambda: defaultdict(list))
    for (qid, pid), votes in pairs.items():
        if qid not in queries or pid in blocked_products or len(votes) != 1:
            continue
        product = products[pid]
        key = normalized(product["product_name"]), normalized(product["product_description"])
        grouped[qid][key].append((pid, next(iter(votes))))
    pools, text_conflicts, conflicting_products, duplicate_extras = {}, 0, 0, 0
    for qid in queries:
        pool = []
        for group in grouped[qid].values():
            if len({label for _, label in group}) != 1:
                text_conflicts += 1
                conflicting_products += len(group)
                continue
            pid, label = min(group, key=lambda item: int(item[0]))
            duplicate_extras += len(group) - 1
            pool.append({"id": pid, "text": texts[pid], "label": label,
                         "has_description": bool(products[pid]["product_description"].strip())})
        pools[qid] = pool
    return queries, pools, {
        "source_queries": len(query_rows), "source_products": len(products),
        "judgment_rows": len(labels), "distinct_judged_pairs": len(pairs),
        "pair_grade_conflicts_excluded": sum(len(votes) > 1 for votes in pairs.values()),
        "same_grade_duplicate_judgment_extras": sum(n - 1 for v in pairs.values() for n in v.values()),
        "rendered_grade_conflict_groups_excluded": text_conflicts,
        "rendered_grade_conflict_product_pairs_excluded": conflicting_products,
        "same_grade_rendered_duplicate_extras": duplicate_extras,
        "excluded_query_ids": sorted(blocked_queries, key=int),
        "excluded_product_ids": sorted(blocked_products, key=int),
    }


def split_groups(queries, seed):
    groups = sorted({normalized(text) for text in queries.values()})
    random.Random(f"{seed}:roles").shuffle(groups)
    proportions = (60, 15, 10, 15)
    counts = [len(groups) * p // 100 for p in proportions]
    remainder = len(groups) - sum(counts)
    order = sorted(range(4), key=lambda i: (-(len(groups) * proportions[i] % 100), i))
    for i in order[:remainder]:
        counts[i] += 1
    roles, offset = {}, 0
    for role, count in zip(ROLES, counts):
        roles.update((group, role) for group in groups[offset:offset + count])
        offset += count
    return roles


def records(queries, pools, roles, seed):
    for qid, query in sorted(queries.items(), key=lambda item: int(item[0])):
        pool = sorted(pools[qid], key=lambda c: int(c["id"]))
        if len(pool) < KS[0] or not any(c["label"] == "Exact" for c in pool) or not any(
                c["label"] != "Exact" for c in pool):
            continue
        random.Random(f"{seed}:{qid}:members").shuffle(pool)
        # Eligibility changes membership, never the independently shuffled presentation.
        for exact in (True, False):
            if not any((c["label"] == "Exact") == exact for c in pool[:KS[0]]):
                index = next(i for i, c in enumerate(pool) if (c["label"] == "Exact") == exact)
                pool[KS[0] - 1], pool[index] = pool[index], pool[KS[0] - 1]
        for k in KS:
            if len(pool) < k:
                continue
            selected = pool[:k].copy()
            random.Random(f"{seed}:{qid}:{k}:presentation").shuffle(selected)
            acceptable = [c["id"] for c in selected if c["label"] == "Exact"]
            group = normalized(query)
            yield {"id": f"wands:{qid}:k{k}:seeded", "source": "wands",
                   "source_question_id": qid, "original_split": "unsplit",
                   "group": "wands-query:" + group, "role": roles[group],
                   "primitive": "choice", "modality": "text", "context": CONTEXT,
                   "question": "Which product best matches this search query? " + query,
                   "candidates": [{"id": c["id"], "text": c["text"]} for c in selected],
                   "target": [1 / len(acceptable) if c["id"] in acceptable else 0.0 for c in selected],
                   "grades": [GRADES[c["label"]] for c in selected],
                   "grade_labels": [c["label"] for c in selected],
                   "acceptable_candidate_ids": acceptable, "k": k, "order": "seeded",
                   "seed": seed, "missing_description_candidate_ids":
                   [c["id"] for c in selected if not c["has_description"]]}


def prepare(args):
    excluded_texts, excluded_ids, history = historical_inputs(args.exclude_records)
    queries, pools, summary = load_pools(args.raw, excluded_texts, excluded_ids)
    roles = split_groups(queries, args.seed)
    rows = list(records(queries, pools, roles, args.seed))
    args.output.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        with (args.output / f"{role}.jsonl").open("x") as handle:
            for row in rows:
                if row["role"] == role:
                    handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    summary.update({"seed": args.seed,
                    "excluded_input_row_counts": history, "known_input_texts": len(excluded_texts),
                    "all_query_groups_by_role": dict(Counter(roles.values())),
                    "retained_queries": len(queries), "rows": len(rows),
                    "by_role_k": {role: {str(k): sum(r["role"] == role and r["k"] == k for r in rows)
                                         for k in KS} for role in ROLES},
                    "by_k": {str(k): {
                        "rows": sum(r["k"] == k for r in rows),
                        "multiple_exact": sum(r["k"] == k and len(r["acceptable_candidate_ids"]) > 1 for r in rows),
                        "with_irrelevant": sum(r["k"] == k and 0 in r["grades"] for r in rows),
                        "missing_description_occurrences": sum(len(r["missing_description_candidate_ids"])
                                                               for r in rows if r["k"] == k)} for k in KS},
                    "limitation": "Lexical query groups/exact input overlap only; semantic, pretraining and text-only judgment sufficiency unverified."})
    print(json.dumps(summary, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude-records", type=Path, nargs="+", required=True)
    prepare(parser.parse_args(argv))


if __name__ == "__main__":
    main()
