import argparse
import gzip
import hashlib
import json
import math
import re
import tarfile
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .acquire import dataset_purpose, validate_dataset, write_json
from .external_sampling import SEED as EXTERNAL_SEED, sample_evidence
from .paths import project_path
from .scope import REVISION as SCOPE_REVISION, validate_row as validate_scope_row
from .state import Store


@lru_cache(maxsize=1)
def acquisition():
    return json.loads(project_path("data/manifests/acquisition.json").read_text())["datasets"]


def download(name, filename):
    info = validate_dataset(name, acquisition()[name])
    return Path(hf_hub_download(info["repo"], filename, repo_type="dataset", revision=info["revision"]))


def acquire_text():
    for name, patterns in [("helpsteer2", ["README.md", "train.jsonl.gz", "validation.jsonl.gz", "preference/*"]),
                           ("helpsteer3", ["README.md", "preference/train.jsonl.gz", "preference/validation.jsonl.gz"]),
                           ("boolq", ["README.md", "data/train-*.parquet", "data/validation-*.parquet"]),
                           ("mm-rlhf", ["README.md", "data.jsonl"]),
                           ("qasper", ["README.md", "qasper.py"])]:
        info = validate_dataset(name, acquisition()[name])
        print(f"Acquiring annotations: {name}", flush=True)
        snapshot_download(info["repo"], repo_type="dataset", revision=info["revision"],
                          allow_patterns=patterns, max_workers=2)


def json_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def stable_id(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def conversation(value):
    if isinstance(value, list):
        messages = [{"role": row["role"], "content": row["content"]} for row in value]
        if any(row["role"] not in {"user", "assistant", "system"}
               or not isinstance(row["content"], str) for row in messages):
            raise ValueError("Unsupported source conversation schema")
        return messages
    parts = re.split(r"<extra_id_1>(User|Assistant)\n", value)
    if len(parts) == 1:
        return [{"role": "user", "content": value}]
    messages = ([{"role": "user", "content": parts[0].rstrip()}]
                if parts[0].strip() else [])
    for role, content in zip(parts[1::2], parts[2::2]):
        content = content.removesuffix("<extra_id_1>").strip()
        if content:
            messages.append({"role": role.lower(), "content": content})
    if not messages:
        raise ValueError("Empty source conversation")
    return messages


def prompt_group(messages):
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "prompt:" + stable_id(canonical)


def group_role(group, source, original_split):
    if original_split in {"validation", "val", "dev", "test"}:
        return "test"
    number = int(stable_id("20261001", group)[:16], 16) / 2**64
    if source == "mm-rlhf":
        return "fit" if number < .7 else "development" if number < .8 else "calibration" if number < .9 else "test"
    return "fit" if number < .8 else "development" if number < .9 else "calibration"


def record(source, split, group, context, question, candidates, target, primitive="choice", **metadata):
    if len(target) != len(candidates) or any(not math.isfinite(x) or x < 0 for x in target) or not math.isclose(sum(target), 1):
        raise ValueError("Target must be a finite probability distribution over candidates")
    return {"id": stable_id(source, group, context, question, json.dumps(candidates), primitive, target),
            "source": source, "revision": acquisition()[source]["revision"], "group": group,
            "original_split": split, "role": group_role(group, source, split), "modality": "text",
            "context": context, "question": question, "candidates": candidates,
            "target": target, "primitive": primitive, **metadata}


def scoped_record(source, split, group, context, question, candidates, target, primitive="choice", **metadata):
    purpose = dataset_purpose(source)
    validate_dataset(source, acquisition()[source], purpose)
    if purpose == "external" and split != "test":
        raise ValueError("External adapters require the original official test split")
    row = record(source, split, group, context, question, candidates, target, primitive,
                 dataset_scope_revision=SCOPE_REVISION, dataset_purpose=purpose, **metadata)
    return validate_scope_row(row)


def passage_group(passage):
    canonical = " ".join(unicodedata.normalize("NFC", passage).split())
    if not canonical:
        raise ValueError("BoolQ requires a nonempty passage")
    return "passage:" + stable_id(canonical)


def boolq_records(row, split, source_id=None):
    if (split not in {"train", "validation"} or type(row.get("answer")) is not bool
            or not isinstance(row.get("question"), str) or not row["question"].strip()
            or not isinstance(row.get("passage"), str)):
        raise ValueError("Unsupported official google/boolq schema or split")
    yes = float(row["answer"])
    yield scoped_record("boolq", split, passage_group(row["passage"]),
                        [{"role": "user", "content": row["passage"]}], row["question"],
                        [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
                        [yes, 1 - yes], "noul", adaptation="boolean_truth",
                        original_answer=row["answer"], evaluation_scope="training",
                        source_example_id=source_id or stable_id(split, row["passage"], row["question"]))


def boolq_source_rows():
    import pyarrow.parquet as parquet

    info = validate_dataset("boolq", acquisition()["boolq"], "training")
    for split in ("train", "validation"):
        files = sorted(item["path"] for item in info["files"]
                       if re.fullmatch(rf"data/{split}-\d+-of-\d+\.parquet", item["path"]))
        if not files:
            raise ValueError(f"Pinned google/boolq is missing its {split} parquet files")
        for filename in files:
            file = parquet.ParquetFile(download("boolq", filename))
            offset = 0
            for batch in file.iter_batches(batch_size=128, columns=["question", "answer", "passage"]):
                for row in batch.to_pylist():
                    yield split, f"{filename}:{offset}", row
                    offset += 1


def boolq_training_records():
    for split, source_id, row in boolq_source_rows():
        yield from boolq_records(row, split, source_id)


def ordinal_records(row, split):
    prompt, response = row["prompt"], row["response"]
    messages = conversation(prompt)
    group = prompt_group(messages)
    context = [*messages, {"role": "assistant", "content": response}]
    for attribute in ("helpfulness", "correctness", "coherence", "complexity", "verbosity"):
        score = row.get(attribute)
        if score is None or not 0 <= score <= 4:
            continue
        target = [0.] * 5
        low, high = int(score), min(4, int(score) + 1)
        target[low] += 1 - (score - low)
        target[high] += score - low
        yield record("helpsteer2", split, group, context,
                     f"Rate the response's {attribute} on the HelpSteer2 scale from 0 to 4.",
                     [{"id": str(i), "text": f"{attribute}: {i}", "ordinal_value": i} for i in range(5)],
                     target, "ordinal", attribute=attribute, original_score=score)


def preference_records(row, source, split):
    split = row.get("split", split)
    if source == "helpsteer2":
        messages = conversation(row["prompt"])
        a, b, preference = row["response_1"], row["response_2"], row["preference_strength"]
    elif source == "helpsteer3":
        messages = conversation(row["context"])
        a, b, preference = row["response1"], row["response2"], row["overall_preference"]
    else:
        raise ValueError(f"Unknown {source} preference schema: {sorted(row)}")
    if not isinstance(preference, (int, float)) or not math.isfinite(preference):
        raise ValueError("Nonfinite signed preference")
    # NVIDIA's signed preference is positive when response 2 is preferred.
    target = [1., 0.] if preference < 0 else [0., 1.] if preference > 0 else [.5, .5]
    yield record(source, split, prompt_group(messages), messages,
                 "Which response better answers the user's request?",
                 [{"id": "a", "text": a}, {"id": "b", "text": b}], target,
                 original_preference=preference)


QASPER_ARCHIVES = {
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
QASPER_TEST_SHA256 = "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"
QASPER_CENSUS_COUNTS = {"yes_no": 239, "answerability": 1450}


def qasper_documents():
    validate_dataset("qasper", acquisition()["qasper"], "external")
    path = project_path("data/raw/qasper/test.tgz")
    if not path.is_file():
        raise FileNotFoundError("Acquire the pinned official QASPER test archive before preparation")
    if path.stat().st_size > 512 << 20:
        raise ValueError("QASPER archive exceeds acquisition bound")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != QASPER_TEST_SHA256:
        raise ValueError("Official QASPER test archive SHA256 mismatch")
    with tarfile.open(path, "r:gz") as handle:
        members = [member for member in handle if member.isfile()
                   and Path(member.name).name == "qasper-test-v0.3.json"]
        if len(members) != 1 or members[0].size > 512 << 20:
            raise ValueError("Expected exactly one bounded official QASPER test JSON")
        with handle.extractfile(members[0]) as source:
            papers = json.load(source)
        for paper_id, paper in sorted(papers.items()):
            yield "test", paper_id, paper


def qasper_records(split, paper_id, paper, *, emit=record):
    """Vote adapter; publication goes through the test-only external builder."""
    paragraphs = [paragraph for section in paper["full_text"]
                  for paragraph in section["paragraphs"]]
    full_text = "\n\n".join([paper["title"], paper["abstract"], *paragraphs])
    context = [{"role": "user", "content": full_text}]
    group = "paper:" + str(paper_id)
    candidates = [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
    for qa in paper["qas"]:
        answers = [answer["answer"] for answer in qa["answers"]]
        if not answers:
            continue
        yes = sum(not answer["unanswerable"] for answer in answers) / len(answers)
        yield emit("qasper", split, group, context,
                     f"Can the supplied paper answer this question? {qa['question']}",
                     candidates, [yes, 1 - yes], "noul", adaptation="answerability",
                     source_question_id=qa["question_id"])
        valid = [answer["yes_no"] for answer in answers if answer.get("yes_no") is not None
                 and not answer["unanswerable"]]
        if valid:
            yes = sum(valid) / len(valid)
            yield emit("qasper", split, group, context, qa["question"], candidates,
                         [yes, 1 - yes], "noul", adaptation="yes_no",
                         source_question_id=qa["question_id"])
        answerable = [answer for answer in answers if not answer["unanswerable"]]
        if not answerable:
            continue
        for paragraph in dict.fromkeys(paragraphs):
            votes = sum(paragraph in answer.get("evidence", []) for answer in answerable)
            yes = votes / len(answerable)
            yield emit("qasper", split, group, context,
                         f"Does this paragraph support an answer to the question?\n"
                         f"Question: {qa['question']}\nParagraph: {paragraph}",
                         candidates, [yes, 1 - yes], "noul", adaptation="evidence",
                         source_question_id=qa["question_id"], evidence_id=stable_id(paragraph))


def qasper_external_records(documents, *, evidence_size=2000, expected_census=None):
    expected_census = QASPER_CENSUS_COUNTS if expected_census is None else expected_census
    pool, papers, raw_counts, duplicates = {}, set(), Counter(), Counter()

    def compact(source, split, group, context, question, candidates, target, primitive, **metadata):
        # Context is shared within a paper; do not hash its full text for every
        # unselected paragraph. The final record retains the original renderer.
        key = stable_id(group, question, json.dumps(candidates), primitive, target)
        return {"id": key, "group": group, "context": context, "question": question,
                "candidates": candidates, "target": target, "primitive": primitive, **metadata}

    for split, paper_id, paper in documents:
        if split != "test":
            raise ValueError("QASPER external pool must contain only original test papers")
        if paper_id in papers:
            raise ValueError("Duplicate QASPER paper in sampling frame")
        papers.add(paper_id)
        question_ids = [qa["question_id"] for qa in paper["qas"]]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("Duplicate QASPER source question ID")
        for row in qasper_records(split, paper_id, paper, emit=compact):
            adaptation = row["adaptation"]
            raw_counts[adaptation] += 1
            source_id = str(row["source_question_id"])
            if row["id"] in pool:
                previous = pool[row["id"]]
                if previous["adaptation"] != adaptation:
                    raise ValueError("QASPER adaptations have ambiguous record identity")
                previous["source_question_ids"].append(source_id)
                duplicates[adaptation] += 1
            else:
                pool[row["id"]] = {**row, "source_paper_id": str(paper_id),
                                    "source_question_ids": [source_id]}
    for row in pool.values():
        row["source_question_ids"] = sorted(set(row["source_question_ids"]))
        row["source_question_id"] = row["source_question_ids"][0]
        row["source_test_example_id"] = row["source_question_id"]
    census = [row for row in pool.values() if row["adaptation"] != "evidence"]
    census_counts = dict(Counter(row["adaptation"] for row in census))
    if census_counts != expected_census:
        raise ValueError(f"QASPER census count mismatch: {census_counts} != {expected_census}")
    evidence, design = sample_evidence((row for row in pool.values() if row["adaptation"] == "evidence"),
                                       size=evidence_size, seed=EXTERNAL_SEED)
    for row in census:
        row.update(sampling_weight=1., inclusion_probability=1., sampling_design="census",
                   sampling_seed=EXTERNAL_SEED, sampling_cluster=row["group"],
                   population_size=census_counts[row["adaptation"]],
                   sample_size=census_counts[row["adaptation"]])
    result = []
    for item in sorted([*census, *evidence], key=lambda row: (row["adaptation"], row["id"])):
        row = {key: value for key, value in item.items() if key != "id"}
        result.append(scoped_record("qasper", "test", evaluation_scope="external",
                                    source_original_split="test", **row))
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("Duplicate rendered QASPER records after sampling")
    counts = {**census_counts, "evidence": evidence_size}
    if dict(Counter(row["adaptation"] for row in result)) != counts:
        raise AssertionError("QASPER external exact-count invariant failed")
    return result, {"adaptation_counts": counts, "raw_counts": dict(raw_counts),
                    "excluded_exact_duplicates": dict(duplicates), "sampling": design,
                    "census_unit": "unique rendered paper/question/target adaptation",
                    "source_archive_sha256": QASPER_TEST_SHA256,
                    "source_archive_url": QASPER_ARCHIVES["test"]}


def text_records():
    for split in ("train", "validation"):
        for row in json_lines(download("helpsteer2", f"{split}.jsonl.gz")):
            yield from ordinal_records(row, split)
    for row in json_lines(download("helpsteer2", "preference/preference.jsonl.gz")):
        yield from preference_records(row, "helpsteer2", "train")
    for split in ("train", "validation"):
        for row in json_lines(download("helpsteer3", f"preference/{split}.jsonl.gz")):
            yield from preference_records(row, "helpsteer3", split)


def prepared_records(factory, path, summary_path, *, seed=20261001, details=None,
                     expected_records=None, filename=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    roles, counts = {}, Counter()
    for row in factory():
        if filename is not None:
            validate_scope_row(row, filename)
        old = roles.get(row["group"])
        roles[row["group"]] = "test" if old == "test" or row["role"] == "test" else row["role"]
    seen = set()
    digest = hashlib.sha256()
    with path.with_suffix(".tmp").open("w") as handle:
        for row in factory():
            if filename is not None:
                validate_scope_row(row, filename)
            if row["role"] != "test" and roles[row["group"]] == "test":
                counts["excluded_holdout_overlap"] += 1
                continue
            if row["id"] in seen:
                counts["excluded_exact_duplicates"] += 1
                continue
            seen.add(row["id"])
            counts[f"{row['source']}/{row['primitive']}/{row['role']}"] += 1
            line = json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
            digest.update(line.encode())
            handle.write(line)
    if expected_records is not None and len(seen) != expected_records:
        path.with_suffix(".tmp").unlink()
        raise ValueError(f"Prepared record count mismatch: {len(seen)} != {expected_records}")
    path.with_suffix(".tmp").replace(path)
    summary = {"records": len(seen), "counts": dict(counts),
               "sha256": digest.hexdigest(), "seed": seed,
               "groups": len(roles), "path": str(path), **(details or {})}
    write_json(summary_path, summary)
    Store().record_result("data.splits", summary)
    display = {**summary}
    if "sampling" in display:
        display["sampling"] = {key: value for key, value in summary["sampling"].items() if key != "strata"}
        display["sampling"]["strata_count"] = len(summary["sampling"].get("strata", []))
    print(json.dumps(display, indent=2), flush=True)
    return summary


def prepare_text():
    store = Store()
    store.update("data.splits", "in_progress", "Preparing grouped text records; official held-outs remain untouched.")
    return prepared_records(text_records, project_path("data/prepared/text.jsonl"),
                            project_path("data/manifests/text-splits.json"))


def prepare_qasper():
    rows, summary = qasper_external_records(qasper_documents())
    return prepared_records(lambda: iter(rows), project_path("data/prepared/qasper-external.jsonl"),
                            project_path("data/manifests/qasper-external-splits.json"),
                            seed=EXTERNAL_SEED, expected_records=3689, filename="qasper-external.jsonl",
                            details={**summary, "dataset_scope_revision": SCOPE_REVISION,
                                     "evaluation_scope": "external", "original_split": "test"})


def prepare_boolq():
    return prepared_records(boolq_training_records, project_path("data/prepared/boolq.jsonl"),
                            project_path("data/manifests/boolq-splits.json"), filename="boolq.jsonl",
                            details={"dataset_scope_revision": SCOPE_REVISION,
                                     "evaluation_scope": "training", "adaptation": "boolean_truth"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire-text", "prepare-text", "prepare-boolq",
                                           "prepare-qasper", "prepare-qasper-external", "inspect"])
    args = parser.parse_args()
    if args.command == "acquire-text":
        acquire_text()
    elif args.command == "prepare-text":
        prepare_text()
    elif args.command == "prepare-boolq":
        prepare_boolq()
    elif args.command in {"prepare-qasper", "prepare-qasper-external"}:
        prepare_qasper()
    else:
        sources = [("helpsteer2", "train.jsonl.gz"), ("helpsteer2", "preference/preference.jsonl.gz"),
                   ("helpsteer3", "preference/train.jsonl.gz"), ("mm-rlhf", "data.jsonl")]
        for source, filename in sources:
            row = next(json_lines(download(source, filename)))
            print(json.dumps({"source": source, "fields": {k: type(v).__name__ for k,v in row.items()},
                              "preference_value": row.get("overall_preference", row.get("preference")),
                              "ranking": row.get("final_ranking")}), flush=True)


if __name__ == "__main__":
    main()
