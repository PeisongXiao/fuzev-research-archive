"""Still-image adapters and cross-corpus, holdout-first image grouping.

Raw images and annotations are local acquisition artifacts, never head releases.
No rationale, judge explanation or model-generated gold answer enters a prompt.
"""

import argparse
import hashlib
import io
import json
import math
import sqlite3
import zipfile
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path, PurePosixPath

import imagehash
import pyarrow.parquet as parquet
from PIL import Image, ImageOps

from .acquire import validate_dataset, write_json
from .data import (acquisition, download, group_role, json_lines, prepared_records,
                   scoped_record, stable_id)
from .paths import project_path
from .scope import REVISION as SCOPE_REVISION, validate_row as validate_scope_row
from .state import Store


ACTIVE_IMAGE_SOURCES = ("mm-rlhf", "vl-rewardbench")


def source_files(name):
    if name not in ACTIVE_IMAGE_SOURCES:
        raise ValueError(f"Inactive image source: {name}")
    validate_dataset(name, acquisition()[name])
    files = [item["path"] for item in acquisition()[name]["files"]]
    if name == "vl-rewardbench":
        return [path for path in files if path.startswith("data/test-") and path.endswith(".parquet")]
    return [path for path in files if path in {"mcq.zip", "short.zip", "long.zip", "safety.zip"}]


def acquire_images(names):
    for name in names:
        for filename in source_files(name):
            print(f"Acquiring {name}/{filename}", flush=True)
            download(name, filename)


class ImageIndex:
    def __init__(self, path=None, image_root=None, *, readonly=False):
        self.path = path or project_path("data/manifests/image-index.sqlite3")
        self.image_root = image_root or project_path("data/images")
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            return
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                digest TEXT PRIMARY KEY, phash TEXT NOT NULL, width INTEGER,
                height INTEGER, path TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS references_ (
                source TEXT, source_id TEXT, digest TEXT, heldout INTEGER,
                PRIMARY KEY(source,source_id));
        """)

    def add(self, raw, source, source_id, heldout=False):
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > 32_000_000:
                raise ValueError("Image exceeds the 32-million-pixel decoding bound")
            image = ImageOps.exif_transpose(opened).convert("RGB")
        digest = hashlib.sha256(f"{image.width},{image.height}:".encode() + image.tobytes()).hexdigest()
        phash = str(imagehash.phash(image))
        path = self.image_root / f"{digest}.png"
        if not path.exists():
            image.save(path, format="PNG")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO images VALUES(?,?,?,?,?)",
                            (digest, phash, image.width, image.height, str(path)))
            previous = self.db.execute("SELECT digest FROM references_ WHERE source=? AND source_id=?",
                                       (source, str(source_id))).fetchone()
            if previous and previous[0] != digest:
                raise ValueError("Pinned source image identity changed")
            self.db.execute("INSERT OR REPLACE INTO references_ VALUES(?,?,?,?)",
                            (source, str(source_id), digest, int(heldout)))
        return digest, str(path)

    def resolve(self, source, source_id):
        row = self.db.execute("SELECT i.digest,i.path FROM references_ r JOIN images i USING(digest) "
                              "WHERE r.source=? AND r.source_id=?", (source, str(source_id))).fetchone()
        if row is None:
            raise ValueError(f"Unindexed image: {source}/{source_id}")
        return row

    def groups(self, max_distance=4, *, active_sources=None):
        if not 0 <= max_distance <= 4:
            raise ValueError("The five-band index is only complete at Hamming distances 0 through 4")
        # Five disjoint hash bands guarantee a shared band at Hamming distance <=4.
        # Exact verification follows the candidate lookup; hash similarity is not
        # asserted to prove semantic identity. Conservative connected components
        # keep near duplicates from straddling training and final evaluation.
        if active_sources is None:
            rows = self.db.execute("SELECT digest,phash FROM images ORDER BY digest").fetchall()
            heldout_digests = self.db.execute(
                "SELECT DISTINCT digest FROM references_ WHERE heldout=1").fetchall()
        else:
            sources = tuple(sorted(set(active_sources)))
            if not sources or any(source not in ACTIVE_IMAGE_SOURCES for source in sources):
                raise ValueError("Image grouping requires only active image sources")
            placeholders = ",".join("?" for _ in sources)
            rows = self.db.execute(
                "SELECT DISTINCT i.digest,i.phash FROM images i JOIN references_ r USING(digest) "
                f"WHERE r.source IN ({placeholders}) ORDER BY i.digest", sources).fetchall()
            heldout_digests = self.db.execute(
                f"SELECT DISTINCT digest FROM references_ WHERE source IN ({placeholders}) "
                "AND (heldout=1 OR source='vl-rewardbench')", sources).fetchall()
        parent = {digest: digest for digest, _ in rows}
        hashes = {digest: int(phash, 16) for digest, phash in rows}
        bands = {}

        def root(digest):
            while parent[digest] != digest:
                parent[digest] = parent[parent[digest]]
                digest = parent[digest]
            return digest

        for digest, value in hashes.items():
            candidates = set()
            keys = []
            for band, (shift, width) in enumerate([(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]):
                key = (band, (value >> shift) & ((1 << width) - 1))
                keys.append(key)
                candidates.update(bands.get(key, []))
            for other in candidates:
                if (value ^ hashes[other]).bit_count() <= max_distance:
                    a, b = root(digest), root(other)
                    parent[max(a, b)] = min(a, b)
            for key in keys:
                bands.setdefault(key, []).append(digest)
        group = {digest: "image:" + root(digest) for digest in hashes}
        heldout = {group[digest] for (digest,) in heldout_digests}
        return group, heldout

    def close(self):
        self.db.close()


def parquet_rows(name, *, include_images=True):
    for filename in source_files(name):
        split = "train" if "/train-" in filename else "validation" if "/validation-" in filename else "test"
        file = parquet.ParquetFile(download(name, filename))
        columns = None if include_images else [name for name in file.schema_arrow.names if name != "image"]
        for batch in file.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                yield split, row


def index_parquet(name, index):
    fields = None
    count = 0
    for split, row in parquet_rows(name):
        if fields is None:
            fields = {key: type(value).__name__ for key, value in row.items()}
            print(json.dumps({"source": name, "schema": fields}), flush=True)
        image = row["image"]
        if not isinstance(image, dict) or not image.get("bytes"):
            raise ValueError(f"{name} must provide embedded image bytes")
        source_id = str(row.get("question_id", row.get("id", count)))
        index.add(image["bytes"], name, source_id, split != "train")
        count += 1
    return count


def index_mm(index):
    required = {row["image"] for row in json_lines(download("mm-rlhf", "data.jsonl")) if "image" in row}
    count = 0
    for filename in source_files("mm-rlhf"):
        with zipfile.ZipFile(download("mm-rlhf", filename)) as archive:
            for member in archive.infolist():
                name = member.filename.removeprefix("./")
                if name not in required:
                    continue
                if member.file_size > 64 << 20 or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                    raise ValueError("Invalid image archive member")
                index.add(archive.read(member), "mm-rlhf", name)
                count += 1
    missing = [name for name in required if index.db.execute(
        "SELECT 1 FROM references_ WHERE source='mm-rlhf' AND source_id=?", (name,)).fetchone() is None]
    if missing:
        raise ValueError(f"MM-RLHF has {len(missing)} missing indexed images")
    return count


def image_roles(index, groups, heldout):
    mm_groups = {groups[digest] for (digest,) in index.db.execute(
        "SELECT DISTINCT digest FROM references_ WHERE source='mm-rlhf'") if digest in groups}
    return {group: "test" if group in heldout else group_role(
        group, "mm-rlhf" if group in mm_groups else "unowned", "train")
        for group in set(groups.values())}


def visual_record(index, groups, heldout, source, split, source_id, question, candidates, target,
                  primitive="choice", roles=None, **metadata):
    digest, path = index.resolve(source, source_id)
    group = groups[digest]
    row = scoped_record(source, split, group, [], question, candidates, target, primitive,
                        image_path=path, image_sha256=digest, source_image_id=str(source_id), **metadata)
    row["modality"] = "image"
    # A cluster belongs to exactly one role across corpora. Official held-outs win.
    row["role"] = roles[group] if roles is not None else (
        "test" if group in heldout else group_role(group, source, "train"))
    row["official_holdout_overlap"] = group in heldout and split == "train"
    if source == "vl-rewardbench":
        row["role"] = "test"
        # Distinct official IDs must survive even if two prompts render alike.
        row["id"] = stable_id(row["id"], row["source_test_example_id"])
    return validate_scope_row(row)


def aok_records(row, split, emit):
    """Historical adapter only; no active acquisition or preparation routes here."""
    choices, gold = row["choices"], row.get("correct_choice_idx")
    if gold is None or not 0 <= gold < len(choices):
        return
    candidates = [{"id": str(i), "text": value} for i, value in enumerate(choices)]
    target = [float(i == gold) for i in range(len(choices))]
    source_id = row["question_id"]
    yield emit("aokvqa", split, source_id, row["question"], candidates, target,
               adaptation="multiple_choice", coco_image_id=row.get("image_id"))
    for i, candidate in enumerate(choices):
        yes = float(i == gold)
        yield emit("aokvqa", split, source_id,
                   f"Question: {row['question']}\nIs this answer correct? {candidate}",
                   [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
                   [yes, 1 - yes], "noul", adaptation="derived_candidate_correctness",
                   coco_image_id=row.get("image_id"))


class MalformedRankError(ValueError):
    pass


def mm_records(row, emit):
    if "image" not in row:
        return
    responses, ranks = row["models_output"], row["final_ranking"]
    if (len(responses) != len(ranks) or not 2 <= len(responses) <= 16
            or any(not isinstance(value, str) for value in responses)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1 for value in ranks)):
        raise MalformedRankError("Malformed MM-RLHF ranked response annotation")
    best = min(ranks)
    target = [float(value == best) / ranks.count(best) for value in ranks]
    question = f"Which response best answers the question about the image?\n{row['question']}"
    yield emit("mm-rlhf", "train", row["image"], question,
               [{"id": str(i), "text": value} for i, value in enumerate(responses)], target,
               adaptation="best_response", annotation_id=row["id"], evaluation_scope="training")
    for a, b in combinations(range(len(responses)), 2):
        target = [1., 0.] if ranks[a] < ranks[b] else [0., 1.] if ranks[a] > ranks[b] else [.5, .5]
        yield emit("mm-rlhf", "train", row["image"], question,
                   [{"id": str(a), "text": responses[a]}, {"id": str(b), "text": responses[b]}], target,
                   adaptation="pairwise_preference", annotation_id=row["id"], evaluation_scope="training")


def vl_reward_records(row, emit):
    responses, ranking = row["response"], row["human_ranking"]
    if len(responses) != 2 or ranking not in ([0, 1], [1, 0]):
        raise ValueError("Unrecognized VL-RewardBench human ranking")
    # The pinned card defines a preferred-order index list, not per-item ranks.
    target = [float(index == ranking[0]) for index in range(2)]
    yield emit("vl-rewardbench", "test", row["id"],
               f"Which response best answers the question about the image?\n{row['query']}",
               [{"id": str(index), "text": text} for index, text in enumerate(responses)],
               target, adaptation="official_heldout_preference", **vl_external_metadata(row))


def vl_external_metadata(row):
    return {"query_source": row.get("query_source"),
            "source_test_example_id": row.get("source_test_example_id", str(row["id"])),
            "source_example_id": str(row["id"]),
            "source_response_pair_sha256": stable_id(row["query"], json.dumps(row["response"], ensure_ascii=True)),
            "source_original_split": "test", "evaluation_scope": "external",
            "sampling_weight": 1., "inclusion_probability": 1., "sampling_design": "census",
            "label_semantics": "relative_pairwise_preference"}


def vl_reward_noul_records(row, emit):
    responses, ranking = row["response"], row["human_ranking"]
    if len(responses) != 2 or ranking not in ([0, 1], [1, 0]):
        raise ValueError("Unrecognized VL-RewardBench human ranking")
    yes = float(ranking[0] == 0)
    yield emit("vl-rewardbench", "test", row["id"],
               f"Question about the image: {row['query']}\n"
               f"Response A: {responses[0]}\nResponse B: {responses[1]}\n"
               "Is response A preferred to response B for this question about the image?",
               [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
               [yes, 1 - yes], "noul", adaptation="pairwise_preference_yes_no",
               **vl_external_metadata(row))


def indexed_vl_rows(index):
    """Verify every pinned official example is indexed before allowing MM fit."""
    seen = set()
    for split, row in parquet_rows("vl-rewardbench", include_images=False):
        if split != "test" or row.get("id") in (None, ""):
            raise ValueError("VL-RewardBench requires original test example IDs")
        index.resolve("vl-rewardbench", row["id"])
        # Official image/question IDs repeat for different response pairs.
        # Preserve every pair, without altering its image lookup or response order.
        example = stable_id(str(row["id"]), vl_external_metadata(row)["source_response_pair_sha256"])
        if example in seen:
            raise ValueError("Duplicate official VL-RewardBench response-pair identity")
        seen.add(example)
        yield {**row, "source_test_example_id": example}
    if not seen:
        raise ValueError("Cannot protect VL-RewardBench heldouts without official test examples")


def index_images():
    index = ImageIndex()
    try:
        return {"vl-rewardbench": index_parquet("vl-rewardbench", index), "mm-rlhf": index_mm(index)}
    finally:
        index.close()


def prepare_images():
    index = ImageIndex(readonly=True)
    try:
        return _prepare_images(index)
    finally:
        index.close()


def _prepare_images(index):
    store = Store()
    store.update("data.splits.images", "in_progress", "Preparing MM-RLHF with active-source holdout-first image groups.")
    vl_count = sum(1 for _ in indexed_vl_rows(index))
    groups, heldout = index.groups(active_sources=ACTIVE_IMAGE_SOURCES)
    roles = image_roles(index, groups, heldout)
    emit = lambda *args, **kwargs: visual_record(index, groups, heldout, *args, roles=roles, **kwargs)
    counts, exclusions = Counter(), Counter()
    path = project_path("data/prepared/images.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.with_suffix(".tmp").open("w") as output:
        def write(row):
            validate_scope_row(row, "images.jsonl")
            if row["official_holdout_overlap"]:
                exclusions["official_holdout_overlap"] += 1
                return
            if row["id"] in seen:
                exclusions["exact_duplicate"] += 1
                return
            seen.add(row["id"])
            counts[f"{row['source']}/{row['primitive']}/{row['role']}"] += 1
            output.write(json.dumps(row, allow_nan=False) + "\n")

        for row in json_lines(download("mm-rlhf", "data.jsonl")):
            try:
                for adapted in mm_records(row, emit):
                    write(adapted)
            except MalformedRankError:
                exclusions["malformed_mm_rank_record"] += 1
    path.with_suffix(".tmp").replace(path)
    summary = {"sources": ["mm-rlhf"], "records": len(seen), "counts": dict(counts), "exclusions": dict(exclusions),
               "dataset_scope_revision": SCOPE_REVISION, "evaluation_scope": "training",
               "grouping_sources": list(ACTIVE_IMAGE_SOURCES), "protected_vl_test_examples": vl_count,
               "image_groups": len(set(groups.values())), "heldout_groups": len(heldout),
               "perceptual_distance": 4, "seed": 20261001,
               "vl_rewardbench": "official test-only preferences; preferred-order indices verified against pinned card",
               }
    digest = hashlib.sha256()
    with path.open("rb") as output:
        while chunk := output.read(1 << 20):
            digest.update(chunk)
    summary.update({"path": str(path), "sha256": digest.hexdigest()})
    write_json(project_path("data/manifests/image-splits.json"), summary)
    store.record_result("data.splits", summary)
    store.update("data.splits.images", "completed", "Image clusters, exclusive heldouts and verified visual preference adapters prepared.")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def prepare_vl_rewardbench():
    index = ImageIndex(readonly=True)
    try:
        groups, heldout = index.groups(active_sources=ACTIVE_IMAGE_SOURCES)
        roles = image_roles(index, groups, heldout)
        emit = lambda *args, **kwargs: visual_record(index, groups, heldout, *args, roles=roles, **kwargs)
        choice, noul = [], []
        for row in indexed_vl_rows(index):
            choice.extend(vl_reward_records(row, emit))
            noul.extend(vl_reward_noul_records(row, emit))
    finally:
        index.close()
    ids = [row["source_test_example_id"] for row in choice]
    if ids != [row["source_test_example_id"] for row in noul] or len(set(ids)) != len(ids):
        raise ValueError("VL-RewardBench adapters must use exactly the same unique official examples")
    for row in choice + noul:
        row.update(sampling_cluster=row["group"], population_size=len(ids), sample_size=len(ids))
    summary = {"dataset_scope_revision": SCOPE_REVISION, "evaluation_scope": "external",
               "original_split": "test", "sources": ["vl-rewardbench"],
               "source_test_examples": len(ids), "source_test_example_ids_sha256": stable_id(*ids),
               "example_identity": "sha256(original ID + response-pair checksum); raw ID retained",
               "sampling_design": "census", "label_semantics": "relative_pairwise_preference",
               "grouping_sources": list(ACTIVE_IMAGE_SOURCES), "perceptual_distance": 4,
               }
    result = {}
    for primitive, rows in (("choice", choice), ("noul", noul)):
        name = f"vl-rewardbench-{primitive}"
        result[primitive] = prepared_records(
            lambda: iter(rows), project_path(f"data/prepared/{name}.jsonl"),
            project_path(f"data/manifests/{name}-splits.json"), expected_records=len(ids),
            filename=f"{name}.jsonl", details=summary)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acquire", "index", "prepare", "prepare-external"])
    parser.add_argument("--names", nargs="+", choices=ACTIVE_IMAGE_SOURCES,
                        default=list(ACTIVE_IMAGE_SOURCES))
    args = parser.parse_args()
    if args.command == "acquire":
        acquire_images(args.names)
    elif args.command == "index":
        index_images()
    elif args.command == "prepare-external":
        prepare_vl_rewardbench()
    else:
        prepare_images()


if __name__ == "__main__":
    main()
