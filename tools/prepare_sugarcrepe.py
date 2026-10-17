"""Prepare a grouped SugarCrepe attribute-contrast sample without model execution."""

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen


SOURCES = {
    "examples.jsonl": "https://huggingface.co/datasets/Mayfull/sugarcrepe_vlms/resolve/60cca7eb6fde729093e41555bc0c5b6a930e0212/examples.jsonl",
    "replace_att.json": "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/6e45274de98440ad8ff4207bddba0afcc1a4526f/data/replace_att.json",
    "SUGARCREPE-LICENSE": "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/6e45274de98440ad8ff4207bddba0afcc1a4526f/LICENSE",
    "HF-README.md": "https://huggingface.co/datasets/Mayfull/sugarcrepe_vlms/resolve/60cca7eb6fde729093e41555bc0c5b6a930e0212/README.md",
}
ROLES = ("fit", "development", "test")
DOWNLOAD_TIMEOUT = 30
SOURCE_REVISIONS = {"Mayfull/sugarcrepe_vlms": "60cca7eb6fde729093e41555bc0c5b6a930e0212",
                    "RAIVNLab/sugar-crepe": "6e45274de98440ad8ff4207bddba0afcc1a4526f"}


def fetch(url, path):
    if not path.exists():
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            raw = response.read()
        with path.open("xb") as handle:
            handle.write(raw)


def changed_span(positive, negative):
    a, b = [list(re.finditer(r"\S+", text)) for text in (positive, negative)]
    if len(a) != len(b):
        return None
    changed = [(x, y) for x, y in zip(a, b) if x.group() != y.group()]
    if len(changed) != 1:
        return None
    x, y = changed[0]
    if (positive[:x.start()] != negative[:y.start()] or positive[x.end():] != negative[y.end():]
            or re.sub(r"\W", "", x.group()).casefold() == re.sub(r"\W", "", y.group()).casefold()):
        return None
    return {name: {"text": match.group(), "start": match.start(), "end": match.end()}
            for name, match in (("positive", x), ("negative", y))}


def load_pairs(examples, official):
    upstream = {}
    for identifier, row in json.loads(official.read_text()).items():
        key = (row["filename"], row["caption"], row["negative_caption"])
        if (not re.fullmatch(r"[0-9]{12}\.jpg", key[0])
                or not all(isinstance(text, str) and text.strip() for text in key[1:])
                or key in upstream):
            raise ValueError("Malformed or duplicate official caption pair")
        upstream[key] = str(identifier)
    matched, pairs = set(), []
    with examples.open() as handle:
        for position, line in enumerate(handle):
            row = json.loads(line)
            if row["original_file_name"] != "replace_att":
                continue
            if len(row["positive_caption"]) != 1 or len(row["negative_caption"]) != 1:
                raise ValueError("Expected one positive and one negative caption")
            key = (row["image_file_name"], row["positive_caption"][0], row["negative_caption"][0])
            if key not in upstream or key in matched:
                raise ValueError("HF caption pair differs from the official source or is duplicated")
            matched.add(key)
            span = changed_span(*key[1:])
            if span:
                pairs.append({"source_example_id": upstream[key], "hf_row": position,
                              "source_image_id": key[0], "positive_caption": key[1],
                              "negative_caption": key[2], "changed_span": span})
    if matched != set(upstream):
        raise ValueError("HF and official attribute-pair coverage differ")
    return pairs, {"official_pairs": len(upstream), "single_token_pairs": len(pairs)}


def select_pairs(pairs, blocked, counts, seed):
    if set(counts) != set(ROLES) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Supply nonnegative integer image-group counts for all three roles")
    groups = defaultdict(list)
    for pair in pairs:
        coco_id = str(int(Path(pair["source_image_id"]).stem))
        if coco_id not in blocked:
            groups[coco_id].append(pair)
    images = sorted(groups, key=int)
    if sum(counts.values()) > len(images):
        raise ValueError(f"Requested {sum(counts.values())} image groups; only {len(images)} eligible")
    rng = random.Random(seed)
    rng.shuffle(images)
    selected, offset = [], 0
    for role in ROLES:
        for identifier in images[offset:offset + counts[role]]:
            pair = rng.choice(sorted(groups[identifier], key=lambda row: row["source_example_id"]))
            selected.append({**pair, "role": role, "group": f"coco:val2017:{identifier}",
                             "positive_id": str(rng.randrange(2))})
        offset += counts[role]
    return selected, len(groups)


def records(selected, image_root=None):
    for pair in selected:
        positive = pair["positive_id"]
        candidates = [{"id": str(i), "text": pair["positive_caption"] if str(i) == positive
                       else pair["negative_caption"]} for i in range(2)]
        for order in ("ab", "ba"):
            ordered = candidates if order == "ab" else candidates[::-1]
            yield {"id": f"sugarcrepe:replace_att:{pair['source_example_id']}:{order}",
                   "source": "sugarcrepe", "source_repository": "Mayfull/sugarcrepe_vlms",
                   "original_split": "test", "primitive": "choice", "modality": "image",
                   "dataset_purpose": "visual_contrast", "adaptation": "replace_att",
                   "context": [{"role": "user", "content": "Consider the supplied image."}],
                   "question": "Which caption accurately describes the image?",
                   "candidates": ordered, "target": [float(c["id"] == positive) for c in ordered],
                   "image_path": pair["source_image_id"],
                   "image_url": "http://images.cocodataset.org/val2017/" + pair["source_image_id"],
                   "order": order, "sampling_cluster": pair["group"], **pair}


def native_request(row, image_dir=Path(".")):
    from fuzev.service_contracts import ServiceDecisionRequest

    payload = base64.b64encode((image_dir / row["image_path"]).read_bytes()).decode("ascii")
    return ServiceDecisionRequest(primitive=row["primitive"], messages=row["context"],
                                  question=row["question"], candidates=row["candidates"],
                                  image="data:image/jpeg;base64," + payload)


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    for filename, url in SOURCES.items():
        fetch(url, raw / filename)
    pairs, summary = load_pairs(raw / "examples.jsonl", raw / "replace_att.json")
    blocked = {str(int(value)) for value in json.loads(args.exclude_coco_ids.read_text())}
    counts = dict(zip(ROLES, (args.fit_groups, args.dev_groups, args.eval_groups)))
    selected, eligible = select_pairs(pairs, blocked, counts, args.seed)
    all_rows = list(records(selected))
    for role in ROLES:
        with (output / f"{role}.jsonl").open("x") as handle:
            for row in all_rows:
                if row["role"] == role:
                    handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    pool = {str(int(Path(pair["source_image_id"]).stem)) for pair in pairs}
    summary.update({"created_utc": datetime.now(timezone.utc).isoformat(), "sources": SOURCES,
                    "seed": args.seed, "eligible_image_groups": eligible,
                    "excluded_coco_ids": sorted(pool & blocked, key=int),
                    "selected_image_groups": counts, "decision_rows": {k: 2 * v for k, v in counts.items()},
                    "source_revisions": SOURCE_REVISIONS,
                    "scope": "New-source attribute contrasts; image-unseen status is unverified.",
                    "use": "Fit/calibrate on FIT; select on DEV; keep EVAL untouched until frozen.",
                    "images": "Run the images command to fetch the already selected image IDs."})
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def acquire_images(args):
    from PIL import Image

    root = args.output.resolve()
    images = {}
    for role in ROLES:
        with (root / f"{role}.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                images[row["source_image_id"]] = row["image_url"]
    image_root = root / "images"
    image_root.mkdir(exist_ok=True)
    for filename, url in sorted(images.items()):
        if not re.fullmatch(r"[0-9]{12}\.jpg", filename):
            raise ValueError("Invalid COCO image filename")
        path = image_root / filename
        fetch(url, path)
        with Image.open(path) as image:
            image.verify()
        print(filename, flush=True)
    print(json.dumps({"verified_images": len(images),
                      "image_bytes": sum((image_root / filename).stat().st_size for filename in images)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-coco-ids", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fit-groups", type=int)
    parser.add_argument("--dev-groups", type=int)
    parser.add_argument("--eval-groups", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if (args.exclude_coco_ids is None or args.seed is None
                or any(n is None or n < 0 for n in (args.fit_groups, args.dev_groups, args.eval_groups))):
            parser.error("prepare requires exclusion COCO IDs, seed and nonnegative group counts")
        prepare(args)
    else:
        acquire_images(args)


if __name__ == "__main__":
    main()
