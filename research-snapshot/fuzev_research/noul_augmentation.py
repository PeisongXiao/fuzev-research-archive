"""Preparation-only, immutable-parent relative-preference Noul augmentation."""

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from .paths import project_path


REVISION = "2026-10-05-v3-noul-preference"
ADAPTER_VERSION = "noul-relative-preference-v1"
SEED = 20261005
LABEL_SEMANTICS = "relative_pairwise_preference"
ROLE_BUDGETS = {"fit": 4096, "development": 512, "calibration": 512}
SOURCES = ("helpsteer2", "helpsteer3", "mm-rlhf")
STRATA = ("yes", "no", "tie")
BUDGETS = {role: {"helpsteer2": size // 4, "helpsteer3": size // 4, "mm-rlhf": size // 2}
           for role, size in ROLE_BUDGETS.items()}
OUTPUTS = {"text": "noul-preference-text.jsonl", "image": "noul-preference-images.jsonl"}
YES_NO = [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]
MM_QUESTION_PREFIX = "Which response best answers the question about the image?\n"
MAX_LINE_BYTES = 8 << 20


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _hash(*values):
    return hashlib.sha256(_json_bytes(values)).hexdigest()


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _frame_row(parent, modality=None):
    if not isinstance(parent, dict) or parent.get("source") not in SOURCES:
        raise ValueError("Only canonical HelpSteer2/3 and MM-RLHF parents are allowed")
    if parent.get("role") not in (*ROLE_BUDGETS, "test"):
        raise ValueError("Unknown parent split ownership")
    if any(not _nonempty(parent.get(key)) for key in ("id", "group", "revision", "original_split")):
        raise ValueError("Missing parent identity or original split")
    if (parent.get("evaluation_scope", "training") != "training"
            or parent.get("dataset_purpose", "training") != "training"):
        raise ValueError("External parents cannot enter augmentation")
    if parent["original_split"] != "train" and parent["role"] != "test":
        raise ValueError("Official heldouts cannot be promoted to training roles")
    expected = "image" if parent["source"] == "mm-rlhf" else "text"
    if parent.get("modality") != expected or modality is not None and modality != expected:
        raise ValueError("Parent source/modality mismatch")
    if parent["source"] == "mm-rlhf" and parent.get("official_holdout_overlap") is not False:
        raise ValueError("MM-RLHF parents must retain their holdout-overlap exclusion")
    return parent


def _parent_contract(parent):
    _frame_row(parent)
    if parent["role"] == "test":
        raise ValueError("TEST parents are never augmentation parents")
    if parent.get("primitive") != "choice":
        raise ValueError("Only relative-preference Choice parents are supported")
    candidates, target, context = parent.get("candidates"), parent.get("target"), parent.get("context")
    if (not isinstance(candidates, list) or len(candidates) != 2
            or any(not isinstance(c, dict) or not _nonempty(c.get("id"))
                   or not isinstance(c.get("text"), str) for c in candidates)
            or candidates[0]["id"] == candidates[1]["id"]):
        raise ValueError("Preference parents require two exact ordered responses")
    if (not isinstance(target, list) or len(target) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in target)
            or target not in ([1., 0.], [0., 1.], [.5, .5])):
        raise ValueError("Preference targets must preserve Yes/No or tied gold labels")
    if (not isinstance(context, list) or any(not isinstance(m, dict)
            or m.get("role") not in ("user", "assistant", "system")
            or not isinstance(m.get("content"), str) for m in context)):
        raise ValueError("Unsupported parent context")
    if parent["source"] == "mm-rlhf":
        if (parent.get("adaptation") != "pairwise_preference"
                or not _nonempty(parent.get("image_path"))
                or not _sha256(parent.get("image_sha256"))
                or not _nonempty(parent.get("source_image_id"))
                or (not _nonempty(parent.get("annotation_id")) and type(parent.get("annotation_id")) is not int)):
            raise ValueError("Only admitted MM-RLHF pairwise image parents are supported")
        if not isinstance(parent.get("question"), str) or not parent["question"].startswith(MM_QUESTION_PREFIX):
            raise ValueError("Unrecognized canonical MM-RLHF question template")
    elif parent.get("question") != "Which response better answers the user's request?":
        raise ValueError("Unrecognized canonical HelpSteer preference question")


def _input_key(parent):
    # Canonical parent IDs contain targets, so cannot determine label-independent orientation.
    return _hash({key: parent.get(key) for key in
                  ("source", "revision", "group", "context", "question", "candidates", "image_sha256")})


def _orientation(parent, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Orientation seed must be a nonnegative integer")
    swap = int(_hash("orientation", ADAPTER_VERSION, seed, _input_key(parent)), 16) & 1
    return [1, 0] if swap else [0, 1]


def _question(parent, permutation):
    a, b = (parent["candidates"][index]["text"] for index in permutation)
    if parent["source"] == "mm-rlhf":
        query = parent["question"][len(MM_QUESTION_PREFIX):]
        prefix = f"Question about the image: {query}\n"
        question = "Is response A preferred to response B for this question about the image?"
    else:
        prefix = ""
        question = "Is response A preferred to response B for this request?"
    return f"{prefix}Response A: {a}\nResponse B: {b}\n{question}"


def augment_parent(parent, *, seed=SEED):
    """Return a new view; never change the parent, its context, image or ownership."""
    _parent_contract(parent)
    permutation = _orientation(parent, seed)
    target = [parent["target"][index] for index in permutation]
    result = copy.deepcopy(parent)
    result.update(id=_hash(ADAPTER_VERSION, seed, parent["id"], permutation),
                  primitive="noul", question=_question(parent, permutation),
                  candidates=copy.deepcopy(YES_NO), target=target,
                  adaptation="pairwise_preference_yes_no", label_semantics=LABEL_SEMANTICS,
                  adapter_version=ADAPTER_VERSION, orientation_seed=seed,
                  candidate_permutation=permutation, parent_id=parent["id"],
                  parent_role=parent["role"], parent_source=parent["source"],
                  parent_candidate_ids=[candidate["id"] for candidate in parent["candidates"]],
                  parent_record_sha256=hashlib.sha256(_json_bytes(parent)).hexdigest(),
                  parent_input_sha256=_input_key(parent), parent_primitive=parent["primitive"],
                  parent_adaptation=parent.get("adaptation"), parent_question=parent["question"],
                  parent_candidates=copy.deepcopy(parent["candidates"]),
                  parent_target=copy.deepcopy(parent["target"]),
                  dataset_scope_revision=REVISION, dataset_purpose="training", evaluation_scope="training")
    return result


@dataclass(frozen=True)
class _Candidate:
    source: str
    role: str
    group: str
    stratum: str
    parent_id: str
    input_key: str
    path: Path
    offset: int
    length: int
    line_sha256: str


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _manifest(root, modality):
    stem, manifest_name = ("text", "text-splits.json") if modality == "text" else ("images", "image-splits.json")
    path = root / "data/prepared" / (stem + ".jsonl")
    manifest_path = root / "data/manifests" / manifest_name
    if path.resolve().parent != (root / "data/prepared").resolve():
        raise ValueError("Parent path escapes canonical prepared directory")
    with manifest_path.open("rb") as handle:
        raw = handle.read(1 << 20)
        if handle.read(1):
            raise ValueError("Parent manifest is oversized")
    value = json.loads(raw)
    if (not isinstance(value, dict) or not _sha256(value.get("sha256"))
            or type(value.get("records")) is not int or value["records"] <= 0
            or not isinstance(value.get("counts"), dict)
            or Path(value.get("path", "")).resolve() != path.resolve()):
        raise ValueError("Invalid canonical parent manifest")
    if modality == "image":
        policy = value.get("dimension_eligibility", {})
        required = {"min_axis_pixels": 32, "max_aspect_ratio": 200, "boundary": "inclusive",
                    "resized_images": False, "raw_images_deleted": False}
        if any(policy.get(key) != expected for key, expected in required.items()):
            raise ValueError("Original-shape image eligibility must be attested by the parent manifest")
    provenance = {"path": str(path.resolve()), "sha256": value["sha256"], "records": value["records"],
                  "manifest_path": str(manifest_path.resolve()),
                  "manifest_sha256": hashlib.sha256(raw).hexdigest()}
    return path, value, provenance, _stamp(manifest_path)


def _scan(root, seed):
    owners, seen_ids, revisions, unique = {}, set(), {}, {}
    provenances, stamps = {}, {}
    for modality in OUTPUTS:
        path, manifest, provenance, manifest_stamp = _manifest(root, modality)
        provenances[modality] = provenance
        stamps[path] = _stamp(path)
        stamps[Path(provenance["manifest_path"])] = manifest_stamp
        digest, counts, count = hashlib.sha256(), Counter(), 0
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("Canonical parent row exceeds bounded preparation reader")
                digest.update(line)
                parent = _frame_row(json.loads(line), modality)
                count += 1
                if parent["id"] in seen_ids:
                    raise ValueError("Duplicate canonical parent ID")
                seen_ids.add(parent["id"])
                owner = owners.setdefault(parent["group"], parent["role"])
                if owner != parent["role"]:
                    raise ValueError("Parent group spans multiple roles; refuse leakage")
                revision = revisions.setdefault(parent["source"], parent["revision"])
                if revision != parent["revision"]:
                    raise ValueError("Parent source revision is inconsistent")
                counts[f"{parent['source']}/{parent.get('primitive')}/{parent['role']}"] += 1
                if parent["role"] == "test":
                    continue
                if modality == "text" and parent.get("primitive") == "ordinal":
                    continue
                if modality == "image" and parent.get("adaptation") == "best_response":
                    continue
                view = augment_parent(parent, seed=seed)
                stratum = "tie" if view["target"] == [.5, .5] else "yes" if view["target"][0] == 1 else "no"
                candidate = _Candidate(parent["source"], parent["role"], parent["group"], stratum,
                                       parent["id"], view["parent_input_sha256"], path, offset,
                                       len(line), hashlib.sha256(line).hexdigest())
                semantic = _hash(parent["source"], parent["group"], parent["context"], view["question"], view["target"])
                old = unique.get(semantic)
                if old is None or (_hash(seed, candidate.parent_id), candidate.parent_id) < (
                        _hash(seed, old.parent_id), old.parent_id):
                    unique[semantic] = candidate
        expected = {key: value for key, value in manifest["counts"].items() if not key.startswith("excluded_")}
        if count != manifest["records"] or dict(counts) != expected or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("Canonical parent counts/hash do not match the manifest")
    _unchanged(stamps)
    populations = Counter(f"{candidate.source}/{candidate.role}/{candidate.stratum}"
                          for candidate in unique.values())
    return list(unique.values()), provenances, stamps, dict(populations)


def _unchanged(stamps):
    if any(_stamp(path) != stamp for path, stamp in stamps.items()):
        raise ValueError("Canonical parents changed during preparation")


def _stratum_quotas(size, counts):
    if sum(counts.values()) < size:
        raise ValueError("Insufficient eligible parent views for the approved budget")
    present = [label for label in STRATA if counts[label]]
    quotas = {label: int(label in present and size >= len(present)) for label in STRATA}
    remaining = size - sum(quotas.values())
    weights = {label: counts[label] - quotas[label] for label in STRATA}
    population = sum(weights.values())
    if remaining:
        for label in STRATA:
            quotas[label] += remaining * weights[label] // population
        spare = size - sum(quotas.values())
        order = sorted(STRATA, key=lambda label: (-(remaining * weights[label] % population), STRATA.index(label)))
        for label in order:
            if spare and quotas[label] < counts[label]:
                quotas[label] += 1
                spare -= 1
    return quotas


def _select(candidates, role_budgets, seed):
    if set(role_budgets) != set(ROLE_BUDGETS) or any(
            type(size) is not int or size <= 0 or size % 4 for size in role_budgets.values()):
        raise ValueError("Role budgets must retain all three roles and half/quarter/quarter source shares")
    pools = defaultdict(list)
    for candidate in candidates:
        pools[candidate.role, candidate.source, candidate.stratum].append(candidate)
    selected = []
    for role, size in role_budgets.items():
        for source in SOURCES:
            budget = size // (2 if source == "mm-rlhf" else 4)
            counts = {label: len(pools[role, source, label]) for label in STRATA}
            for label, quota in _stratum_quotas(budget, counts).items():
                groups = defaultdict(list)
                for candidate in pools[role, source, label]:
                    groups[candidate.group].append(candidate)
                ordered = deque(deque(sorted(groups[group], key=lambda c: (_hash(seed, c.input_key), c.parent_id)))
                                for group in sorted(groups, key=lambda group: (_hash(seed, source, label, group), group)))
                for _ in range(quota):
                    group = ordered.popleft()
                    selected.append(group.popleft())
                    if group:
                        ordered.append(group)
    return sorted(selected, key=lambda c: (c.role, c.source, c.parent_id))


def prepare(root=None, *, seed=SEED, role_budgets=None):
    """Publish only the two new preparations/manifests, exclusively; no runtime jobs."""
    root = Path(root).expanduser().resolve() if root is not None else project_path("")
    role_budgets = dict(ROLE_BUDGETS if role_budgets is None else role_budgets)
    targets = {modality: (root / "data/prepared" / filename,
                          root / "data/manifests" / (Path(filename).stem + "-splits.json"))
               for modality, filename in OUTPUTS.items()}
    if any(path.exists() or path.is_symlink() for pair in targets.values() for path in pair):
        raise FileExistsError("Augmentation outputs already exist; refuse overwrite")
    candidates, provenances, stamps, populations = _scan(root, seed)
    selected = _select(candidates, role_budgets, seed)
    budgets = {role: {source: size // (2 if source == "mm-rlhf" else 4) for source in SOURCES}
               for role, size in role_budgets.items()}
    manifests, published = {}, []
    try:
        with tempfile.TemporaryDirectory(prefix=".noul-preference-", dir=root / "data") as temporary:
            staged = []
            for modality, (destination, manifest_path) in targets.items():
                output = Path(temporary) / destination.name
                counts, labels, groups, digest, records = Counter(), Counter(), set(), hashlib.sha256(), 0
                with output.open("xb") as writer:
                    for candidate in selected:
                        if (candidate.source == "mm-rlhf") != (modality == "image"):
                            continue
                        with candidate.path.open("rb") as reader:
                            reader.seek(candidate.offset)
                            line = reader.read(candidate.length)
                        if hashlib.sha256(line).hexdigest() != candidate.line_sha256:
                            raise ValueError("Selected parent bytes changed during preparation")
                        parent = json.loads(line)
                        view = augment_parent(parent, seed=seed)
                        view.update(parent_path=str(candidate.path.resolve()), parent_line_sha256=candidate.line_sha256,
                                    parent_preparation_sha256=provenances[modality]["sha256"])
                        raw = _json_bytes(view) + b"\n"
                        writer.write(raw)
                        digest.update(raw)
                        counts[f"{view['source']}/noul/{view['role']}"] += 1
                        labels[f"{view['source']}/{view['role']}/{candidate.stratum}"] += 1
                        groups.add(view["group"])
                        records += 1
                manifest = {"records": records, "counts": dict(counts), "groups": len(groups),
                            "sha256": digest.hexdigest(), "path": str(destination.resolve()),
                            "sources": ["mm-rlhf"] if modality == "image" else ["helpsteer2", "helpsteer3"],
                            "dataset_scope_revision": REVISION, "dataset_purpose": "training",
                            "evaluation_scope": "training", "purpose": "training",
                            "adapter_version": ADAPTER_VERSION, "label_semantics": LABEL_SEMANTICS,
                            "seed": seed, "budgets": budgets, "label_counts": dict(labels),
                            "parent_preparations": [provenances[modality]],
                            "selection": {"method": "group-round-robin-label-stratified-v1",
                                          "allocation": "proportional largest remainder; nonempty strata retained when budget permits",
                                          "eligible_parent_label_counts": populations}}
                if modality == "image":
                    manifest["dimension_eligibility"] = {"min_axis_pixels": 32, "max_aspect_ratio": 200,
                        "boundary": "inclusive", "resized_images": False, "raw_images_deleted": False,
                        "preserved_from_parent_manifest_sha256": provenances[modality]["manifest_sha256"]}
                manifest_output = Path(temporary) / manifest_path.name
                manifest_output.write_bytes(_json_bytes(manifest) + b"\n")
                manifests[destination.name] = manifest
                staged.extend(((output, destination), (manifest_output, manifest_path)))
            _unchanged(stamps)
            # Hard-link publication is atomic and refuses existing names, unlike replace().
            for origin, destination in sorted(staged, key=lambda pair: pair[1].suffix != ".jsonl"):
                os.link(origin, destination)
                published.append((destination, origin.stat().st_dev, origin.stat().st_ino))
    except BaseException:
        for path, device, inode in published:
            try:
                metadata = path.stat()
            except FileNotFoundError:
                continue
            if (metadata.st_dev, metadata.st_ino) == (device, inode):
                path.unlink()
        raise
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="FUZEV_ROOT containing canonical prepared parents")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
