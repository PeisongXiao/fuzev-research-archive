"""Offline final-test evaluation. No backbone, fitting, calibration or promotion."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
import hashlib
from itertools import islice, zip_longest
import json
import math
from pathlib import Path
import random

import numpy as np

from . import external_sampling
from .evaluation_metrics import MAX_RECORDS, METHODS, compare_tracks, gate
from .features import SCOPE_METADATA, FeatureStore, collate, file_sha256
from .metrics import per_record, probabilities
from .paths import project_path
from .scope import EXTERNAL_SOURCES, REVISION, TRAINING_SOURCES, required_preparations, validate_contract, validate_row


REQUIRED_CELLS = ("native-3.05", "native-4.05", "native-5.05", "vllm-nvidia",
                  "vllm-radixark", "vllm-exl3-4.05")
HEAD_FILES = ("head.safetensors", "manifest.json", "calibration.json")
TEST_SCOPE = "evaluation-test-only"
SUBSET_FILE = "subset-extraction.json"
LEGACY_SEMANTICS = ("id", "group", "role", "primitive", "modality", "source", "revision",
                    "candidates", "target", "adaptation", "attribute")
SEMANTICS = tuple(dict.fromkeys((*LEGACY_SEMANTICS, *SCOPE_METADATA,
                                "image_sha256", "image_path", "source_image_id", "label_semantics",
                                "source_paper_id", "source_question_ids", "sampling_label", "sampling_seed",
                                "inclusion_probability_numerator", "inclusion_probability_denominator", "question")))
LIMITS = {"test_records": MAX_RECORDS, "batch_size": 256,
          "feature_cache_bytes": 64 << 20, "shard_bytes": 256 << 20,
          "metadata_bytes": 64 << 20}


def dataset_scope(features):
    identity = getattr(features, "identity", {})
    return validate_contract(identity["dataset_scope"]) if "dataset_scope" in identity else None


def semantic_fields(meta):
    # Imported HelpSteer metadata is byte-preserved, not retroactively enriched.
    # Its immutable prepared-file hash and producer receipt bind the full input.
    if meta["source"] in {"helpsteer2", "helpsteer3"} or "evaluation_scope" not in meta:
        return LEGACY_SEMANTICS
    return SEMANTICS


def read_json(path):
    path = Path(path)
    if path.stat().st_size > LIMITS["metadata_bytes"]:
        raise MemoryError(f"Metadata resource bound exceeded: {path.name}")
    return json.loads(path.read_text())


def sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def check_hashes(directory, expected):
    if set(expected) != set(HEAD_FILES):
        raise ValueError("Pin the weights, manifest and calibration checksums before final evaluation")
    actual = {name: file_sha256(Path(directory) / name) for name in HEAD_FILES}
    if actual != expected:
        raise ValueError("Frozen reference checksum mismatch")
    return actual


def open_features(directory, expected_sha256):
    directory = Path(directory)
    if file_sha256(directory / "identity.json") != expected_sha256:
        raise ValueError("Pinned feature identity checksum mismatch")
    return FeatureStore(directory, read_json(directory / "identity.json"), read_only=True,
                        cache_bytes=LIMITS["feature_cache_bytes"])


def require_complete(features):
    """Require full active-training readiness, including images, never a pilot."""
    if not features.read_only:
        raise ValueError("Final evaluation requires a read-only feature corpus")
    if features.identity.get("corpus_scope") not in (None, "full"):
        raise ValueError("Reference readiness requires a full corpus, not evaluation-only features")
    dataset_scope(features)
    readiness = features.require_ready("calibrate")
    require_feature_ownership(features)
    return readiness


def require_feature_ownership(features):
    """Check committed storage and ownership without asserting training readiness."""
    if not features.read_only:
        raise ValueError("Final evaluation requires a read-only feature corpus")
    if dataset_scope(features) is not None:
        # Include non-test external rows and forged external-purpose training rows
        # in the audit rather than allowing the test-role SELECT to hide them.
        for row in external_rows(features):
            metadata(dict(row), scoped=True)
    if features.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]:
        raise ValueError("Extraction failures remain in the comparison corpus")
    identity = features.identity
    required = ("engine", "model", "revision", "encoding", "datasets", "runtime_sources",
                "readout_runtime", "cache_precision", "feature_tap", "hidden_size")
    if any(not identity.get(key) for key in required):
        raise ValueError("Incomplete feature execution provenance")
    if features.db.execute("SELECT group_ FROM records GROUP BY group_ HAVING COUNT(DISTINCT role)>1 LIMIT 1").fetchone():
        raise ValueError("Training/calibration/test group leakage")
    if features.db.execute("SELECT r.id FROM records r LEFT JOIN groups_ g ON r.group_=g.id "
                           "WHERE g.id IS NULL OR r.role!=g.role LIMIT 1").fetchone():
        raise ValueError("Feature group ownership mismatch")
    if features.db.execute("SELECT shard,row_ FROM records GROUP BY shard,row_ HAVING COUNT(*)>1 LIMIT 1").fetchone():
        raise ValueError("Duplicate feature row location")
    shards = []
    for shard, count, first, last in features.db.execute(
            "SELECT shard,COUNT(*),MIN(row_),MAX(row_) FROM records GROUP BY shard ORDER BY shard"):
        if shard < 0 or first != 0 or last != count - 1:
            raise ValueError("Incomplete feature shard index")
        for suffix, bound in (("safetensors", LIMITS["shard_bytes"]), ("json", LIMITS["metadata_bytes"])):
            path = features.directory / f"shard-{shard:06d}.{suffix}"
            if not path.is_file():
                raise ValueError(f"Missing committed feature shard: {path.name}")
            if path.stat().st_size > bound:
                raise MemoryError(f"Feature shard resource bound exceeded: {path.name}")
        shards.append({"shard": shard, "records": count})
    count = features.db.execute("SELECT COUNT(*) FROM records WHERE role='test'").fetchone()[0]
    if not count:
        raise ValueError("No exclusively test-role records")
    if count > MAX_RECORDS:
        raise MemoryError("Final evaluation test-record bound exceeded; no subsampling is permitted")
    return shards


def load_reference(features, directory, expected):
    from .head import load_bundle
    from .train import membership_identity

    check_hashes(directory, expected)
    readiness = require_complete(features)
    bundle, manifest, temperatures = load_bundle(directory)
    if (manifest.get("artifact_kind") != "calibrated-reference-head" or manifest.get("stage") != "mixed"
            or manifest.get("calibration_role") != "calibration" or manifest.get("precision") != "float32"
            or manifest.get("feature_identity") != features.identity or manifest.get("preparation") != readiness):
        raise ValueError("Require a frozen, full-corpus calibrated reference head with matching provenance")
    require_reference_origins(features, manifest)
    for role, field in (("fit", "fit_membership"), ("development", "development_membership"),
                        ("calibration", "calibration_membership")):
        actual = membership_identity(features.iter_records(role))
        if not actual["records"] or manifest.get(field) != actual:
            raise ValueError(f"Frozen reference {role} membership mismatch")
    for field in ("training_code", "calibration_code"):
        hashes = manifest.get(field, {})
        if manifest.get("architecture") in {"shared-candidate-dot-product-v1", "shared-lm-assisted-candidate-dot-product-v1",
                                           "shared-lm-assisted-candidate-dot-product-rank512-v1",
                                           "shared-lm-assisted-context-trust-candidate-dot-product-rank512-v1",
                                           "shared-lm-assisted-query-adapter-candidate-dot-product-rank512-v1",
                                           "shared-lm-assisted-query-channel-gate-candidate-dot-product-rank512-v1",
                                           "shared-lm-assisted-query-channel-gate-adapter64-candidate-dot-product-rank512-v1",
                                           "shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1"}:
            from .shared_head import CODE_FILES
            valid_files = set(hashes) == CODE_FILES
        else:
            valid_files = set(hashes) == {"train.py", "features.py", "head.py", "metrics.py"}
        if not valid_files or any(
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest) for digest in hashes.values()):
            raise ValueError("Missing frozen training/calibration code provenance")
    if manifest.get("architecture") in {"shared-candidate-dot-product-v1", "shared-lm-assisted-candidate-dot-product-v1",
                                       "shared-lm-assisted-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-context-trust-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-adapter-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-channel-gate-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-channel-gate-adapter64-candidate-dot-product-rank512-v1",
                                       "shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1"}:
        from .shared_qualification import validate_shared_reference
        validate_shared_reference(bundle, manifest, temperatures, features=features, corpus=True)
    else:
        from .composite_head import validate_composite
        validate_composite(bundle, manifest, temperatures, features=features, corpus=True)
    bundle.requires_grad_(False)
    check_hashes(directory, expected)
    return bundle, manifest, temperatures


def require_reference_origins(features, manifest):
    """Producer receipts belong to the frozen head, not cross-engine readiness."""
    if dataset_scope(features) is not None:
        origins = getattr(features, "producer_origins", None)
        if (not callable(origins) or "feature_origins" not in manifest
                or manifest["feature_origins"] != origins()):
            raise ValueError("Frozen reference feature producer origins mismatch or missing provenance")


def test_rows(features):
    for row in features.db.execute("SELECT * FROM records WHERE role='test' ORDER BY id"):
        yield dict(row)


def metadata(row, *, scoped=False):
    meta = json.loads(row["metadata"])
    if meta.get("role") != "test" or row["role"] != "test":
        raise ValueError("Scoring is restricted to test roles")
    if (any(meta.get(key) != row[key] for key in ("id", "role", "primitive", "modality", "source"))
            or meta.get("group") != row["group_"]):
        raise ValueError("Feature index and semantic metadata disagree")
    if (meta["primitive"] not in {"choice", "noul", "ordinal"} or meta["modality"] not in {"text", "image"}
            or any(not isinstance(meta.get(key), str) or not meta[key] for key in ("id", "group", "source", "revision"))):
        raise ValueError("Invalid task/group identity")
    candidates = meta["candidates"]
    # Official HelpSteer3 preferences include empty responses; preserve their
    # bytes and labels rather than dropping heldout records after extraction.
    response_pair = (meta["source"] == "helpsteer3" and meta["primitive"] == "choice"
                     and meta["modality"] == "text" and meta.get("adaptation") is None
                     and meta.get("attribute") is None and isinstance(candidates, list)
                     and [c.get("id") if isinstance(c, dict) else None for c in candidates] == ["a", "b"])
    if (not isinstance(candidates, list) or any(not isinstance(candidate, dict) or
            not isinstance(candidate.get("id"), str) or not candidate["id"] or
            not isinstance(candidate.get("text"), str) or (not candidate["text"] and not response_pair)
            for candidate in candidates)):
        raise ValueError("Invalid candidate semantics")
    ids = [candidate["id"] for candidate in candidates]
    if not 2 <= len(ids) <= 16 or len(set(ids)) != len(ids):
        raise ValueError("Invalid or repeated candidate identifiers")
    if meta["primitive"] == "noul" and set(ids) != {"yes", "no"}:
        raise ValueError("Noul candidates must preserve yes/no meaning")
    if meta["primitive"] == "ordinal":
        values = [candidate["ordinal_value"] for candidate in candidates]
        if not np.isfinite(values).all() or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("Ordinal values must be finite and strictly increasing")
    target = meta.get("target")
    if (not isinstance(target, list) or len(target) != len(ids)
            or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0
                   for x in target) or not math.isclose(sum(target), 1., rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError("Invalid candidate target probabilities")
    if scoped:
        validate_row(meta)
        if meta["source"] in EXTERNAL_SOURCES:
            validate_external_metadata(meta)
    return meta


def validate_external_metadata(meta):
    """Reject missing design information rather than silently imputing a census."""
    validate_row(meta)
    if meta.get("dataset_scope_revision") != REVISION or meta.get("dataset_purpose") != "external":
        raise ValueError("External records require the active scope revision and external dataset purpose")
    if meta.get("attribute") is not None:
        raise ValueError("External subtype must not be hidden by an attribute track")
    if not isinstance(meta.get("source_test_example_id"), str) or not meta["source_test_example_id"]:
        raise ValueError("External records require a source test example ID")
    weight, probability = (meta.get(key) for key in ("sampling_weight", "inclusion_probability"))
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
            or value <= 0 for value in (weight, probability)) or probability > 1
            or not math.isclose(weight * probability, 1., rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("External sampling weights must be finite positive inverse inclusion probabilities")
    ids = [candidate["id"] for candidate in meta["candidates"]]
    if meta["primitive"] == "noul" and any(candidate["text"].strip().lower() != candidate["id"]
                                             for candidate in meta["candidates"]):
        raise ValueError("External Noul must retain literal Yes/No candidate semantics")
    sampled = meta["source"] == "qasper" and meta["adaptation"] == "evidence"
    if sampled:
        if not isinstance(meta.get("sampling_stratum"), str) or not meta["sampling_stratum"].strip():
            raise ValueError("Sampled QASPER evidence requires its sampling stratum")
        if not isinstance(meta.get("evidence_id"), str) or not meta["evidence_id"]:
            raise ValueError("QASPER evidence requires paragraph identity")
        if meta.get("sampling_design") != external_sampling.DESIGN or meta.get("sampling_seed") != external_sampling.SEED:
            raise ValueError("Unknown QASPER sampling design or fixed seed")
        numerator, denominator = (meta.get("inclusion_probability_" + key) for key in ("numerator", "denominator"))
        if (type(numerator) is not int or type(denominator) is not int or not 0 < numerator <= denominator
                or not math.isclose(probability, float(Fraction(numerator, denominator)), rel_tol=1e-12, abs_tol=0.)):
            raise ValueError("QASPER requires the exact marginal inclusion probability, not realized n/N")
        label = external_sampling.evidence_label([dict(zip(ids, meta["target"]))[key] for key in ("yes", "no")])
        if meta.get("sampling_label") != label or json.loads(meta["sampling_stratum"]) != [
                meta["group"], meta["source_question_id"], label]:
            raise ValueError("QASPER sampling stratum must retain paper/question/annotation-label meaning")
    else:
        if weight != 1 or probability != 1 or meta.get("sampling_design") != "census":
            raise ValueError("External census records must declare census design and unit weight/probability")
        stratum = meta["adaptation"] if meta["source"] == "qasper" else "official_test"
        if meta.get("sampling_stratum") not in (None, stratum):
            raise ValueError("Unexpected census sampling stratum")
    if meta["source"] == "qasper":
        if (meta["modality"] != "text" or not meta["group"].startswith("paper:")
                or not isinstance(meta.get("source_question_id"), str) or not meta["source_question_id"]
                or meta["group"] != "paper:" + str(meta.get("source_paper_id", ""))):
            raise ValueError("QASPER requires paper-level grouping and source question identity")
        aliases = meta.get("source_question_ids")
        if (not isinstance(aliases, list) or any(not isinstance(key, str) or not key for key in aliases)
                or len(set(aliases)) != len(aliases) or meta["source_question_id"] not in aliases):
            raise ValueError("QASPER requires preserved source question aliases after deduplication")
    else:
        digests = [meta.get(key) for key in ("image_sha256", "source_response_pair_sha256")]
        if (meta["modality"] != "image" or any(not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest) for digest in digests)
                or any(not isinstance(meta.get(key), str) or not meta[key] for key in
                       ("image_path", "source_image_id", "source_example_id", "question"))):
            raise ValueError("VL-RewardBench requires preserved image identity")
        expected_example = hashlib.sha256((meta["source_example_id"] + "\0" +
                                           meta["source_response_pair_sha256"]).encode()).hexdigest()
        if meta["source_test_example_id"] != expected_example:
            raise ValueError("VL test example ID must bind the raw ID and original response pair")
        if meta["primitive"] == "choice" and ids != ["0", "1"]:
            raise ValueError("Original VL Choice must retain response 0/1 in original order")
        if sorted(meta["target"]) != [0., 1.]:
            raise ValueError("VL-RewardBench requires the official untied pairwise preference")
        if meta.get("label_semantics") != "relative_pairwise_preference":
            raise ValueError("VL labels represent relative pairwise preference, not absolute correctness")


def external_rows(features):
    allowed = TRAINING_SOURCES + EXTERNAL_SOURCES
    return features.db.execute(
        "SELECT * FROM records WHERE source IN (?,?) OR source IS NULL OR source NOT IN (" +
        ",".join("?" for _ in allowed) + ") OR json_extract(metadata,'$.evaluation_scope')='external' ORDER BY id",
        (*EXTERNAL_SOURCES, *allowed))


def external_test_membership(features):
    """Audit the exact QASPER census and one-to-one original/adapted VL pairs."""
    scope = dataset_scope(features)
    if scope is None:
        return None
    counts, subtypes, seen, visual = Counter(), Counter(), set(), {}
    qasper_questions = {name: set() for name in ("yes_no", "answerability", "evidence")}
    qasper_aliases = {name: set() for name in qasper_questions}
    for count, row in enumerate(external_rows(features), 1):
        if count > MAX_RECORDS:
            raise MemoryError("External membership exceeds the evaluation bound")
        meta = metadata(dict(row), scoped=True)
        source, primitive = meta["source"], meta["primitive"]
        counts["/".join((meta["modality"], source, primitive, "test"))] += 1
        key = (source, meta["adaptation"], meta["source_test_example_id"], meta.get("evidence_id"),
               meta["group"] if source == "qasper" else None)
        if key in seen:
            raise ValueError("Duplicate external source test example membership")
        seen.add(key)
        if source == "qasper":
            subtype, question = meta["adaptation"], meta["source_question_id"]
            qasper_aliases[subtype].update((meta["group"], alias) for alias in meta["source_question_ids"])
            unit = (meta["group"], question, meta.get("evidence_id") if subtype == "evidence" else None)
            if unit in qasper_questions[subtype]:
                raise ValueError("Duplicate QASPER question/paragraph membership")
            qasper_questions[subtype].add(unit)
            subtypes[subtype] += 1
        else:
            pair = visual.setdefault(meta["source_test_example_id"], {})
            if primitive in pair:
                raise ValueError("Duplicate VL original/adapted membership")
            pair[primitive] = meta
    expected = {name: scope["qasper"][name] for name in qasper_questions}
    if subtypes != expected:
        raise ValueError("Incomplete external QASPER membership: require 239 yes_no, 1450 answerability, 2000 evidence")
    if any(not qasper_aliases[name] <= qasper_aliases["answerability"] for name in ("yes_no", "evidence")):
        raise ValueError("QASPER subtype has no matching paper/question alias in the answerability census")
    if not visual:
        raise ValueError("Missing original Choice and adapted Noul VL memberships")
    paired_digest, raw_images, image_groups = hashlib.sha256(), {}, {}
    for example, pair in sorted(visual.items()):
        if set(pair) != {"choice", "noul"}:
            raise ValueError("VL Choice and Noul must have exactly the same source test example IDs")
        choice, noul = pair["choice"], pair["noul"]
        fields = ("group", "revision", "image_sha256", "image_path", "source_image_id", "source_response_pair_sha256",
                  "source_example_id")
        if any(choice.get(key) != noul.get(key) for key in fields):
            raise ValueError("Paired VL original/adapted group or image identity mismatch")
        raw_id, image_digest = choice["source_example_id"], choice["image_sha256"]
        image_identity = tuple(choice[key] for key in ("group", "image_sha256", "image_path", "source_image_id"))
        if raw_images.setdefault(raw_id, image_identity) != image_identity or (
                image_groups.setdefault(image_digest, choice["group"]) != choice["group"]):
            raise ValueError("Repeated VL images must retain shared image identity and bootstrap group")
        preference = dict(zip((c["id"] for c in noul["candidates"]), noul["target"]))
        if [preference["yes"], preference["no"]] != choice["target"]:
            raise ValueError("VL Noul Yes must denote response 0 preferred, No response 1 preferred")
        prefix = "Which response best answers the question about the image?\n"
        query = choice["question"].removeprefix(prefix)
        pair_digest = hashlib.sha256((query + "\0" + json.dumps([candidate["text"] for candidate in choice["candidates"]],
                                                               ensure_ascii=True)).encode()).hexdigest()
        if pair_digest != choice["source_response_pair_sha256"]:
            raise ValueError("VL original query/response pair does not match its source pair checksum")
        expected_question = (f"Question about the image: {query}\n"
                             f"Response A: {choice['candidates'][0]['text']}\n"
                             f"Response B: {choice['candidates'][1]['text']}\n"
                             "Is response A preferred to response B for this question about the image?")
        if not choice["question"].startswith(prefix) or noul["question"] != expected_question:
            raise ValueError("VL adapter question must explicitly compare original response 0/A against 1/B")
        paired_digest.update(json.dumps([example, *(choice[key] for key in fields)],
                                         separators=(",", ":"), allow_nan=False).encode() + b"\n")
    return {"records": sum(counts.values()), "counts": dict(sorted(counts.items())),
            "qasper_subtypes": dict(sorted(subtypes.items())),
            "vl_paired_examples": len(visual), "vl_raw_example_ids": len(raw_images),
            "vl_image_groups": len(set(image_groups.values())), "vl_paired_examples_sha256": paired_digest.hexdigest()}


def validate_sampling_design(design, rows):
    """Attest marginal pi from the full frame, not from realized stratum counts."""
    if (not isinstance(design, dict) or design.get("design") != external_sampling.DESIGN
            or design.get("seed") != external_sampling.SEED or design.get("sample_size") != 2000):
        raise ValueError("Missing or unsupported QASPER sampling-design attestation")
    frame = design.get("strata")
    if not isinstance(frame, list) or not 1 <= len(frame) <= MAX_RECORDS:
        raise ValueError("Missing or unbounded full QASPER sampling frame")
    strata, papers, questions, labels = {}, Counter(), defaultdict(Counter), defaultdict(dict)
    for entry in frame:
        key = json.loads(entry["sampling_stratum"])
        if (not isinstance(key, list) or len(key) != 3 or any(not isinstance(value, str) or not value for value in key)
                or key[2] not in {"yes", "no", "mixed"}):
            raise ValueError("Malformed paper/question/label sampling frame")
        key = tuple(key)
        population, sampled = entry.get("population"), entry.get("sampled")
        if (key in strata or type(population) is not int or population <= 0
                or type(sampled) is not int or not 0 <= sampled <= population):
            raise ValueError("Invalid or duplicate QASPER sampling-frame counts")
        strata[key] = entry
        papers[key[0]] += population
        questions[key[0]][key[1]] += population
        labels[key[:2]][key[2]] = population
    if list(strata) != sorted(strata) or design.get("population") != sum(papers.values()) or (
            sum(papers.values()) <= 2000 or sum(entry["sampled"] for entry in frame) != 2000):
        raise ValueError("QASPER frame must describe the full population and exact 2000-row sample")
    paper_quotas = external_sampling._equal_quotas(papers, 2000)
    quotas = {}
    for paper, capacities in questions.items():
        for question, quota in external_sampling._equal_quotas(capacities, paper_quotas[paper]).items():
            for label, value in external_sampling._equal_quotas(labels[paper, question], quota).items():
                quotas[paper, question, label] = value
    denominator = math.lcm(*(value.denominator for value in quotas.values()))
    offset = random.Random(external_sampling.SEED).randrange(denominator)
    counts = external_sampling._systematic_counts([quotas[key] for key in strata], offset, denominator)
    if design.get("systematic_denominator") != denominator or design.get("systematic_offset") != offset:
        raise ValueError("QASPER sampling offset must match the fixed-seed systematic design")
    for (key, entry), count in zip(strata.items(), counts):
        quota, probability = quotas[key], quotas[key] / entry["population"]
        expected = {"expected_quota_numerator": quota.numerator, "expected_quota_denominator": quota.denominator,
                    "inclusion_probability_numerator": probability.numerator,
                    "inclusion_probability_denominator": probability.denominator, "sampled": count}
        if any(type(entry.get(name)) is not int or entry[name] != value for name, value in expected.items()):
            raise ValueError("QASPER inclusion probability must use hierarchical expected quota, not realized n/N")
    actual = Counter()
    sampled_papers, sampled_questions = set(), set()
    for row in rows:
        validate_external_metadata(row)
        key = tuple(json.loads(row["sampling_stratum"]))
        entry = strata.get(key)
        if not entry or any(row[name] != entry[name] for name in
                            ("inclusion_probability_numerator", "inclusion_probability_denominator")):
            raise ValueError("Sampled QASPER record does not match its attested stratum probability")
        actual[key] += 1
        sampled_papers.add(key[0])
        sampled_questions.add(key[:2])
    if actual != Counter({key: entry["sampled"] for key, entry in strata.items()}):
        raise ValueError("Sampled QASPER membership does not match the complete design frame")
    expected = {"population_papers": len(papers), "population_questions": sum(map(len, questions.values())),
                "sampled_papers": len(sampled_papers), "sampled_questions": len(sampled_questions),
                "label_population": {label: sum(entry["population"] for key, entry in strata.items() if key[2] == label)
                                     for label in ("no", "mixed", "yes")},
                "label_sample": dict(Counter({label: sum(value for key, value in actual.items() if key[2] == label)
                                              for label in {key[2] for key in actual}}))}
    if any(design.get(key) != value for key, value in expected.items()):
        raise ValueError("QASPER sampling-frame summary disagrees with its population/sample")
    return {key: value for key, value in design.items() if key != "strata"} | {
        "strata": len(strata), "frame_sha256": sha256(design)}


def require_external_complete(features):
    if dataset_scope(features) is None:
        return None
    require = getattr(features, "require_external_ready", None)
    if not callable(require):
        raise ValueError("Scoped final evaluation requires require_external_ready()")
    readiness = require()
    membership = external_test_membership(features)
    if not isinstance(readiness, dict) or readiness.get("records") != membership["records"] or (
            readiness.get("counts") != membership["counts"]):
        raise ValueError("External readiness does not account for the exact external test membership")
    preparations = readiness.get("preparations", [])
    expected = {"qasper-external.jsonl": sum(membership["qasper_subtypes"].values()),
                **{f"vl-rewardbench-{primitive}.jsonl": membership["vl_paired_examples"]
                   for primitive in ("choice", "noul")}}
    if (len(preparations) != 3 or set(expected) != set(required_preparations("external"))
            or {Path(item["path"]).name: item["records"] for item in preparations} != expected):
        raise ValueError("Require all three complete external preparations, not a replacement adapter")
    qasper = next(item for item in preparations if Path(item["path"]).name == "qasper-external.jsonl")
    path = project_path("data/manifests/qasper-external-splits.json")
    if file_sha256(path) != qasper.get("manifest_sha256"):
        raise ValueError("QASPER sampling manifest checksum mismatch")
    manifest = read_json(path)
    rows = (metadata(dict(row), scoped=True) for row in features.db.execute(
        "SELECT * FROM records WHERE source='qasper' AND json_extract(metadata,'$.adaptation')='evidence' ORDER BY id"))
    design = validate_sampling_design(manifest.get("sampling"), rows)
    if file_sha256(path) != qasper["manifest_sha256"]:
        raise ValueError("QASPER sampling manifest changed during attestation")
    return {**readiness, "sampling_design": design}


def require_final_ready(features):
    """Evaluation readiness is strictly stronger than frozen training readiness."""
    training = require_complete(features)
    external = require_external_complete(features)
    if external is None:
        return training
    return {"training": training, "external": external,
            "preparations": training["preparations"] + external["preparations"],
            "records": training["records"] + external["records"],
            "counts": {**training["counts"], **external["counts"]}}


def aligned_test_rows(reference, candidate):
    """Merge by id, not shard order; never sort or repair a candidate list."""
    scope = dataset_scope(reference)
    if dataset_scope(candidate) != scope:
        raise ValueError("Comparison dataset scope mismatch")
    for left, right in zip_longest(test_rows(reference), test_rows(candidate)):
        if left is None or right is None:
            raise ValueError("Test membership mismatch (missing or extra records)")
        a, b = metadata(left, scoped=scope is not None), metadata(right, scoped=scope is not None)
        if semantic_fields(a) != semantic_fields(b) or any(a.get(key) != b.get(key) for key in semantic_fields(a)):
            raise ValueError(f"Per-id/group/candidate test alignment mismatch: {a['id']}")
        if not a.get("has_baseline") or not b.get("has_baseline"):
            raise ValueError("Missing untouched label-token baseline; never substitute zero logits")
        yield right, b


def task_track(meta):
    return "/".join((meta["source"], meta["primitive"], meta["modality"],
                     meta.get("attribute") or meta.get("adaptation") or "main"))


def update_membership(digest, meta):
    digest.update(json.dumps({key: meta.get(key) for key in semantic_fields(meta)}, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode() + b"\n")


def canonical_test_membership(reference):
    """Authoritative semantic membership, independent of shard layout or engine."""
    scope = dataset_scope(reference)
    external = external_test_membership(reference)
    digest, groups, counts = hashlib.sha256(), set(), Counter()
    for index, row in enumerate(test_rows(reference), 1):
        if index > MAX_RECORDS:
            raise MemoryError("Canonical test membership exceeds the evaluation bound")
        meta = metadata(row, scoped=scope is not None)
        update_membership(digest, meta)
        groups.add(meta["group"])
        counts[task_track(meta)] += 1
    if not counts:
        raise ValueError("Canonical reference has no final-test records")
    return {"records": sum(counts.values()), "groups": len(groups),
            "ordered_sha256": digest.hexdigest(), "counts": dict(sorted(counts.items())),
            **({"external": external, "dataset_scope_sha256": sha256(scope)} if scope is not None else {})}


def require_comparison_complete(reference, features, subset_sha256=None, audit=None):
    """Accept a full corpus or an explicitly sealed complete canonical test set."""
    if dataset_scope(reference) != dataset_scope(features):
        raise ValueError("Comparison dataset scope mismatch")
    reference_readiness = require_final_ready(reference)
    if features.identity.get("corpus_scope") != TEST_SCOPE:
        if subset_sha256 is not None:
            raise ValueError("Subset manifest pins require explicit evaluation-test-only scope")
        readiness = require_final_ready(features)
        if readiness != reference_readiness:
            raise ValueError("Transfer requires the same complete prepared corpus")
        return readiness
    shards = require_feature_ownership(features)
    if features.preparations():
        raise ValueError("Evaluation-only corpora must not register or fake full preparation completion")
    if features.db.execute("SELECT COUNT(*) FROM records WHERE role!='test' OR role IS NULL").fetchone()[0]:
        raise ValueError("Evaluation-only corpora may contain exclusively test roles")
    path = features.directory / SUBSET_FILE
    if not subset_sha256 or file_sha256(path) != subset_sha256:
        raise ValueError("Explicit subset extraction manifest checksum mismatch or missing pin")
    manifest = read_json(path)
    if (set(manifest) != {"schema_version", "artifact_kind", "roles", "reference", "extraction", "ownership"}
            or manifest["schema_version"] != 1 or manifest["artifact_kind"] != TEST_SCOPE or manifest["roles"] != ["test"]):
        raise ValueError("Invalid explicit test-only extraction manifest")
    canonical = canonical_test_membership(reference)
    expected_reference = {"feature_identity_sha256": file_sha256(reference.directory / "identity.json"),
                          "preparation_sha256": sha256(reference_readiness), "test_membership": canonical}
    if manifest["reference"] != expected_reference:
        raise ValueError("Subset manifest does not identify the complete canonical reference test membership")
    count = features.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if count != canonical["records"] or manifest["extraction"] != {
            "completed": True, "examined": count, "committed": count, "failures": 0}:
        raise ValueError("Canonical test extraction is incomplete; no missing or extra records allowed")
    ownership = manifest["ownership"]
    if set(ownership) != {"feature_identity_sha256", "index_sha256", "shards"}:
        raise ValueError("Missing explicit subset feature ownership")
    # A sealed SQLite index must account for every committed row in its checksum.
    # A nonempty WAL would place rows outside that pinned file.
    wal = features.directory / "index.sqlite3-wal"
    if (wal.exists() and wal.stat().st_size) or (
            ownership["feature_identity_sha256"] != file_sha256(features.directory / "identity.json")
            or ownership["index_sha256"] != file_sha256(features.directory / "index.sqlite3")):
        raise ValueError("Subset feature identity/index ownership mismatch or unsealed WAL")
    declared = ownership["shards"]
    if (not isinstance(declared, list) or any(set(entry) != {
            "shard", "records", "tensors_sha256", "metadata_sha256"} for entry in declared)
            or [{key: entry[key] for key in ("shard", "records")} for entry in declared] != shards):
        raise ValueError("Subset shard ownership must account for every committed feature row")
    audit = audit or ShardAudit(features)
    if audit.features is not features:
        raise ValueError("Subset audit belongs to a different feature corpus")
    audit.expected = {f"shard-{entry['shard']:06d}.{suffix}": entry[key] for entry in declared
                      for suffix, key in (("safetensors", "tensors_sha256"), ("json", "metadata_sha256"))}
    for shard in shards:
        sidecar = audit.metadata(shard["shard"])
        if len(sidecar) != shard["records"]:
            raise ValueError("Subset sidecar does not account for its owned shard rows")
        indexed = features.db.execute("SELECT row_,metadata FROM records WHERE shard=? ORDER BY row_", (shard["shard"],))
        if any(sidecar[row] != json.loads(meta) for row, meta in indexed):
            raise ValueError("Subset sidecar/index row ownership mismatch")
        from safetensors import safe_open

        with safe_open(str(features.directory / f"shard-{shard['shard']:06d}.safetensors"), framework="pt") as handle:
            shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
        width = shapes.get("mask", [0, 0])[-1]
        expected_shapes = {"query": [shard["records"], 2560], "candidates": [shard["records"], width, 2560],
                           **{key: [shard["records"], width] for key in ("mask", "targets", "baseline", "ordinal_values")}}
        if not 2 <= width <= 16 or shapes != expected_shapes:
            raise ValueError("Subset tensor rows/shapes do not match declared feature ownership")
    # This independent pass establishes completeness before any head predictions.
    # It compares semantics to the full reference, not to the subset's own claims.
    digest = hashlib.sha256()
    for _, meta in aligned_test_rows(reference, features):
        update_membership(digest, meta)
    if digest.hexdigest() != canonical["ordered_sha256"]:
        raise ValueError("Subset canonical test semantic membership mismatch")
    audit.finish()
    return {"scope": TEST_SCOPE, "manifest_sha256": subset_sha256, "manifest": manifest}


class ShardAudit:
    def __init__(self, features):
        self.features = features
        self.files = {}
        self.expected = {}

    def metadata(self, shard):
        stem = self.features.directory / f"shard-{shard:06d}"
        for suffix in (".safetensors", ".json"):
            path = stem.with_suffix(suffix)
            if path.name not in self.files:
                before = path.stat()
                digest = file_sha256(path)
                after = path.stat()
                stamp = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                if stamp != (before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                    raise ValueError("Feature shard changed during checksum verification")
                if self.expected and self.expected.get(path.name) != digest:
                    raise ValueError("Subset shard checksum does not match declared feature ownership")
                self.files[path.name] = {"stamp": stamp, "sha256": digest}
        return read_json(stem.with_suffix(".json"))

    def finish(self):
        for name, value in self.files.items():
            stat = (self.features.directory / name).stat()
            if value["stamp"] != (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
                raise ValueError("Feature shard changed during final evaluation")
        return {name: value["sha256"] for name, value in sorted(self.files.items())}


def checked_batch(features, rows, metas, audit):
    import torch

    for shard in {row["shard"] for row in rows}:
        sidecar = audit.metadata(shard)
        for row, meta in zip(rows, metas):
            if row["shard"] == shard and sidecar[row["row_"]] != meta:
                raise ValueError("Shard row does not match its id/group/candidate metadata")
        del sidecar
    batch = collate(features, rows, "cpu")
    for key in ("query", "candidates", "targets", "baseline", "ordinal_values"):
        if batch[key].dtype != torch.float32 or not torch.isfinite(batch[key]).all():
            raise ValueError(f"Invalid FP32 frozen feature tensor: {key}")
    for i, meta in enumerate(metas):
        count = len(meta["candidates"])
        mask = torch.arange(batch["mask"].shape[1]) < count
        target = torch.zeros_like(batch["targets"][i])
        target[:count] = torch.tensor(meta["target"], dtype=torch.float32)
        if (batch["mask"].dtype != torch.bool or not torch.equal(batch["mask"][i], mask)
                or not torch.equal(batch["targets"][i], target)):
            raise ValueError("Candidate mask/target tensor alignment mismatch")
        if meta["primitive"] == "ordinal":
            values = torch.tensor([c["ordinal_value"] for c in meta["candidates"]], dtype=torch.float32)
            if not torch.equal(batch["ordinal_values"][i, :count], values):
                raise ValueError("Ordinal tensor/candidate alignment mismatch")
    return batch


def quality_gates(statistics, track_metadata):
    result = {}
    for method in METHODS[1:]:
        checks = {"macro_normalized_nll": gate(statistics["macro_normalized_nll"]["paired"][method], 0., strict=True)}
        for track, values in statistics["tracks"].items():
            primitive = track_metadata[track]["primitive"]
            metrics = ["ordinal_mae"] if primitive == "ordinal" else ["accuracy", "tie_aware_accuracy"]
            if track_metadata[track].get("adaptation") == "evidence":
                metrics.append("evidence_f1")
            for metric in metrics:
                checks[f"{track}/{metric}"] = gate(values["paired"][method][metric],
                                                     .1 if metric == "ordinal_mae" else .02,
                                                     higher_is_better=metric != "ordinal_mae")
        statuses = {check["status"] for check in checks.values()}
        result[method] = {"status": "fail" if "fail" in statuses else
                          "inconclusive" if "inconclusive" in statuses else "pass", "checks": checks}
    return result


def primitive_batches(chunk):
    """Separate metric/calibration semantics without requiring separate scorers."""
    if any(meta["primitive"] not in {"choice", "noul", "ordinal"} for _, meta in chunk):
        raise ValueError("Unknown evaluation primitive; never silently omit a task")
    for primitive in ("choice", "noul", "ordinal"):
        selection = [(row, meta) for row, meta in chunk if meta["primitive"] == primitive]
        if selection:
            rows, metas = zip(*selection)
            yield primitive, rows, metas


def evaluate_cell(reference, features, bundle, temperatures, *, batch_size=256, draws=2000, seed=17,
                  subset_manifest_sha256=None, progress=None, device="cpu"):
    import torch

    if not 1 <= batch_size <= LIMITS["batch_size"]:
        raise ValueError("Batch size must be between 1 and 256")
    version = features.db.execute("PRAGMA data_version").fetchone()[0]
    audit = ShardAudit(features)
    readiness = require_comparison_complete(reference, features, subset_manifest_sha256, audit)
    for key in ("encoding", "datasets", "feature_tap", "hidden_size"):
        if features.identity[key] != reference.identity[key]:
            raise ValueError(f"Incompatible frozen-reference feature semantics: {key}")
    device = torch.device(device)
    if device not in (torch.device('cpu'), torch.device('cuda:0')):
        raise ValueError("Processed evaluation supports CPU or explicit cuda:0 only")
    if bundle.training or any(p.requires_grad or p.device != device or p.dtype != torch.float32
                              for p in bundle.parameters()):
        raise ValueError("Evaluation requires a frozen FP32 head on the declared device")
    values = {method: [] for method in METHODS}
    passed = 0
    total = features.db.execute("SELECT COUNT(*) FROM records WHERE role='test'").fetchone()[0]
    if progress is not None:
        progress(0, passed, total)
    evidence = {method: [[], []] for method in METHODS}
    tracks, groups, track_metadata = [], [], {}
    weights, inclusions, strata = [], [], []
    scoped = dataset_scope(reference) is not None
    membership = hashlib.sha256()
    pairs = aligned_test_rows(reference, features)
    while chunk := list(islice(pairs, batch_size)):
        for _, meta in chunk:
            update_membership(membership, meta)
        # Calibration and ordinal metrics remain typed even for a shared scorer.
        for primitive, rows, metas in primitive_batches(chunk):
            batch = checked_batch(features, rows, metas, audit)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                kwargs = {"baseline": inputs["baseline"]} if getattr(bundle, "requires_lm_scores", False) else {}
                head_logits = bundle(primitive, inputs["query"], inputs["candidates"], inputs["mask"],
                                     **kwargs).detach().cpu().numpy()
            target, mask = batch["targets"].numpy(), batch["mask"].numpy()
            if not np.isfinite(head_logits[mask]).all():
                raise ValueError("Nonfinite frozen-head logits")
            for method in METHODS:
                logits = batch["baseline"].numpy() if method == "label_token" else head_logits
                temperature = temperatures[primitive] if method == "head_calibrated" else 1.
                kwargs = {"ordinal_values": batch["ordinal_values"].numpy()} if primitive == "ordinal" else {}
                metrics = per_record(logits, target, mask, temperature=temperature, **kwargs)
                if method == "head_calibrated":
                    passed += int(np.count_nonzero(metrics["tie_aware_accuracy"]))
                for name in ("ordinal_mae", "ranked_probability_score"):
                    metrics.setdefault(name, np.full(len(rows), np.nan))
                values[method].append(metrics)
                p = probabilities(np.where(mask, logits, -np.inf), temperature)
                for i, meta in enumerate(metas):
                    is_evidence = primitive == "noul" and meta.get("adaptation") == "evidence"
                    yes = [c["id"] for c in meta["candidates"]].index("yes") if is_evidence else 0
                    evidence[method][0].append(p[i, yes] if is_evidence else np.nan)
                    evidence[method][1].append(target[i, yes] if is_evidence else np.nan)
            for meta in metas:
                track = task_track(meta)
                description = {key: meta.get(key) for key in
                               ("source", "primitive", "modality", "attribute", "adaptation", "evaluation_scope")}
                if scoped and meta["source"] == "vl-rewardbench":
                    description["estimand"] = "pairwise response preference, not absolute correctness"
                if track in track_metadata and track_metadata[track] != description:
                    raise ValueError("Ambiguous task-track identity")
                track_metadata[track] = description
                tracks.append(track)
                groups.append(meta["group"])
                external = scoped and meta["source"] in EXTERNAL_SOURCES
                weights.append(meta["sampling_weight"] if external else 1.)
                inclusions.append(meta["inclusion_probability"] if external else 1.)
                strata.append((meta.get("sampling_stratum") or "") if external else "")
            if progress is not None:
                progress(len(tracks), passed, total)
    variants = {method: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
                for method, parts in values.items()}
    statistics = compare_tracks(tracks, groups, variants,
                                {method: tuple(map(np.asarray, pair)) for method, pair in evidence.items()},
                                draws=draws, seed=seed,
                                **({"weights": weights, "inclusion_probabilities": inclusions,
                                    "sampling_strata": strata} if scoped else {}))
    if readiness != require_comparison_complete(reference, features, subset_manifest_sha256, audit) or (
            version != features.db.execute("PRAGMA data_version").fetchone()[0]):
        raise ValueError("Corpus changed during final evaluation")
    return {"status": "evaluated", "feature_identity": features.identity,
            "corpus_scope": features.identity.get("corpus_scope", "full"),
            "preparation": readiness, "test_membership": {"records": len(tracks), "ordered_sha256": membership.hexdigest()},
            **({"external_membership": external_test_membership(features)} if scoped else {}),
            "feature_files_sha256": audit.finish(), "track_metadata": track_metadata,
            **statistics, "gates": quality_gates(statistics, track_metadata)}


def validate_plan(plan):
    if set(plan) != {"schema_version", "reference", "cells"} or plan["schema_version"] != 1:
        raise ValueError("Expected evaluation plan schema_version 1; no adaptation overrides")
    ref = plan["reference"]
    if set(ref) != {"features", "feature_identity_sha256", "head", "sha256"}:
        raise ValueError("Pin one reference corpus and one head/calibration for the entire matrix")
    if not Path(ref["features"]).is_absolute() or not Path(ref["head"]).is_absolute():
        raise ValueError("Evaluation artifact paths must be absolute")
    seen = set()
    for cell in plan["cells"]:
        if (set(cell) - {"id", "status", "features", "feature_identity_sha256", "subset_manifest_sha256", "reason", "evidence"}
                or not {"id", "status"} <= set(cell) or not isinstance(cell["id"], str) or not cell["id"]):
            raise ValueError("Invalid cell fields; per-cell head/calibration adaptation is not allowed")
        if cell["id"] in seen:
            raise ValueError("Duplicate evaluation cell")
        seen.add(cell["id"])
        status = cell["status"]
        if status == "evaluate":
            required = {"id", "status", "features", "feature_identity_sha256"}
            if (not required <= set(cell) or set(cell) - required - {"subset_manifest_sha256"}
                    or not Path(cell["features"]).is_absolute()):
                raise ValueError("Evaluated cells require an absolute corpus path and pinned identity")
            if cell["id"] == "native-4.05" and (
                    Path(cell["features"]).resolve() != Path(ref["features"]).resolve()
                    or cell["feature_identity_sha256"] != ref["feature_identity_sha256"]):
                raise ValueError("native-4.05 must use the actual reference feature corpus")
        elif status in {"not_run", "unsupported", "memory_infeasible", "failed"}:
            if (set(cell) - {"id", "status", "reason", "evidence"} or not cell.get("reason")
                    or (status != "not_run" and not cell.get("evidence"))):
                raise ValueError("Unavailable cells need an explicit reason and evidence; no silent omission")
        else:
            raise ValueError("Unknown evaluation cell status")


def evaluate(plan, *, batch_size=256, draws=2000, seed=17, progress=None):
    """Evaluate supplied cells sequentially; missing required cells remain visible."""
    import torch

    validate_plan(plan)
    if not 1 <= batch_size <= LIMITS["batch_size"] or not 1 <= draws <= 10_000:
        raise ValueError("Evaluation resource bounds exceeded")
    ref = plan["reference"]
    with ExitStack() as stack:
        reference = open_features(ref["features"], ref["feature_identity_sha256"])
        stack.callback(reference.close)
        reference_version = reference.db.execute("PRAGMA data_version").fetchone()[0]
        bundle, manifest, temperatures = load_reference(reference, ref["head"], ref["sha256"])
        final_readiness = require_final_ready(reference)
        cells = {cell["id"]: cell for cell in plan["cells"]}
        report = {"schema_version": 1, "scope": "offline final-test quality only; not runtime qualification",
                  "track": "frozen-reference-head", "primary_method": "head_calibrated",
                  "plan_sha256": sha256(plan), "reference": {**ref, "manifest": manifest, "temperatures": temperatures,
                                                               "evaluation_preparation": final_readiness},
                  "evaluation_code_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                              for name in ("evaluate.py", "evaluation_metrics.py", "metrics.py", "features.py", "head.py", "train.py",
                                                           *(("scope.py", "external_sampling.py") if
                                                             dataset_scope(reference) is not None else ()),
                                                           *(("shared_head.py", "shared_train.py", "shared_training_data.py",
                                                              "shared_qualification.py") if manifest.get("architecture") in
                                                             {"shared-candidate-dot-product-v1",
                                                              "shared-lm-assisted-candidate-dot-product-v1",
                                                              "shared-lm-assisted-candidate-dot-product-rank512-v1",
                                                              "shared-lm-assisted-context-trust-candidate-dot-product-rank512-v1",
                                                              "shared-lm-assisted-query-adapter-candidate-dot-product-rank512-v1",
                                                              "shared-lm-assisted-query-channel-gate-candidate-dot-product-rank512-v1",
                                                              "shared-lm-assisted-query-channel-gate-adapter64-candidate-dot-product-rank512-v1",
                                                              "shared-lm-assisted-query-key-adapter-channel-gate-adapter64-candidate-dot-product-rank512-v1"} else ()))},
                  "execution": {"device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
                                "threads": torch.get_num_threads(), "batch_size": batch_size, "limits": LIMITS,
                                "bootstrap_draws": draws, "bootstrap_seed": seed},
                  "gate_policy": {"accuracy_f1_max_loss": .02, "ordinal_mae_max_increase": .1,
                                  "macro_normalized_nll": "paired group 95% upper CI strictly below zero",
                                  "intervals": "unadjusted two-sided 95% percentile; all required checks must pass",
                                  "insufficient_groups_or_invalid_draws": "inconclusive, never pass"}, "cells": {}}
        for name in (*REQUIRED_CELLS, *(key for key in cells if key not in REQUIRED_CELLS)):
            cell = cells.get(name, {"id": name, "status": "not_run", "reason": "Not supplied in the evaluation plan"})
            if cell["status"] != "evaluate":
                report["cells"][name] = {**cell, "status_origin": "declared; not tested by this offline evaluator"}
                continue
            features = None
            try:
                check_hashes(ref["head"], ref["sha256"])
                features = open_features(cell["features"], cell["feature_identity_sha256"])
                result = evaluate_cell(reference, features, bundle, temperatures,
                                       batch_size=batch_size, draws=draws, seed=seed,
                                       subset_manifest_sha256=cell.get("subset_manifest_sha256"),
                                       progress=(None if progress is None else
                                                 lambda processed, passed, total: progress(name, processed, passed, total)))
                if file_sha256(features.directory / "identity.json") != cell["feature_identity_sha256"]:
                    raise ValueError("Feature identity changed during final evaluation")
                report["cells"][name] = {**cell, **result}
            except (ValueError, KeyError, IndexError, OSError, RuntimeError, MemoryError) as error:
                report["cells"][name] = {**cell, "status": "memory_infeasible" if isinstance(error, MemoryError) else "failed",
                                        "status_origin": "offline evaluation; not a GPU feasibility result",
                                        "reason": f"{type(error).__name__}: {error}"}
            finally:
                if features is not None:
                    features.close()
        check_hashes(ref["head"], ref["sha256"])
        if require_final_ready(reference) != final_readiness:
            raise ValueError("Reference preparations changed during final evaluation")
        if reference_version != reference.db.execute("PRAGMA data_version").fetchone()[0]:
            raise ValueError("Reference corpus changed during final evaluation")
        if file_sha256(reference.directory / "identity.json") != ref["feature_identity_sha256"]:
            raise ValueError("Reference identity changed during final evaluation")
    report["coverage_complete"] = all(report["cells"][name]["status"] == "evaluated" for name in REQUIRED_CELLS)
    report["all_required_quality_gates_pass"] = report["coverage_complete"] and all(
        report["cells"][name]["gates"]["head_calibrated"]["status"] == "pass" for name in REQUIRED_CELLS)
    return report


def markdown_report(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    def estimate(value):
        return f"{number(value['difference'])} [{number(value['low'])}, {number(value['high'])}]"

    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Frozen-Reference Final Evaluation", "", report["scope"], "",
             f"Frozen weights: `{report['reference']['sha256']['head.safetensors']}`",
             f"Frozen calibration: `{report['reference']['sha256']['calibration.json']}`", "",
             "All differences are head minus untouched label-token control. Intervals are paired",
             "group 95% bootstrap intervals; raw and frozen-calibrated results are separate.",
             "Accuracy/F1 loss must be <=0.02; MAE increase <=0.1; macro normalized NLL",
             "must have an upper interval endpoint <0. Missing or insufficient evidence cannot pass.", "",
             "| Cell | Status | Raw macro NLL delta [95% CI] | Calibrated delta [95% CI] | Calibrated gates |",
             "| --- | --- | --- | --- | --- |"]
    for name, cell in report["cells"].items():
        if cell["status"] != "evaluated":
            lines.append(f"| {text(name)} | {text(cell['status'])} | unavailable | unavailable | not evaluated |")
        else:
            macro = cell["macro_normalized_nll"]["paired"]
            lines.append(f"| {text(name)} | evaluated | {estimate(macro['head_raw'])} | "
                         f"{estimate(macro['head_calibrated'])} | {cell['gates']['head_calibrated']['status']} |")
    for name, cell in report["cells"].items():
        lines.extend(["", f"## {text(name)}", ""])
        if cell["status"] != "evaluated":
            lines.extend([text(cell["reason"]), "", f"Evidence: {text(cell.get('evidence', cell.get('status_origin', 'not supplied')))}"])
            continue
        lines.extend([f"Feature corpus: {cell['corpus_scope']}. Test records: {cell['test_membership']['records']}.",
                      "The frozen reference weights and calibration apply to this complete test set.", ""])
        if "external_membership" in cell:
            external = cell["external_membership"]
            lines.extend([f"External QASPER subtypes: {text(external['qasper_subtypes'])}.",
                          f"VL-RewardBench: {external['vl_paired_examples']} shared examples in EACH of Choice and Noul.",
                          "VL accuracy is pairwise response preference, not absolute correctness. "
                          "QASPER subtype accuracies are never pooled.",
                          "Weighted ratios use fixed inverse inclusion probabilities. Intervals resample whole "
                          "papers/image groups; they are not exact multistage survey-design intervals.", "",
                          "| Task track | Records | Groups | Weight sum | Weight ESS |",
                          "| --- | --- | --- | --- | --- |"])
            for track, values in cell["tracks"].items():
                sampling = values["sampling"]
                lines.append(f"| {text(track)} | {values['records']} | {values['groups']} | "
                             f"{number(sampling['sum_weights'])} | {number(sampling['effective_records'])} |")
            lines.append("")
        lines.extend(["| Task track | Metric | Label-token | Raw head | Calibrated head | Raw delta [95% CI] | Calibrated delta [95% CI] |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for track, values in cell["tracks"].items():
            for metric in values["paired"]["head_raw"]:
                points = [values["methods"][method][metric] for method in METHODS]
                if all(value is None for value in points):
                    continue
                columns = [text(track), metric, *map(number, points),
                           estimate(values["paired"]["head_raw"][metric]),
                           estimate(values["paired"]["head_calibrated"][metric])]
                lines.append("| " + " | ".join(columns) + " |")
        lines.extend(["", "| Method | Gate | Status |", "| --- | --- | --- |"])
        for method, gates in cell["gates"].items():
            for key, check in gates["checks"].items():
                lines.append(f"| {method} | {text(key)} | {check['status']} |")
    lines.extend(["", "Evidence F1 is recomputed from group-resampled TP/FP/FN, not averaged per record.",
                  "Unique-gold accuracy excludes tied targets; expected accuracy and tie-aware accuracy",
                  "retain their distinct meanings. Counts, eligible groups, valid bootstrap draws,",
                  "artifact provenance and complete gate intervals are in the companion JSON.", ""])
    return "\n".join(lines)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .json report; also writes sibling .md")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-output", type=Path, help="Optional atomic pass-rate telemetry, not a quality gate")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Choose a new .json report path; existing evidence is never overwritten")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    latest = {}

    def progress(cell, processed, passed, total):
        import time
        from .acquire import write_json
        latest.update(run_id=f"{args.output.stem}:{cell}", stage="processed-records",
                      status="in_progress", processed=processed, passed=passed, total=total,
                      pass_rate=passed / processed if processed else None, updated_at=time.time())
        write_json(args.progress_output, latest)

    try:
        report = evaluate(read_json(args.plan), batch_size=args.batch_size, draws=args.draws, seed=args.seed,
                          progress=progress if args.progress_output else None)
    except Exception:
        if latest:
            from .acquire import write_json
            latest["status"] = "failed"
            write_json(args.progress_output, latest)
        raise
    if latest:
        import time
        from .acquire import write_json
        cell = report["cells"][latest["run_id"].rsplit(":", 1)[1]]
        latest.update(status="completed" if cell["status"] == "evaluated" else "failed",
                      quality_gate_status=cell.get("gates", {}).get("head_calibrated", {}).get("status", "unavailable"),
                      updated_at=time.time())
        write_json(args.progress_output, latest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".md").open("x") as handle:
        handle.write(markdown_report(report))
    print(json.dumps({"json": str(args.output), "markdown": str(args.output.with_suffix('.md')),
                      "coverage_complete": report["coverage_complete"],
                      "all_required_quality_gates_pass": report["all_required_quality_gates_pass"]}))


if __name__ == "__main__":
    main()
