"""Stdlib-only KEV suite records -> independent FuZev ReadoutPreparer rows.

Text follows kev.api.render/option_text at KEV_SOURCE_REVISION, without importing
KEV or FuZev. Call adapt_record(record) or iter_request_rows(records) on already
loaded official records. Only context, question and candidates are model inputs;
targets and official_* fields are evaluation metadata. No dataset I/O occurs here.

Choice/Score keep official order. Noul uses FuZev [Yes, No] = [true, false],
not KEV [false, true]. The *_from_* fields are gather indices, not new labels.
prediction_metadata(..., input_order="fuzev"|"official") returns canonical KEV p.
"""

import copy
import hashlib
import json
import math


KEV_SOURCE_REVISION = "fc4a17e194bca35a48c57211392ec49ea54a94ac"
ADAPTER_VERSION = "kev-requests-v1"


def state_text(value, indent=0):
    """Exact kev.api.render for JSON content; preserve field and list order."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON content must not contain nonfinite numbers")
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {state_text(item, indent + 1).lstrip()}" for item in value)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected JSON content with string object keys")
    return "\n".join(
        f"{pad}{key}:\n{state_text(item, indent + 1)}"
        if isinstance(item, (dict, list)) else f"{pad}{key}: {state_text(item)}"
        for key, item in value.items()
    )


def _option_text(key, description):
    return key if description is None or description == "" else f"{key}: {state_text(description)}"


def _labels(question):
    """kev.benchmark.labels semantics, rejecting coerced/ambiguous labels."""
    kind, criteria = question["type"], question.get("criteria")
    label = question["label"]
    if kind == "noul":
        if type(label) is not bool or criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul needs a boolean label and optional object criteria")
        return ["false", "true"], int(label)
    if kind == "choice":
        if (not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255
                or any(not isinstance(key, str) for key in criteria)
                or not isinstance(label, str) or label not in criteria):
            raise ValueError("Choice needs ordered object criteria and an existing option-key label")
        keys = list(criteria)
        return keys, keys.index(label)
    if kind == "score":
        if (not isinstance(criteria, list) or not 1 <= len(criteria) <= 255
                or type(label) is not int or not 0 <= label < len(criteria)):
            raise ValueError("Score needs ordered levels and a zero-based integer label")
        return [str(index) for index in range(len(criteria))], label
    raise ValueError(f"Unknown KEV question type: {kind!r}")


def adapt_record(record, *, split="test", suite="breadth-v1"):
    """Return one eval-only row per question in official insertion order.

    Preserve the source _meta and original label/instructions/criteria outside
    model inputs. Development is also role=test: this adapter never emits FIT.
    The caller owns suite loading/admission and permission to read locked test.
    """
    if split not in {"test", "development"}:
        raise ValueError("KEV adaptation is evaluation-only (test or development)")
    meta = record["_meta"]
    if any(not isinstance(meta.get(key), str) or not meta[key] for key in ("id", "source")):
        raise ValueError("Official _meta needs nonempty id and source")
    questions = record["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Official questions must be a nonempty ordered object")
    context_text = state_text(record["state"])
    rows = []
    for qid, question in questions.items():
        if not isinstance(qid, str):
            raise ValueError("Official question IDs must be strings")
        try:
            keys, label_index = _labels(question)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{meta['id']!r}/{qid!r}: {error}") from error
        kind, criteria = question["type"], question.get("criteria")
        if kind == "noul":
            criteria = criteria or {}
            candidate_keys = ["true", "false"]
            candidates = [
                {"id": "yes", "text": _option_text("Yes", criteria.get("true"))},
                {"id": "no", "text": _option_text("No", criteria.get("false"))},
            ]
        elif kind == "choice":
            candidate_keys = keys[:]
            candidates = [{"id": key, "text": _option_text(key, description)}
                          for key, description in criteria.items()]
        else:
            candidate_keys = keys[:]
            candidates = [{"id": key, "text": state_text(level), "ordinal_value": index}
                          for index, (key, level) in enumerate(zip(keys, criteria))]
        target_index = candidate_keys.index(keys[label_index])
        identity = json.dumps([ADAPTER_VERSION, KEV_SOURCE_REVISION, suite, split, meta["id"], qid],
                              ensure_ascii=False, separators=(",", ":"))
        rows.append({
            "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "source": meta["source"], "revision": KEV_SOURCE_REVISION,
            "group": meta.get("group_id") or meta["id"], "original_split": split,
            "role": "test", "modality": "text", "dataset_purpose": "external",
            "evaluation_scope": "external", "adapter_version": ADAPTER_VERSION, "suite": suite,
            "context": [{"role": "user", "content": context_text}],
            "question": state_text(question.get("instructions")), "candidates": candidates,
            "target": [float(index == target_index) for index in range(len(candidates))],
            "primitive": "ordinal" if kind == "score" else kind,
            "official_record_id": meta["id"], "official_question_id": qid,
            "official_type": kind, "official_keys": keys, "candidate_keys": candidate_keys,
            "official_label": copy.deepcopy(question["label"]), "official_label_index": label_index,
            "official_instructions": copy.deepcopy(question.get("instructions")),
            "official_criteria": copy.deepcopy(question.get("criteria")),
            "task": question.get("src", meta["source"]), "_meta": copy.deepcopy(meta),
            "official_from_fuzev": [candidate_keys.index(key) for key in keys],
            "fuzev_from_official": [keys.index(key) for key in candidate_keys],
        })
    return rows


def iter_request_rows(records, *, split="test", suite="breadth-v1"):
    """Flatten an iterable of already loaded records without reordering."""
    for record in records:
        yield from adapt_record(record, split=split, suite=suite)


def _gather(values, indices):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError("Prediction values must be an ordered vector")
    values = list(values)
    if len(values) != len(indices):
        raise ValueError("Prediction vector length differs from candidate count")
    return [values[index] for index in indices]


def to_official_order(row, values):
    """FuZev candidate vector -> KEV keys order; works for p or logits."""
    return _gather(values, row["official_from_fuzev"])


def to_fuzev_order(row, values):
    """KEV keys vector -> FuZev candidate order, including official baseline p."""
    return _gather(values, row["fuzev_from_official"])


def prediction_metadata(row, probabilities, *, input_order):
    """Official benchmark identity/label with p explicitly in canonical KEV order.

    No normalization/calibration is applied. Callers retain responsibility for
    distribution validation. Supply input_order explicitly to avoid double flips.
    """
    if input_order == "fuzev":
        p = to_official_order(row, probabilities)
    elif input_order == "official":
        p = _gather(probabilities, list(range(len(row["official_keys"]))))
    else:
        raise ValueError("input_order must be 'fuzev' or 'official'")
    meta = row["_meta"]
    variant = meta.get("variant", "clean")
    return {
        "id": row["official_record_id"], "group": row["group"],
        "question": row["official_question_id"], "source": row["source"], "task": row["task"],
        "type": row["official_type"], "variant": variant,
        "keys": row["official_keys"][:], "label": row["official_label_index"], "p": p,
        "control_id": meta.get("control_id"), "pair_id": meta.get("pair_id"),
        "sibling": meta.get("sibling"),
        "parent": meta.get("parent_id") or (meta["id"] if variant == "clean" else row["group"]),
    }
