"""Versioned append-only decision encoding, shared by every engine."""

import hashlib
import json
from pathlib import Path


VERSION = "fuzev-candidate-branches-v1"
TEMPLATE = Path(__file__).resolve().parent / "templates/chat_template.jinja"


def identity():
    return {"version": VERSION, "chat_template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "thinking": False, "candidate_features": "independent candidate suffix from identical query prefix",
            "readout": "last consumed token, post-collapse, FP32 owned copy"}


def query_suffix(question, candidates, additional_context=""):
    labels = [chr(ord("A") + index) for index in range(len(candidates))]
    choices = "\n".join(f"({label}) {candidate['text']}" for label, candidate in zip(labels, candidates))
    evidence = f"Additional evidence:\n{additional_context}\n" if additional_context else ""
    return ("\n<|im_start|>user\nSelect one candidate. Return only its label.\n"
            f"{evidence}Question: {question}\nCandidates:\n{choices}"
            "<|im_end|>\n<|im_start|>assistant\nAnswer:")


def candidate_suffix(candidate):
    return "\nCandidate under consideration:\n" + candidate["text"]


def token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()
