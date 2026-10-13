"""Opt-in, single-owner native correctness checks, not a qualification receipt.

Accept an already admitted, exclusively owned NativeService. Never construct a
backend, fetch inputs, or replace the calibrated head. Importing is CPU-only.
State observations are optional and never close unobserved runtime gates.
"""

import hashlib
import json
import math
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from threading import Event
from uuid import uuid4
from weakref import WeakSet

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .encoding import candidate_suffix, query_suffix, token_hash
from .guard_policy import verify_memory_guard
from .runtime_harness import DecisionSpec
from .service_contracts import GenerationRequest, ServiceDecisionRequest, TrustedImage
from .telemetry import MemoryGuard, admit


WINDOW = 262144
STATE_GAP = "Complete native-state observation is disabled or uncompleted; readout agreement is not state proof"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class QualificationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    input_origin: str = Field(min_length=1, max_length=1024)
    input_role: str = Field(pattern=r"^(operator-non-heldout|cpu-fake)$")
    generation: GenerationRequest
    decisions: list[DecisionSpec] = Field(min_length=4, max_length=4)


class ImageControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    input_origin: str = Field(min_length=1, max_length=1024)
    input_role: str = Field(pattern=r"^(operator-non-heldout|cpu-fake)$")
    request: ServiceDecisionRequest
    mismatched_image: TrustedImage
    expected_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def controls(self):
        require(self.request.prefix_id is None and self.request.image is not None,
                "Image controls require an explicit standalone real-image request")
        require(self.request.image.sha256 != self.mismatched_image.sha256,
                "Mismatched image must have a different RGB digest")
        require(self.expected_id in {candidate.id for candidate in self.request.candidates},
                "Image control expected_id is not a candidate")
        return self


class FullWindowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: QualificationCase
    additional_peak_bytes: int = Field(strict=True, gt=0)
    estimate_basis: str = Field(min_length=1, max_length=2048)


class QualificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cases: list[QualificationCase] = Field(min_length=1, max_length=8)
    image_controls: list[ImageControl] = Field(default_factory=list, max_length=8)
    full_window: FullWindowPlan | None = None
    run_full_window: bool = Field(default=False, strict=True)
    observe_native_state: bool = Field(default=False, strict=True)
    atol: float = Field(default=1e-5, gt=0, le=1e-5)
    rtol: float = Field(default=1e-4, gt=0, le=1e-4)

    @model_validator(mode="after")
    def distinct_cases(self):
        names = [case.name for case in self.cases + self.image_controls]
        if self.full_window is not None:
            names.append(self.full_window.case.name)
        require(len(set(names)) == len(names), "Qualification case names must be unique")
        require(not self.run_full_window or self.full_window is not None,
                "Full-window inference requires an explicit case and peak-memory estimate")
        return self


def full_window_feasibility(plan, *, enabled, frozen_head_ready, total, available,
                            fraction=None):
    """Arithmetic only. A feasible estimate is NOT an allocation or memory proof."""
    require(type(enabled) is bool and type(frozen_head_ready) is bool, "Full-window gates require explicit booleans")
    require(fraction is None or (type(fraction) in (float, int) and math.isfinite(fraction) and 0 < fraction <= 1),
            "Declared native memory fraction must be in (0, 1]")
    require(type(total) is int and type(available) is int and 0 <= available <= total and total > 0,
            "Invalid shared-memory observation")
    result = {"window_tokens": WINDOW, "status": "not_requested", "allocated": False,
              "ram_total": total, "ram_available": available, "memory_fraction": fraction}
    if not enabled:
        return result
    require(plan is not None, "Missing full-window plan")
    result.update(additional_peak_bytes=plan.additional_peak_bytes, estimate_basis=plan.estimate_basis)
    if not frozen_head_ready:
        result.update(status="blocked", reason="frozen_head_not_ready")
    elif available - plan.additional_peak_bytes < total * (1 - (fraction if fraction is not None else 1)):
        result.update(status="memory_infeasible", reason="would_breach_shared_headroom")
    else:
        result.update(status="estimate_feasible", reason="unmeasured_operator_estimate")
    return result


def _metadata(service, cpu_fake, *, research_cuda=False):
    if research_cuda and not cpu_fake:
        from .native_research_qualification import research_metadata
        return research_metadata(service)
    runtime = deepcopy(service.backend.readout_runtime)
    result = {"service_provenance": deepcopy(service.provenance), "readout_runtime": runtime,
              "model_name": service.model_name, "window_tokens": service.lane.context_limit,
              "head_manifest_sha256": service.head.manifest_sha256,
              "calibration_id": service.head.calibration_id}
    if cpu_fake:
        return {**result, "source_verification": "cpu_fake_not_native_provenance"}

    from .native import NativeBackend, runtime_sources
    from .paths import project_path
    from .serving_head import ServingHead, checked_bytes

    require(type(service.backend) is NativeBackend and type(service.head) is ServingHead,
            "Native evidence requires the actual NativeBackend and unchanged ServingHead")
    sources = runtime_sources()
    models = json.loads(checked_bytes(project_path("data/manifests/acquisition.json"), 1 << 20))["models"]
    model = models[service.model_name]
    snapshot = str(Path(service.backend.config.directory).resolve())
    require(model.get("local_snapshot") and snapshot == str(Path(model["local_snapshot"]).resolve()),
            "Loaded model directory does not match acquisition metadata")
    actual = service.provenance["serving_identity"]
    require(actual["model"] == model["repo"] and actual["revision"] == model["revision"]
            and actual["cell"] == service.model_name and actual["readout_runtime"] == runtime
            and actual["cache_precision"] == "float16", "Actual serving metadata mismatch")
    cache = service.backend.cache
    require(cache.max_num_tokens == runtime["cache_tokens"] and cache.num_slots == runtime["max_hybrid_slots"],
            "Actual cache geometry mismatch")
    require(service.backend.config.infer_params.ngram_stream_from_disk is True,
            "Native PLE disk streaming must remain enabled")
    require(service.provenance["head_manifest_sha256"] == service.head.manifest_sha256
            and service.head.calibration_id == "sha256:" + service.provenance["calibration_sha256"],
            "Frozen head pin mismatch")
    result.update(source_verification="observed_sources", runtime_sources=sources,
                  recorded_runtime_sources=service.provenance.get("service_sources"),
                  model={"repo": model["repo"], "revision": model["revision"], "local_snapshot": snapshot},
                  cache={"class": type(cache).__module__ + "." + type(cache).__qualname__,
                         "max_num_tokens": cache.max_num_tokens, "num_slots": cache.num_slots,
                         "declared_precision": actual["cache_precision"],
                         "storage_dtype_verified": False},
                  weight_bytes_verified=False)
    return result


def _resources(service):
    """Existing native lane/cache ledgers; never infer cleanup from token hashes."""
    return {
        "branches": sorted(id(branch) for branch in service.lane.branches if not branch.closed),
        "free_pages": len(service.lane.pool.free),
        "free_slots": len(service.backend.cache.free_list),
        "prefixes": sorted(service.prefixes),
        "feature_prefixes": sorted(service.backend.prefixes),
        "images": sorted(service.backend.images),
    }


@contextmanager
def _replace(obj, name, replacement):
    """Restore the original descriptor as well as the method after injected faults."""
    owned = name in vars(obj)
    old = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def _owned(tensor):
    return tensor.detach().float().cpu().clone()


def _state_observer(service, config, cpu_fake):
    from .native_state_observer import NativeStateObserver
    return NativeStateObserver(service, allow_observation=True, cpu_fake=cpu_fake,
                               atol=config.atol, rtol=config.rtol)


class _StateUnavailable(RuntimeError):
    """An unsupported observation is incomplete, not native correctness evidence."""


class _StateObservation:
    """Only lifetime hooks around the existing observer; no backend or alternate inference path."""
    def __init__(self, service, config, cpu_fake):
        self.service, self.observer, self.ready, self.reason = service, None, False, None
        self.witnessed = WeakSet()
        self.evidence_kind = "cpu_fake" if cpu_fake else "native"
        self.forward_flags = {"absent": 0, "explicit_false": 0, "rejected": 0}
        self.forward_calls = {"forward": 0, "prefill": 0}
        self.forward_digest = hashlib.sha256()
        self.forward_failed, self.active, self.parameter_plan = False, None, None
        try:
            self.observer = _state_observer(service, config, cpu_fake)
        except (ValueError, NotImplementedError, AttributeError) as error:
            self.reason = f"{type(error).__name__}: {error}"
        self.default_semantics_verified = not cpu_fake and self.observer is not None

    def digest(self, branch):
        if self.observer is None:
            return None
        if branch.recurrent is None and (branch not in self.witnessed
                or not isinstance(branch.checkpoint, dict) or "_coalesced_slab" not in branch.checkpoint):
            raise _StateUnavailable("Unwitnessed or unsupported frozen checkpoint")
        try:
            receipt = self.observer.digest(branch)
        except (ValueError, NotImplementedError, AttributeError) as error:
            if self.ready:
                if isinstance(error, (NotImplementedError, AttributeError)):
                    raise _StateUnavailable(str(error)) from error
                raise  # A previously observed inventory becoming invalid is not a harmless capability gap.
            self.reason, self.observer = f"{type(error).__name__}: {error}", None
            return None
        require(receipt["qualified"] is False and receipt["evidence_kind"] == self.evidence_kind
                and receipt["consumed_tokens"] == len(branch.tokens), "State observer evidence mismatch")
        self.ready = True
        if branch.recurrent is not None:
            self.witnessed.add(branch)
        return {key: receipt[key] for key in ("sha256", "consumed_tokens", "evidence_kind")}

    def unchanged(self, watched):
        for branch, before in watched:
            if before is not None:
                require(self.digest(branch) == before, "Parent or sibling native state changed")

    def compare(self, reused, fresh):
        try:
            receipt = self.observer.compare(reused, fresh)
        except (NotImplementedError, AttributeError) as error:
            raise _StateUnavailable(str(error)) from error
        require(receipt["qualified"] is False and receipt["evidence_kind"] == self.evidence_kind
                and receipt["status"] == "observed_equal", "State comparison did not observe equivalence")
        return receipt

    def reset(self, branch):
        try:
            receipt = self.observer.verify_reset(branch)
        except (NotImplementedError, AttributeError) as error:
            raise _StateUnavailable(str(error)) from error
        require(receipt["qualified"] is False and receipt["evidence_kind"] == self.evidence_kind
                and receipt["status"] == "observed_reset", "Fresh-slot reset was not observed")
        return receipt

    def _planned(self, branches, tails, features, params, operation):
        from .service_images import embedding_identity

        lane = self.service.lane
        self.service._owner()
        with lane.lock:
            require(self.active is None and 1 <= len(branches) <= lane.batch_capacity
                    and len(branches) == len(tails), "Nested or invalid lane plan")
            require(len(set(branches)) == len(branches) and len({len(t) for t in tails}) == 1
                    and tails[0] and all(type(t) is int and 0 <= t < 2**63 for tail in tails for t in tail),
                    "Invalid exact chunk plan")
            require(all(b.owner is lane and not b.closed and not b.immutable and b in lane.branches for b in branches),
                    "Lane plan contains an unowned branch")
            require(len({len(b.tokens) for b in branches}) == 1, "Lane batch needs a common consumed position")
            if self.parameter_plan is None or self.parameter_plan[0] is not params:
                raise _StateUnavailable("Lane parameters need a witnessed backend.parameters call")
            _, embeddings, identities = self.parameter_plan
            require(tuple(map(embedding_identity, embeddings)) == identities, "Planned image binding changed")
            if set(params) - {"indexed_embeddings", "recurrent_history"}:
                raise _StateUnavailable("Only pinned linear-position/image parameters are supported")
            self.service.admission((sum(map(len, tails)) + WINDOW) * 8
                                   + len(branches) * (lane.chunk_size + WINDOW // 256) * 128
                                   + sum(len(identity[3]) for identity in identities) * 32,
                                   self.service.backend.readout_runtime["memory_fraction"])
            self.active = {"branches": branches, "tails": tails, "base": tuple(b.tokens for b in branches),
                           "offset": 0, "features": features, "embeddings": embeddings, "identities": identities}
            try:
                result = operation()
                require(self.active["offset"] == len(tails[0]), "Lane completed without all actual model-entry witnesses")
                return result
            finally:
                self.active = None

    def _forward(self, name, original, input_ids, params):
        import torch
        from .service_images import embedding_identity

        def cpu(tensor, dtype, shape):
            require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and tensor.dtype == dtype
                    and tensor.layout == torch.strided and tuple(tensor.shape) == shape and not tensor.requires_grad,
                    "Forward metadata must have the exact CPU dtype/shape")
            return tensor.tolist()  # Never transfer a GPU tensor or retain/dump its data.

        self.service._owner()
        plan = self.active
        if plan is None:
            raise _StateUnavailable("Actual model entry has no active NativeLane chunk plan")
        require(type(params) is dict, "Unsupported actual forward parameter container")
        if params.get("recurrent_history", False) is not False:
            self.forward_flags["rejected"] += 1
            raise _StateUnavailable("Actual forward recurrent_history must be absent or explicit False")
        branches, offset = plan["branches"], plan["offset"]
        width = min(self.service.lane.chunk_size, len(plan["tails"][0]) - offset)
        require(width > 0 and name == ("forward" if offset + width == len(plan["tails"][0]) else "prefill"),
                "Unexpected or duplicate forward/prefill entry for planned chunk")
        expected = [list(tail[offset:offset + width]) for tail in plan["tails"]]
        ids = cpu(input_ids, torch.int64, (len(branches), width))
        require(ids == expected, "Actual input IDs differ from planned lane chunk")
        positions = [len(base) + offset for base in plan["base"]]
        require(all(b.tokens == base + tail[:offset] for b, base, tail in zip(branches, plan["base"], plan["tails"])),
                "Branch tokens changed before actual model entry")
        page_width = max(len(b.pages) for b in branches)
        pages = cpu(params.get("block_table"), torch.int32, (len(branches), page_width))
        require(pages == [b.pages + [-1] * (page_width - len(b.pages)) for b in branches]
                and all(len(b.pages) == (p + width + 255) // 256 and len(set(b.pages)) == len(b.pages)
                        and all(type(page) is int and 0 <= page < len(self.service.lane.pool.references)
                                and self.service.lane.pool.references[page] > 0 for page in b.pages)
                        for b, p in zip(branches, positions)), "Actual block table differs from owned logical pages")
        writable = [page for b, p in zip(branches, positions) for page in b.pages[p // 256:]]
        require(len(set(writable)) == len(writable)
                and all(type(self.service.lane.pool.references[page]) is int
                        and self.service.lane.pool.references[page] == 1 for page in writable),
                "Actual forward writable pages are not exclusively owned")
        require(cpu(params.get("cache_seqlens"), torch.int32, (len(branches),)) == positions,
                "Actual cache sequence lengths differ from planned positions")
        states = params.get("recurrent_states")
        require(type(states) is list and len(states) == len(branches)
                and all(s is b.recurrent and s is not None and s.cache is self.service.backend.cache
                        and type(s.slot) is int and 0 <= s.slot < self.service.backend.cache.num_slots
                        and s.slot not in self.service.backend.cache.free_list and not s.exported
                        and type(s.position) is int and s.position == p
                        and type(s.last_history) is int and s.last_history == 0
                        for s, b, p in zip(states, branches, positions)), "Actual recurrent slots/positions differ from plan")
        slots = [s.slot for s in states]
        require(len(set(slots)) == len(slots), "Forward branches alias recurrent slots")
        expected_keys = {"attn_mode", "cache", "block_table", "cache_seqlens", "last_tokens_only", "recurrent_states"}
        allowed = expected_keys | {"indexed_embeddings", "recurrent_history", "_fuzev_features"}
        if set(params) - allowed:
            raise _StateUnavailable("Unreviewed actual forward position/cache arguments")
        require(params.get("cache") is self.service.backend.cache and params.get("attn_mode") == "flash_attn"
                and type(params.get("last_tokens_only")) is int and params["last_tokens_only"] == 1
                and params.get("_fuzev_features", False) is (plan["features"] and name == "forward"),
                "Actual forward mode changed")
        embeddings = params.get("indexed_embeddings", [])
        require(type(embeddings) in (tuple, list)
                and tuple(map(id, embeddings)) == tuple(map(id, plan["embeddings"]))
                and tuple(map(embedding_identity, embeddings)) == plan["identities"], "Actual image bindings changed")
        kind = "explicit_false" if "recurrent_history" in params else "absent"
        value = original(input_ids, params)
        if params.get("recurrent_history", False) is not False:
            self.forward_flags["rejected"] += 1
            raise _StateUnavailable("Forward changed recurrent_history; non-speculative state is unproven")
        require(params.get("recurrent_states") is states
                and all(type(s.slot) is int and s.slot == slot and type(s.position) is int and s.position == p + width
                        and type(s.last_history) is int and s.last_history == 0 for s, slot, p in zip(states, slots, positions)),
                "Forward did not advance exact recurrent positions")
        # The pinned preparer adds this CPU tensor; get_for_device caches copies without replacing it.
        require(cpu(params.get("recurrent_slots"), torch.int32, (len(branches),)) == slots,
                "Prepared CPU recurrent slots differ from entry ownership")
        require(tuple(map(id, params.get("indexed_embeddings", []))) == tuple(map(id, embeddings))
                and tuple(map(embedding_identity, embeddings)) == plan["identities"], "Forward mutated image binding metadata")
        self.forward_digest.update(json.dumps({"entry": name, "ids": ids, "pages": pages, "positions": positions,
            "slots": slots, "images": plan["identities"], "history": kind}, sort_keys=True).encode())
        plan["offset"] += width
        self.forward_calls[name] += 1
        self.forward_flags[kind] += 1
        return value

    @contextmanager
    def freezes(self):
        original = self.service.lane.freeze
        model = getattr(self.service.lane, "model", None)
        parameters = self.service.backend.parameters
        append, many = self.service.lane.append, self.service.lane.append_features_many

        def bound_parameters(tokens, embeddings, suffix=()):
            from .service_images import embedding_identity
            if getattr(model, "caps", {}).get("mrope"):
                raise _StateUnavailable("Only pinned linear-position Qwen4Exp parameters are witnessed")
            expected, identities = tuple(embeddings), tuple(map(embedding_identity, embeddings))
            result = parameters(tokens, embeddings, suffix)
            require(type(result) is dict and tuple(map(id, result.get("indexed_embeddings", []))) == tuple(map(id, expected))
                    and tuple(map(embedding_identity, expected)) == identities, "Backend parameters changed image bindings")
            self.parameter_plan = (result, expected, identities)
            return result

        def one(branch, tokens, *, features=False, params=None):
            if type(tokens) not in (tuple, list):
                raise _StateUnavailable("Lane witness requires CPU-owned exact token sequences")
            return self._planned((branch,), (tuple(tokens),), features, params,
                                 lambda: append(branch, tokens, features=features, params=params))

        def batch(branches, tails, *, params=None):
            return self._planned(tuple(branches), tuple(tuple(t) for t in tails), True, params,
                                 lambda: many(branches, tails, params=params))

        def entry(name, method):
            def observe(input_ids, params=None):
                try:
                    return self._forward(name, method, input_ids, params)
                except BaseException:
                    self.forward_failed = True
                    raise
            return observe

        def witness(branch):
            before = self.digest(branch)
            if before is not None:
                # Native freeze may allocate its first 128 MiB staging arena plus an owned snapshot.
                size = branch.recurrent.checkpoint_size if branch.recurrent is not None else 0
                self.service.admission((128 << 20) + size, self.service.backend.readout_runtime["memory_fraction"])
            result = original(branch)
            self.unchanged([(branch, before)])
            return result

        with ExitStack() as stack:
            stack.enter_context(_replace(self.service.lane, "freeze", witness))
            for name in ("forward", "prefill"):
                method = getattr(model, name, None)
                if not callable(method):
                    raise _StateUnavailable(f"Missing actual model.{name} entry")
                stack.enter_context(_replace(model, name, entry(name, method)))
            stack.enter_context(_replace(self.service.backend, "parameters", bound_parameters))
            stack.enter_context(_replace(self.service.lane, "append", one))
            stack.enter_context(_replace(self.service.lane, "append_features_many", batch))
            try:
                yield
            finally:
                self.parameter_plan = self.active = None


class _ServingPages(MemoryError):
    """This specific serving plan, not general native-window capability, cannot fit."""


def _compare(left, right, config, label):
    import torch

    require(left.shape == right.shape and left.numel() > 0, f"{label}: shape mismatch/empty tensor")
    require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), f"{label}: nonfinite values")
    delta = (left.double() - right.double()).abs()
    require(bool((delta <= config.atol + config.rtol * right.double().abs()).all()),
            f"{label}: numerical disagreement (max abs {delta.max().item():.9g})")
    return {"max_abs_error": delta.max().item(), "elements": left.numel()}


def _scores(response, request, service):
    import torch

    ids = [candidate.id for candidate in request.candidates]
    require(response["model"] == request.model and response["primitive"] == request.primitive
            and response["head_id"] == service.head.head_id
            and response["head_manifest_sha256"] == service.head.manifest_sha256
            and response["calibration_id"] == service.head.calibration_id, "Decision identity mismatch")
    require([row["id"] for row in response["scores"]] == ids, "Candidate order/coverage mismatch")
    values = [[row[key] for row in response["scores"]]
              for key in ("logit", "raw_probability", "calibrated_probability")]
    require(all(type(value) in (float, int) and math.isfinite(value) for row in values for value in row),
            "Nonfinite or nonnumeric decision score")
    for row in values[1:]:
        require(all(0 <= value <= 1 for value in row) and abs(sum(row) - 1) <= 1e-6,
                "Invalid normalized probabilities")
    if service.head.calibration_id == "uncalibrated:T1":
        from .native_research_qualification import FLAGS
        require(all(response.get(key) is value for key, value in FLAGS.items())
                and values[1] == values[2]
                and torch.allclose(torch.tensor(values[1], dtype=torch.float64),
                                   torch.softmax(torch.tensor(values[0], dtype=torch.float64), dim=0),
                                   atol=1e-6, rtol=1e-5), "Research decision is not a raw-T1 result")
    require(response["selected_id"] == ids[max(range(len(ids)), key=values[0].__getitem__)],
            "Selected candidate disagrees with logits")
    return torch.tensor(values, dtype=torch.float64)


def _decision(service, request, *, state=None, owned=None, cancel=None):
    import torch

    append, score, release = service._append, service.head.score, service.lane.release
    prefix = service._get_prefix(request.prefix_id)
    captured = {"candidate_tokens": [], "appends": []}
    watched = [(prefix.branch, state.digest(prefix.branch))] if state else []
    fork, encode = service.lane.fork, service.backend.encode
    planned_tails = [encode(candidate_suffix(c.model_dump())) for c in request.candidates] if state else []
    keep = False

    def encoded(*args, **kwargs):
        result = encode(*args, **kwargs)
        captured["last_encoded"] = tuple(result)
        return result

    def forked(parent):
        nonlocal keep
        if parent is prefix.branch and state.observer is not None:
            suffix = captured["last_encoded"]
            try:
                plan = state.observer.preflight(prefix, suffix, planned_tails, pending_witness=True)
            except (NotImplementedError, AttributeError) as error:
                raise _StateUnavailable(str(error)) from error
            captured.update(state_preflight=plan, planned_suffix=suffix)
            require(plan["allocated"] is False and plan["window_tokens"] == WINDOW and plan["pending_witness"] is True,
                    "State preflight must not allocate or change the window")
            require(plan["qualified"] is False
                    and plan["longest_tokens"] == len(prefix.accepted) + len(suffix) + max(map(len, planned_tails))
                    and plan["capacity_pages"] == len(service.lane.pool.references), "Stale or mismatched page preflight")
            if plan["serving"]["status"] == "memory_infeasible":
                raise _ServingPages("Specific retained-prefix serving plan cannot fit")
            require(plan["serving"]["status"] == "page_estimate_feasible", "Unknown serving page estimate")
            keep = plan["observer_coexistence"]["status"] == "page_estimate_feasible"
            require(keep or plan["observer_coexistence"]["status"] == "memory_infeasible",
                    "Unknown observer coexistence estimate")
        branch = fork(parent)
        owned.append(branch)  # Track before any observer can raise, even before service receives the branch.
        if state.observer is not None:
            require(state.digest(branch) == state.digest(parent), "Native fork/restore changed state")
            if not any(b is parent for b, _ in watched):
                watched.append((parent, state.digest(parent)))
            state.unchanged(watched)
        return branch

    def trace(branch, tokens, cancel, **kwargs):
        before = tuple(branch.tokens)
        value = append(branch, tokens, cancel, **kwargs)
        captured.update(query=branch, suffix=tuple(tokens), images=tuple(kwargs.get("images", ())))
        captured["appends"].append((before, tuple(tokens), captured["images"]))
        if state:
            state.unchanged(watched)
            digest = state.digest(branch)
            captured.setdefault("state_boundaries", []).append({
                "boundary": "pending_B" if tuple(branch.tokens) == prefix.accepted else "ABC",
                "digest": digest, "parent_and_siblings_checked": state.observer is not None})
            if keep and tuple(branch.tokens) == prefix.accepted:
                # Forking a live branch stashes it; admit that temporary transfer before the fork too.
                service.admission((128 << 20) + branch.recurrent.checkpoint_size,
                                  service.backend.readout_runtime["memory_fraction"])
                pending = fork(branch)
                owned.append(pending)
                require(state.digest(pending) == digest, "Pending-B fork/restore changed state")
                service.lane.freeze(pending)
                watched.append((pending, state.digest(pending)))
                captured["state_pending"] = pending
                state.unchanged(watched)
        return value

    def own(primitive, query, keys):
        captured.update(query_features=_owned(query), candidate_features=_owned(keys),
                        query_tokens=tuple(captured["query"].tokens))
        return score(primitive, query, keys)

    def released(branch):
        if "query" in captured and branch is not captured["query"]:
            captured["candidate_tokens"].append(tuple(branch.tokens))
        if keep and "query" in captured:
            query = captured["query"]
            complete = branch is query and branch.immutable or any(
                tuple(branch.tokens) == tuple(query.tokens) + tail for tail in planned_tails)
            if complete:
                if not branch.immutable:
                    service.lane.freeze(branch)
                state.unchanged(watched)
                saved = state.digest(branch)
                if not any(b is branch for b, _ in watched):
                    watched.append((branch, saved))
                if branch is query:
                    captured["state_query"] = branch
                else:
                    captured.setdefault("state_candidates", []).append(branch)
                return
        return release(branch)

    with ExitStack() as stack:
        for context in (torch.inference_mode(), _replace(service, "_append", trace),
                        _replace(service.head, "score", own), _replace(service.lane, "release", released)):
            stack.enter_context(context)
        if state:
            require(owned is not None, "State observation needs caller-owned branch cleanup")
            stack.enter_context(_replace(service.backend, "encode", encoded))
            stack.enter_context(_replace(service.lane, "fork", forked))
        try:
            response = service.decide(request, cancel or Event(), "qualify_" + uuid4().hex)
        except _ServingPages:
            return {"serving_status": "memory_infeasible", "state_preflight": captured["state_preflight"],
                    "suffix": captured["planned_suffix"], "tails": planned_tails}
    require("query_features" in captured and "query" in captured, "Missing owned decision readout")
    captured.pop("query")
    captured["scores"] = _scores(response, request, service)
    captured["response"] = response
    captured["tails"] = _verify_tokens(service, prefix, request, captured)
    if "state_preflight" in captured:
        require(captured["planned_suffix"] == captured["suffix"] and planned_tails == captured["tails"],
                "Preflight did not use the actual committed C/tails")
    if keep:
        require(len(captured.get("state_candidates", [])) == len(planned_tails)
                and "state_query" in captured and "state_pending" in captured,
                "Missing retained native comparison branch")
        captured["state_watched"] = watched
    captured.pop("last_encoded", None)
    return captured


def _verify_tokens(service, prefix, request, recorded):
    from .service_images import embeddings_for

    accepted, consumed = prefix.accepted, prefix.accepted[:-1]
    images = recorded["images"]
    require(tuple(prefix.branch.tokens) == consumed and prefix.branch.immutable and not prefix.branch.closed,
            "Retained AB checkpoint changed")
    require(tuple(map(id, images[:len(prefix.images)])) == tuple(map(id, prefix.images)),
            "Original AB image bindings were replaced")
    new_images = images[len(prefix.images):]
    require(len(new_images) == int(request.image is not None), "Unexpected C image bindings")
    additional = request.additional_context
    if new_images:
        require(new_images[0].sha256 == request.image.sha256, "C image digest mismatch")
        additional = new_images[0].embedding.text_alias + "\n" + additional
    candidates = [candidate.model_dump() for candidate in request.candidates]
    suffix = service.backend.encode(query_suffix(request.question, candidates, additional),
                                    embeddings_for(images) if request.image else None)
    tails = [service.backend.encode(candidate_suffix(value)) for value in candidates]
    require(recorded["suffix"] == suffix and recorded["query_tokens"] == accepted + suffix,
            "Reused decision did not append exactly C to the accepted AB tokens")
    appends = recorded["appends"]
    require([(before, tokens) for before, tokens, _ in appends] == [(consumed, accepted[-1:]), (accepted, suffix)]
            and tuple(map(id, appends[0][2])) == tuple(map(id, prefix.images)),
            "Pending B token was not consumed exactly once with its original image bindings")
    grouped = [tail for width in dict.fromkeys(map(len, tails)) for tail in tails if len(tail) == width]
    require(recorded["candidate_tokens"] == [accepted + suffix + tail for tail in grouped],
            "Candidate forks did not consume the exact independently encoded tails")
    response = recorded["response"]
    counters = ("logical_prefix_tokens", "reused_tokens", "replayed_tokens", "forwarded_tokens", "consumed_tokens")
    require(type(response["resolved_unconsumed_tokens"]) is int
            and all(type(response["execution"][key]) is int for key in counters), "Nonnumeric token counter receipt")
    require(response["accepted_token_sha256"] == token_hash(accepted)
            and response["query_token_sha256"] == token_hash(accepted + suffix)
            and response["resolved_unconsumed_tokens"] == 1
            and response["prefix_id"] == prefix.id and response["prefix_identity"] == service.identity
            and response["layout"] == "accepted-ab-append-c"
            and response["images"] == [image.receipt() for image in images]
            and response["execution"] == {"mode": "checkpoint", "logical_prefix_tokens": len(accepted),
                "reused_tokens": len(consumed), "replayed_tokens": 0,
                "forwarded_tokens": 1 + len(suffix) + sum(map(len, tails)),
                "consumed_tokens": len(accepted + suffix),
                "feature_tap": "final-mixer/post-collapse/before-lm-head"},
            "Reused ABC execution receipt mismatch")
    return tails


def _fresh(service, prefix, request, recorded, tails, *, state=None):
    """Replay the exact accepted IDs in a new lane, then independent serial tails.

    Deliberately do not call base_tokens, render messages, decode/re-encode B, or
    regenerate B. Reuse the *same* native image binding instances and positions.
    """
    import torch

    query, candidate = None, None
    observations = {"comparisons": [], "boundaries": []}
    watched = list(recorded["state_watched"]) if state else []
    with torch.inference_mode():
        try:
            if state:
                require(recorded["state_preflight"]["observer_coexistence"]["status"] == "page_estimate_feasible",
                        "Oracle forwards require a feasible coexistence preflight")
                state.unchanged(watched)
            query = service.lane.create()
            require(not query.tokens, "Fresh lane did not reset its token boundary")
            if state:
                observations["reset"] = state.reset(query)
                consumed = tuple(prefix.branch.tokens)
                service._append(query, consumed, Event(), features=True, protected=prefix.id, images=prefix.images)
                observations["comparisons"].append({"boundary": "consumed_AB",
                                                    "receipt": state.compare(prefix.branch, query)})
                state.unchanged(watched)
                service._append(query, prefix.accepted[len(consumed):], Event(), features=True,
                                protected=prefix.id, images=prefix.images)
                pending = recorded["state_pending"]
                observations["comparisons"].append({"boundary": "pending_B", "receipt": state.compare(pending, query)})
                state.unchanged(watched)
                observations["boundaries"].append({"boundary": "pending_B", "digest": state.digest(query),
                                                   "parent_and_siblings_checked": True})
                watched = [(b, before) for b, before in watched if b is not pending]
                service._release(pending)
            else:
                service._append(query, prefix.accepted, Event(), features=True,
                                protected=prefix.id, images=prefix.images)
            query_features = _owned(service._append(query, recorded["suffix"], Event(), features=True,
                                    protected=prefix.id, images=recorded["images"])).reshape(-1)
            require(tuple(query.tokens) == recorded["query_tokens"], "Fresh replay changed exact ABC tokens")
            if state:
                observations["comparisons"].append({"boundary": "ABC",
                                                    "receipt": state.compare(recorded["state_query"], query)})
                state.unchanged(watched)
            service.lane.freeze(query)
            if state:
                watched.append((query, state.digest(query)))
                grouped = [i for width in dict.fromkeys(map(len, tails)) for i, tail in enumerate(tails) if len(tail) == width]
                reused_candidates = dict(zip(grouped, recorded["state_candidates"]))
            keys = []
            for index, tail in enumerate(tails):
                candidate = service.lane.fork(query)
                try:
                    if state:
                        require(state.digest(candidate) == state.digest(query), "Fresh candidate restore changed state")
                        state.unchanged(watched)
                    keys.append(_owned(service._append(candidate, tail, Event(), features=True,
                                       protected=prefix.id, images=recorded["images"])).reshape(-1))
                    if state:
                        observations["comparisons"].append({"boundary": "candidate", "index": index,
                            "receipt": state.compare(reused_candidates[index], candidate)})
                        state.unchanged(watched)
                finally:
                    service._release(candidate)
                    candidate = None
            candidates = torch.stack(keys)
            scores = service.head.score(request.primitive, query_features, candidates)
            result = {"query_features": query_features, "candidate_features": candidates,
                      "scores": torch.tensor(scores, dtype=torch.float64)}
            if state:
                result["state_observations"] = observations
            return result
        finally:
            _release_observed(service, [query, candidate])


def _agree(left, right, config):
    errors = {key: _compare(left[key], right[key], config, key)
              for key in ("query_features", "candidate_features", "scores")}
    require(left["scores"][0].argmax().item() == right["scores"][0].argmax().item(),
            "Selected candidate changed between equivalent readouts")
    return errors


class _InjectedFailure(RuntimeError):
    pass


def _fault(service, operation, phase, kind):
    """Inject only AFTER a completed native operation, never simulate kernel proof."""
    cancel, triggered = Event(), []

    def inject():
        triggered.append(True)
        if kind == "cancel":
            cancel.set()
        else:
            raise _InjectedFailure("qualification fault after completed " + phase)

    expected = InterruptedError if kind == "cancel" else _InjectedFailure
    with ExitStack() as stack:
        if phase == "candidate":
            candidate_ids = set()
            parents = {id(prefix.branch) for prefix, _ in service.prefixes.values()}
            fork, append, many = service.lane.fork, service.lane.append, service.lane.append_features_many

            def track(parent):
                branch = fork(parent)
                if id(parent) not in parents:
                    candidate_ids.add(id(branch))
                return branch

            def one(branch, *args, **kwargs):
                value = append(branch, *args, **kwargs)
                if id(branch) in candidate_ids:
                    inject()
                return value

            def batch(branches, *args, **kwargs):
                value = many(branches, *args, **kwargs)
                if any(id(branch) in candidate_ids for branch in branches):
                    inject()
                return value

            for name, replacement in (("fork", track), ("append", one), ("append_features_many", batch)):
                stack.enter_context(_replace(service.lane, name, replacement))
        else:
            name = "score" if phase == "head" else "_append"
            target = service.head if phase == "head" else service
            original = getattr(target, name)

            def after(*args, **kwargs):
                value = original(*args, **kwargs)
                inject()
                return value

            stack.enter_context(_replace(target, name, after))
        try:
            operation(cancel)
        except expected:
            require(bool(triggered), "Fault occurred before the requested injection boundary")
        else:
            raise ValueError("Injected cancellation/failure was not propagated")


def _request(service, spec, prefix_id):
    return ServiceDecisionRequest.model_validate(spec.request(service.model_name, service.head.head_id,
                                                            prefix_id=prefix_id))


def _release_observed(service, branches):
    errors = []
    for branch in reversed(branches):
        try:
            service._release(branch)
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("Observed branch cleanup failed: " + "; ".join(errors))


def _case(service, case, config, *, full_window=False, state=None):
    with state.freezes() if state else nullcontext():
        return _case_run(service, case, config, full_window=full_window, state=state)


def _case_run(service, case, config, *, full_window=False, state=None):
    initial = _resources(service)
    generated = service.generate(case.generation, Event(), "qualify_" + uuid4().hex)
    prefix = service._get_prefix(generated["prefix_id"])
    accepted, consumed = prefix.accepted, tuple(prefix.branch.tokens)
    completions = generated["completion_token_ids"]
    require(len(accepted) > len(completions) and all(type(token) is int and token >= 0 for token in accepted)
            and type(generated["consumed_tokens"]) is int
            and all(type(value) is int for value in generated["usage"].values()), "Invalid generation token receipt")
    require(completions and all(type(token) is int and token >= 0 for token in completions)
            and tuple(completions) == accepted[-len(completions):], "Accepted B token mismatch")
    require(generated["accepted_token_sha256"] == token_hash(accepted)
            and consumed == accepted[:-1] and generated["consumed_tokens"] == len(consumed)
            and generated["unconsumed_token_ids"] == list(accepted[-1:]), "Accepted/consumed boundary mismatch")
    require(generated["usage"] == {"prompt_tokens": len(accepted) - len(completions),
            "completion_tokens": len(completions), "total_tokens": len(accepted)}, "Generation usage mismatch")
    require(generated["prefix_identity"] == service.identity
            and generated["images"] == [image.receipt() for image in prefix.images]
            and len(completions) <= case.generation.max_tokens, "Generation identity/image/budget mismatch")
    eos = service.backend.config.eos_token_id_list
    require(not any(token in eos for token in completions[:-1]), "Generation accepted tokens after EOS")
    boundary = ("eos" if completions[-1] in eos
                else "stop" if generated["matched_stop"] is not None else "length")
    require(generated["choices"][0]["finish_reason"] == ("length" if boundary == "length" else "stop")
            and (boundary != "length" or len(completions) == case.generation.max_tokens)
            and (generated["matched_stop"] is None or generated["matched_stop"] in case.generation.stop),
            "Generation finish/stop receipt mismatch")
    retained = _resources(service)
    parent = [(prefix.branch, state.digest(prefix.branch))] if state else []
    results, references, requests, longest = [], [], [], 0
    for spec in case.decisions:
        request = _request(service, spec, prefix.id)
        requests.append(request)
        owned = []
        try:
            recorded = _decision(service, request, state=state, owned=owned)
            if recorded.get("serving_status") == "memory_infeasible":
                results.append({"primitive": request.primitive.value, "serving_status": "memory_infeasible",
                                "state_preflight": recorded["state_preflight"],
                                "fresh_vs_reused": {"status": "not_run", "reason": "serving_page_capacity"}})
                references.append(None)
                state.unchanged(parent)
                require(_resources(service) == retained, "Serving preflight refusal leaked native resources")
                continue
            suffix, tails = recorded["suffix"], recorded["tails"]
            longest = max(longest, len(accepted + suffix) + max(map(len, tails)))
            result = {"primitive": request.primitive.value, "query_token_sha256": token_hash(accepted + suffix)}
            fresh = None
            if state:
                plan = recorded.get("state_preflight")
                result.update(serving_status="observed", state_preflight=plan,
                              reused_state_boundaries=recorded.get("state_boundaries", []))
                if plan and plan["observer_coexistence"]["status"] == "page_estimate_feasible":
                    try:
                        fresh = _fresh(service, prefix, request, recorded, tails, state=state)
                    except MemoryError as error:
                        result["oracle_status"] = {"status": "memory_infeasible", "reason": str(error),
                                                   "scope": "observer_oracle_not_serving"}
                else:
                    result["oracle_status"] = {"status": "memory_infeasible" if plan else "incomplete",
                                               "reason": "observer_coexistence" if plan else state.reason,
                                               "scope": "observer_oracle_not_serving"}
            else:
                fresh = _fresh(service, prefix, request, recorded, tails)
            result["fresh_vs_reused"] = (_agree(recorded, fresh, config) if fresh else
                {"status": "not_run", "reason": "No admitted complete-state oracle"})
            if state and fresh:
                result.update(oracle_status={"status": "observed", "scope": "observer_oracle_not_serving"},
                              state_observations=fresh["state_observations"])
            results.append(result)
            # Keep only readouts for later stability checks, never borrowed branches or image bindings.
            reference = fresh if fresh is not None else recorded
            references.append({key: reference[key] for key in ("query_features", "candidate_features", "scores")})
        finally:
            _release_observed(service, owned)
        if state:
            state.unchanged(parent)
        require(_resources(service) == retained, "Fresh/reused readout leaked native resources")
    serving_blocked = any(result.get("serving_status") == "memory_infeasible" for result in results)
    require(not full_window or longest == WINDOW or serving_blocked,
            "Full-window case did not exercise exactly 262144 tokens")
    for index in reversed(range(4)):
        if references[index] is None:
            continue
        again = _decision(service, requests[index])
        # New image token ranges can be reallocated; compare numerics, not their allocation IDs.
        results[index]["after_other_branches"] = _agree(again, references[index], config)
        require(prefix.accepted == accepted and tuple(prefix.branch.tokens) == consumed
                and prefix.branch.immutable and not prefix.branch.closed, "Retained AB was mutated")
        if state:
            state.unchanged(parent)
        require(_resources(service) == retained, "Interleaved decision leaked native resources")
    recovery = next((i for i, reference in enumerate(references) if reference is not None), None)
    for phase in (("forward", "candidate", "head") if recovery is not None else ()):
        for kind in ("cancel", "failure"):
            _fault(service, lambda cancel: service.decide(requests[recovery], cancel, "qualify_fault"), phase, kind)
            require(_resources(service) == retained, "Fault cleanup changed the retained resource ledger")
            _agree(_decision(service, requests[recovery]), references[recovery], config)
            if state:
                state.unchanged(parent)
    service.release_prefix(prefix.id)
    require(_resources(service) == initial, "Prefix release did not restore native resources")
    for kind in ("cancel", "failure"):
        _fault(service, lambda cancel: service.generate(case.generation, cancel, "qualify_fault"), "forward", kind)
        require(_resources(service) == initial, "Generation fault leaked native resources")
    hybrid = {"status": "unavailable", "reason": STATE_GAP}
    if state:
        hybrid = {"status": "observed" if all("state_observations" in result for result in results) else "incomplete",
                  "evidence_kind": state.evidence_kind, "qualified": False, "reason": state.reason,
                  "scope": "consumed AB / pending-B / ABC / candidate state comparisons plus fresh-slot active reset"}
    return {"name": case.name, "status": "incomplete" if state and hybrid["status"] != "observed" else "passed",
            "input_origin": case.input_origin,
            "accepted_token_sha256": token_hash(accepted), "accepted_tokens": len(accepted),
            "consumed_tokens": len(consumed), "pending_token_ids": generated["unconsumed_token_ids"],
            "boundary": boundary, "max_exercised_tokens": longest, "branches": results,
            "fault_checks": ("generation-forward and decision-forward/candidate/head; cancellation and injected failure"
                             if recovery is not None else "generation-forward only; serving page preflight blocked decisions"),
            "hybrid_state": hybrid,
            **({"serving_status": "memory_infeasible" if serving_blocked else "observed"} if state else {})}


def _image_control(service, control, config):
    before = _resources(service)
    values, outputs = {}, {}
    for name, image in (("real", control.request.image), ("blind", None),
                        ("mismatched", control.mismatched_image), ("repeat", control.request.image)):
        request = ServiceDecisionRequest.model_validate({**control.request.model_dump(), "image": image})
        response = service.decide(request, Event(), "qualify_image_" + uuid4().hex)
        values[name] = _scores(response, request, service)
        require([value["rgb_sha256"] for value in response["images"]] == ([image.sha256] if image else []),
                "Image-control receipt mismatch")
        outputs[name] = {"selected_id": response["selected_id"], "scores": response["scores"],
                         "images": response["images"]}
        require(_resources(service) == before, "Image control leaked native resources")
    _compare(values["real"], values["repeat"], config, "real-image repeat")
    require(outputs["real"]["selected_id"] == control.expected_id, "Real-image control answer mismatch")
    deltas = {key: (values["real"] - values[key]).abs().max().item() for key in ("blind", "mismatched")}
    sensitivity = {key: bool(((values["real"] - values[key]).abs()
                             > config.atol + config.rtol * values[key].abs()).any()) for key in deltas}
    return {"name": control.name, "status": "measured", "input_origin": control.input_origin,
            "controls": outputs, "max_score_deltas": deltas, "sensitivity_observed": sensitivity,
            "quality_qualified": False, "reason": "Paired controls are not a held-out image quality gate"}


def _frozen_head(service, *, research_cuda=False):
    if research_cuda:
        from .native_research_qualification import head_snapshot
        return head_snapshot(service.head)
    import torch

    head = service.head
    require(not head.bundle.training and all(not parameter.requires_grad and parameter.device.type == "cpu"
            and parameter.dtype == torch.float32 for parameter in head.bundle.parameters()),
            "Require the frozen CPU FP32 serving head, not training or alternate semantics")
    state = head.bundle.state_dict()
    size = sum(value.numel() * value.element_size() for value in state.values())
    require(0 < size <= 64 << 20, "Frozen-head snapshot exceeds its bound")
    admit(size, service.backend.readout_runtime["memory_fraction"])
    require(all(bool(torch.isfinite(value).all()) for value in state.values()), "Nonfinite frozen head")
    return {"weights": {key: value.clone() for key, value in state.items()},
            "temperatures": deepcopy(head.temperatures), "head_id": head.head_id,
            "manifest": head.manifest_sha256, "calibration": head.calibration_id}


def _unchanged_head(service, before, *, research_cuda=False):
    import torch

    after = _frozen_head(service, research_cuda=True) if research_cuda else _frozen_head(service)
    left, right = before.pop("weights"), after.pop("weights")
    require(before == after and left.keys() == right.keys()
            and all(torch.equal(value, right[key]) for key, value in left.items()), "Frozen head changed during qualification")


def run_qualification(service, config, *, allow_inference=False, cpu_fake=False, research_cuda=False):
    """Run synchronously on the existing exclusive owner; return a JSON-safe report.

    The caller owns backend/service construction and final close. No live server
    may share this service. Even a successful run remains incomplete while the
    complete-state observer and required real-runtime gates are unavailable.
    """
    require(allow_inference is True, "Native qualification requires explicit inference opt-in")
    require(type(cpu_fake) is bool, "cpu_fake must be an explicit boolean evidence label")
    require(type(research_cuda) is bool, "Research CUDA policy must be an explicit boolean")
    config = QualificationConfig.model_validate(config)
    service._owner()
    require(not service.prefixes and not service.backend.prefixes and not service.backend.images
            and not any(not branch.closed for branch in service.lane.branches),
            "Qualification requires an empty, exclusive service, never a live producer")
    fraction = service.backend.readout_runtime["memory_fraction"]
    require(fraction is None or (type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1),
            "Declared native memory fraction must be in (0, 1]")
    require(service.lane.context_limit == WINDOW and service.backend.readout_runtime.get("context_tokens") == WINDOW,
            "Native context must remain 262144; no window downgrade")
    require(service.lane.chunk_size == service.backend.readout_runtime.get("feature_chunk_tokens"),
            "Actual native chunk geometry mismatch")
    service_guard_policy = verify_memory_guard(service.guard, fraction) if not cpu_fake else None
    cases = config.cases + ([config.full_window.case] if config.full_window else [])
    role = "cpu-fake" if cpu_fake else "operator-non-heldout"
    require(all(case.input_role == role for case in cases + config.image_controls), "Input role/evidence mismatch")
    require(all(case.generation.model == service.model_name for case in cases)
            and all(control.request.model == service.model_name and control.request.head_id == service.head.head_id
                    for control in config.image_controls), "Case model/head mismatch")
    report = {"schema_version": 1, "status": "incomplete", "qualified": False,
              "evidence_kind": "cpu_fake" if cpu_fake else "native", "cases": [], "image_controls": [],
              "head_policy": "kev-cuda-raw-t1" if research_cuda else "legacy-calibrated",
              "tolerances": {"atol": config.atol, "rtol": config.rtol}, "cleanup_errors": [],
              "gates": {"complete_hybrid_state": {"status": "unavailable", "reason": STATE_GAP},
                        "full_window": {"status": "not_requested", "window_tokens": WINDOW},
                        "real_image_controls": {"status": "not_run"},
                        "primitives": {"status": "not_run"},
                        "accepted_boundaries": {"status": "not_run"}},
              "limitations": ["No weight-byte verification; storage dtype evidence exists only in successful state-observer receipts",
                  "No kernel-failure, scheduler/queue, performance, transfer, or held-out quality proof",
                  "Fresh-slot active reset is observed only in successful opt-in state receipts; no whole-cache zero claim",
                  "Input origin is operator attestation; no dataset is opened or scanned"]}
    if service_guard_policy is not None:
        report["service_memory_guard"] = service_guard_policy
    guard = MemoryGuard(fraction=fraction)
    run_guard_policy = None
    baseline = metadata = head = None
    state = None
    try:
        guard.start()
        if not cpu_fake:
            run_guard_policy = verify_memory_guard(guard, fraction)
        admit(0, fraction)
        metadata = _metadata(service, cpu_fake, research_cuda=True) if research_cuda else _metadata(service, cpu_fake)
        report["metadata"] = metadata
        baseline = _resources(service)
        head = _frozen_head(service, research_cuda=True) if research_cuda else _frozen_head(service)
        if config.observe_native_state:
            state = _StateObservation(service, config, cpu_fake)
            report["gates"]["complete_hybrid_state"] = {
                "status": "incomplete", "qualified": False, "state_observations": "incomplete",
                "reason": state.reason or "State observations have not completed"}
            report["gates"]["forward_arguments"] = {
                "status": "incomplete", "reason": "Actual forward and prefill chunk witnesses have not completed"}
        for case in config.cases:
            guard.check()
            report["cases"].append(_case(service, case, config, state=state))
        for control in config.image_controls:
            guard.check()
            report["image_controls"].append(_image_control(service, control, config))
        if config.image_controls:
            report["gates"]["real_image_controls"] = {"status": "cpu_fake" if cpu_fake else "measured",
                                                      "quality_qualified": False}
        import psutil
        memory = psutil.virtual_memory()
        full = full_window_feasibility(config.full_window, enabled=config.run_full_window,
            frozen_head_ready=not cpu_fake, total=memory.total, available=memory.available, fraction=fraction)
        if service_guard_policy is not None and service_guard_policy["enabled"] is False:
            full["host_ram_admission"] = {"enabled": False, "operator_sha256": service_guard_policy["operator_sha256"],
                                          "estimate_status": full["status"], "estimate_reason": full.get("reason")}
            if full["status"] == "memory_infeasible" and full.get("reason") == "would_breach_shared_headroom":
                full.update(status="estimate_only", reason="approved_host_ram_admission_removed")
        report["gates"]["full_window"] = full
        if full["status"] in {"estimate_feasible", "estimate_only"}:
            try:
                guard.check()
                admit(config.full_window.additional_peak_bytes, fraction)
                full.update(allocation_attempted=True, allocated=None)
                report["cases"].append(_case(service, config.full_window.case, config, full_window=True, state=state))
            except MemoryError as error:
                full.update(status="incomplete" if state else "memory_infeasible", reason=str(error),
                            scope="state_enabled_case_unclassified_memory_phase" if state else "partial_mode")
            except (Exception, KeyboardInterrupt) as error:
                full.update(status="incomplete" if isinstance(error, _StateUnavailable) else "failed", reason=str(error))
                raise
            else:
                full.update(status="passed", allocated=True)
                if state:
                    completed = report["cases"][-1]
                    oracle = [branch.get("oracle_status", {}).get("status") for branch in completed["branches"]]
                    full.update(scope="specific_full_window_serving_case_only",
                        status="memory_infeasible" if completed["serving_status"] == "memory_infeasible" else "passed",
                        observer_coexistence={"status": "memory_infeasible" if "memory_infeasible" in oracle
                                              else "observed" if all(v == "observed" for v in oracle) else "incomplete"})
        if state:
            observed = bool(report["cases"]) and all(case["hybrid_state"]["status"] == "observed" for case in report["cases"])
            report["gates"]["complete_hybrid_state"] = {
                "status": "incomplete", "qualified": False,
                "state_observations": ("cpu_fake_observed" if cpu_fake else "observed") if observed else "incomplete",
                "reason": state.reason or "Observed state boundaries are not native runtime/image/failure/queue/weights/full-window qualification"}
        boundaries = sorted({case["boundary"] for case in report["cases"]})
        report["gates"]["accepted_boundaries"] = {"status": "passed" if set(boundaries) == {"eos", "stop", "length"}
                                                  else "incomplete", "observed": boundaries}
        primitives = sorted({branch["primitive"] for case in report["cases"] for branch in case["branches"]})
        report["gates"]["primitives"] = {"status": "passed" if set(primitives) == {"choice", "noul", "ordinal"}
                                         else "incomplete", "observed": primitives}
        report["execution_status"] = "memory_infeasible" if full["status"] == "memory_infeasible" else "passed"
        if state and (any(case["status"] == "incomplete" for case in report["cases"]) or full["status"] == "incomplete"):
            report["execution_status"] = "incomplete"
        if full["status"] == "memory_infeasible":
            report["status"] = "memory_infeasible"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="cancelled" if isinstance(error, (InterruptedError, KeyboardInterrupt))
                      else "incomplete" if isinstance(error, _StateUnavailable)
                      else "memory_infeasible" if isinstance(error, MemoryError) else "failed",
                      execution_status="incomplete" if isinstance(error, _StateUnavailable) else "failed",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if state:
            counts = state.forward_flags
            report["gates"]["forward_arguments"] = {
                "status": ("cpu_fake_observed" if cpu_fake else "observed")
                          if all(state.forward_calls.values()) and not state.forward_failed else "incomplete",
                "qualified": False, "evidence_kind": state.evidence_kind, "completed_calls": dict(state.forward_calls),
                "arguments_sha256": state.forward_digest.hexdigest(),
                "scope": "CPU model.forward/prefill entry metadata bound to active lane chunks, not CUDA/kernel argument proof",
                "reason": "Both entry paths must actually execute; unexercised prefill stays incomplete"}
            report["gates"]["non_speculative_forward_flags"] = {
                "status": ("cpu_fake_observed" if cpu_fake else "observed")
                          if all(state.forward_calls.values()) and not state.forward_failed and not counts["rejected"]
                          and (cpu_fake or state.default_semantics_verified) else "incomplete",
                "qualified": False, "evidence_kind": state.evidence_kind, "flag_counts": dict(counts),
                "source_default_verified": state.default_semantics_verified,
                "completed_calls": dict(state.forward_calls),
                "scope": "State-enabled case model.forward/prefill entry/return flags, not kernel arguments",
                "default_semantics": "Pinned ba606 GDN/PLE params.get('recurrent_history', False); last_history=0 alone is insufficient"}
        # Preflight requires an empty service, so every remaining service prefix is ours.
        for prefix_id in list(service.prefixes):
            try:
                service.release_prefix(prefix_id)
            except Exception as error:
                report["cleanup_errors"].append(f"prefix {prefix_id}: {error}")
        survivors = [branch for branch in list(service.lane.branches) if not branch.closed]
        if survivors:
            report["cleanup_errors"].append("Service left live branches; attempting final harness release")
        for branch in survivors:
            try:
                service.lane.release(branch)
            except Exception as error:
                report["cleanup_errors"].append(f"branch release: {error}")
        if baseline is not None:
            try:
                require(_resources(service) == baseline, "Final native resource ledger was not restored")
                if head is not None:
                    if research_cuda:
                        _unchanged_head(service, head, research_cuda=True)
                    else:
                        _unchanged_head(service, head)
                after_metadata = (_metadata(service, cpu_fake, research_cuda=True) if research_cuda
                                  else _metadata(service, cpu_fake))
                require(after_metadata == metadata, "Runtime provenance changed during qualification")
                if not cpu_fake:
                    require(verify_memory_guard(service.guard, fraction) == service_guard_policy,
                            "Service host memory-guard policy changed during qualification")
                    require(verify_memory_guard(guard, fraction) == run_guard_policy,
                            "Harness host memory-guard policy changed during qualification")
                guard.check()
            except Exception as error:
                report["cleanup_errors"].append(str(error))
        report["memory_guard"] = {**guard.snapshot(), **(run_guard_policy or {})}
        guard.close()
        if report["cleanup_errors"]:
            report.update(status="failed", execution_status="failed")
    return report
