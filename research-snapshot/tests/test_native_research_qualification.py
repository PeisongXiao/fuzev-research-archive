"""Synthetic CPU tests only; native/CUDA provenance is mocked, never GPU proof."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from threading import Event
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save

from fuzev_research import native_qualification as qualification
from fuzev_research import native_research_qualification as binding
from fuzev_research import kev_research_server as research
from fuzev_research.encoding import candidate_suffix, query_suffix
from fuzev_research.service_contracts import GenerationRequest
from test_native_qualification import case
from test_native_qualification import service as legacy_service  # noqa: F401 -- pytest fixture registration


@pytest.fixture(autouse=True)
def no_cuda():
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from fuzev_research import native

    root = tmp_path / "root"
    (root / "reports").mkdir(parents=True)
    monkeypatch.setattr(binding, "ROOT", root)
    monkeypatch.setenv("FUZEV_ROOT", str(root))
    source = ModuleType("_synthetic_a")
    source.__file__ = str(root / "frozen/fuzev_research/shared_head.py")
    monkeypatch.setitem(sys.modules, source.__name__, source)
    bundle_class = type("SyntheticBundle", (torch.nn.Module,), {"__module__": source.__name__})
    bundle = bundle_class().eval()
    for i in range(15):
        length = binding.PARAMETER_COUNT - 14 if i == 0 else 1
        bundle.register_parameter("p" + str(i), torch.nn.Parameter(torch.zeros(length), requires_grad=False))
    head = object.__new__(research.ResearchHead)
    head.bundle, head.head_id = bundle, "reference"
    head.weights_sha256, head.manifest_sha256 = binding.HEAD_SHA, binding.PLAN_SHA
    head.calibration_id, head.labels = "uncalibrated:T1", list(range(16))
    head.lm = NS(caps={"logits_output": True})
    backend = NS(config=NS(directory=str(root / "snapshot"), infer_params=NS(ngram_stream_from_disk=True)),
                 cache=NS(num_slots=8, max_num_tokens=294912), model=NS(fwd_modules=[(head.lm, None, None)]),
                 preparer=NS(encode=lambda text: [ord(text[-1]) - 65]))
    head.backend = backend
    monkeypatch.setattr(native, "NativeBackend", type(backend))
    sources = {"recipes": "synthetic-committed", "exllamav3-fuzev": binding.NATIVE_COMMIT}
    monkeypatch.setattr(native, "runtime_sources", lambda: dict(sources))
    runtime = {"feature_chunk_tokens": 8192, "cache_tokens": 294912, "max_hybrid_slots": 8,
               "native_extension_sha256": binding.EXTENSION_SHA,
               "context_tokens": 262144, "candidate_batch_size": 8, "batch_capacity": 8,
               "model_batch_capacity": 8, "memory_fraction": .85, "optimized_foundation": True}
    backend.readout_runtime = runtime
    original = {"readout_runtime": runtime, "encoding": {"synthetic": True}}
    model = {"local_snapshot": str(root / "snapshot")}
    reference = root / "identity.json"
    plan = {"original_feature_identity": str(reference)}
    documents = {str(binding.PLAN): plan, str(reference): original,
                 str(research.ACQUISITION): {"models": {"native-4.05": model}}}
    monkeypatch.setattr(research, "read", lambda path: deepcopy(documents[str(path)]))
    monkeypatch.setattr(research, "serving_identity_for", lambda *args: {"synthetic": "native4"})
    from fuzev_research import encoding
    monkeypatch.setattr(encoding, "identity", lambda: {"synthetic": True})
    module = ModuleType("exllamav3.ext")
    module.exllamav3_ext = NS(__file__=str(root / "native.so"))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    hashes = {str(binding.PLAN): binding.PLAN_SHA, str(binding.QUALITY): binding.QUALITY_SHA,
              module.exllamav3_ext.__file__: binding.EXTENSION_SHA}
    monkeypatch.setattr(research, "sha", lambda path: hashes.get(str(path), "a" * 64))
    paths = {binding.PLAN, binding.QUALITY, reference, research.ACQUISITION, Path(binding.__file__),
             Path(research.__file__)}
    paths.update(Path(binding.__file__).with_name(name) for name in (
        "native_qualification.py", "native_state_observer.py", "native.py", "native_service.py", "server.py",
        "contracts.py", "service_contracts.py", "service_images.py", "encoding.py", "preparation.py"))
    pins = {str(path): research.sha(path) for path in paths}
    service = object.__new__(research.ResearchService)
    service.head, service.backend, service.model_name = head, backend, "native-4.05"
    service.lane = NS(context_limit=262144, chunk_size=8192)
    service.provenance = {**binding.FLAGS, "head_execution_device": "cuda:0", "head_lane": "lm-assisted",
        "learned_head_count": 1, "head_sha256": binding.HEAD_SHA, "plan_sha256": binding.PLAN_SHA,
        "quality_sha256": binding.QUALITY_SHA, "calibration_id": "uncalibrated:T1", "raw_inference_temperature": 1.,
        "recipe_sources": sources, "feature_identity": original, "reference_feature_identity": original,
        "serving_identity": {"synthetic": "native4"}, "readout_runtime": runtime, "source_pins": pins,
        "acquisition_manifest_sha256": "a" * 64}
    checkpoint = save(bundle.state_dict())
    artifact = {"tree": str(root / "frozen")}
    monkeypatch.setattr(binding, "_frozen_artifacts", lambda: (artifact, checkpoint))
    return NS(root=root, head=head, service=service, backend=backend, documents=documents)


def test_research_snapshot_accepts_frozen_fp32_cpu_head(rig):
    snapshot = binding.head_snapshot(rig.head)
    assert all(value.device.type == "cpu" for value in snapshot["weights"].values())


def test_exact_research_metadata_and_owned_head_snapshot_with_mock_cuda(rig, monkeypatch):
    monkeypatch.setattr(binding, "_cuda_parameters", lambda bundle: None)  # Test-only; no CUDA qualification.
    metadata = qualification._metadata(rig.service, False, research_cuda=True)
    assert metadata["head_checkpoint_verified"] and metadata["head_execution_device"] == "cpu"
    before = qualification._frozen_head(rig.service, research_cuda=True)
    assert all(value.data_ptr() != rig.head.bundle.state_dict()[name].data_ptr()
               for name, value in before["weights"].items())
    rig.head.bundle.p1.add_(1)
    with pytest.raises(ValueError, match="Frozen head changed"):
        qualification._unchanged_head(rig.service, before, research_cuda=True)
    assert not binding.research_metadata(rig.service)["head_checkpoint_verified"]


@pytest.mark.parametrize("change", ["cal", "labels", "lm", "class", "score", "ple"])
def test_research_binding_rejects_incompatible_provenance(rig, monkeypatch, change):
    monkeypatch.setattr(binding, "_cuda_parameters", lambda bundle: None)
    if change == "cal":
        rig.head.calibration_id = "sha256:legacy"
    elif change == "labels":
        rig.head.labels.reverse()
    elif change == "lm":
        rig.head.lm = NS()
    elif change == "class":
        rig.service = NS(**rig.service.__dict__)
    elif change == "score":
        rig.head.score = lambda *args: None
    else:
        rig.backend.config.infer_params.ngram_stream_from_disk = False
    with pytest.raises(ValueError):
        binding.research_metadata(rig.service)


def test_canonical_research_score_projects_query_lm_once_with_no_math_rewrite(rig, monkeypatch):
    original_to = torch.Tensor.to

    def cpu_only_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only_to)
    projections, calls = [], []

    def project(value, params):
        projections.append((value.clone(), params))
        return torch.arange(16, dtype=torch.float16).reshape(1, 1, 16)

    def shared(primitive, query, keys, *, baseline):
        calls.append((primitive, query.clone(), keys.clone(), baseline.clone()))
        return baseline

    rig.head.lm.forward, rig.head.bundle = project, shared
    logits, raw, legacy = rig.head.score("choice", torch.ones(2560), torch.ones(2, 2560))
    assert logits == [0., 1.] and raw == legacy
    assert len(projections) == len(calls) == 1
    assert projections[0][0].shape == (1, 1, 2560) and projections[0][0].dtype == torch.float16
    assert calls[0][1].shape == (1, 2560) and calls[0][2].shape == (1, 2, 2560)
    assert torch.equal(calls[0][3], torch.tensor([[0., 1.]]))


@pytest.mark.parametrize("reverse", [False, True])
def test_recorded_and_fresh_oracles_use_canonical_criterion_and_one_query_baseline(
        rig, legacy_service, monkeypatch, reverse):
    service = legacy_service
    service.__class__ = research.ResearchService  # Synthetic CPU service, not real CUDA admission.
    service.head = rig.head
    rig.head.backend = service.backend
    original_to = torch.Tensor.to

    def cpu_only_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_only_to)
    projections, calls = [], []

    def project(value, params):
        projections.append((value.clone(), params))
        return torch.arange(16, dtype=torch.float16).reshape(1, 1, 16)

    def shared(primitive, query, keys, *, baseline):
        calls.append((primitive, query.clone(), keys.clone(), baseline.clone()))
        return baseline

    rig.head.lm.forward, rig.head.bundle = project, shared
    # Nonidentity token IDs catch substituting label positions for the LM vocabulary mapping.
    rig.head.labels = [(index + 5) % 16 for index in range(16)]
    item = case()
    item["decisions"][0]["question"] = "Which option meets the stated criterion?"
    item["decisions"][0]["additional_context"] = "The criterion is the exact accepted evidence."
    if reverse:
        item["decisions"][0]["candidates"].reverse()
    config = qualification.QualificationConfig.model_validate({"cases": [item]})
    generated = service.generate(GenerationRequest.model_validate(item["generation"]), Event(), "synthetic")
    prefix = service._get_prefix(generated["prefix_id"])
    try:
        request = qualification._request(service, config.cases[0].decisions[0], prefix.id)
        recorded = qualification._decision(service, request)
        assert len(projections) == len(calls) == 1
        fresh = qualification._fresh(service, prefix, request, recorded, recorded["tails"])
        assert len(projections) == len(calls) == 2  # One per oracle, never one per key or extra baseline.
        rendered = query_suffix(request.question, [candidate.model_dump() for candidate in request.candidates],
                                request.additional_context)
        assert "Question: " + request.question in rendered
        assert "Additional evidence:\n" + request.additional_context in rendered
        assert "Candidates:\n(A) " + request.candidates[0].text + "\n(B) " + request.candidates[1].text in rendered
        assert recorded["suffix"] == tuple(service.backend.encode(rendered))
        assert recorded["query_tokens"] == prefix.accepted + recorded["suffix"]
        assert recorded["tails"] == [service.backend.encode(candidate_suffix(candidate.model_dump()))
                                     for candidate in request.candidates]
        assert [row["id"] for row in recorded["response"]["scores"]] == [c.id for c in request.candidates]
        assert recorded["response"]["selected_id"] == request.candidates[1].id
        for projection, call in zip(projections, calls):
            assert projection[0].shape == (1, 1, 2560) and projection[0].dtype == torch.float16
            assert projection[1] == {} and call[0] == request.primitive
            assert torch.equal(projection[0].reshape(-1), recorded["query_features"].half())
            assert torch.equal(call[2].squeeze(0), recorded["candidate_features"])
            assert torch.equal(call[3], torch.tensor([[5., 6.]]))
        assert all(row["max_abs_error"] == 0 for row in qualification._agree(recorded, fresh, config).values())
        assert "score" not in vars(service.head) and "_append" not in vars(service)
    finally:
        service.release_prefix(prefix.id)


@pytest.mark.parametrize("mode", ["procedural", "full-window"])
@pytest.mark.parametrize("close_failure", [False, True])
def test_offline_operator_modes_cleanup_before_final_report(rig, monkeypatch, mode, close_failure):
    from fuzev_research import native
    events = []
    item = case()
    item["input_role"] = "operator-non-heldout"
    full = deepcopy(item)
    full["name"] = "window"
    document = {"cases": [item], "observe_native_state": True,
                "full_window": {"case": full, "additional_peak_bytes": 1, "estimate_basis": "CPU fixture only"}}
    cases = rig.root / "cases.json"
    cases.write_text(json.dumps(document))
    output = rig.root / "reports/result.json"
    original_sha = research.sha
    monkeypatch.setattr(research, "sha", lambda path: binding.hashlib.sha256(cases.read_bytes()).hexdigest()
                        if Path(path) == cases else original_sha(path))

    class Backend:
        def __init__(self, *args, **kwargs):
            events.append("construct")

        def close(self):
            events.append("backend-close")

    def close():
        events.append("service-close")
        assert not output.read_text()
        if close_failure:
            raise RuntimeError("synthetic close failed")

    monkeypatch.setattr(native, "NativeBackend", Backend)
    monkeypatch.setattr(research, "build_research_service", lambda *args: NS(close=close))

    def run(service, config, **kwargs):
        assert kwargs == {"allow_inference": True, "research_cuda": True}
        assert config.run_full_window is (mode == "full-window")
        assert (config.full_window is not None) is (mode == "full-window")
        return {"status": "incomplete", "execution_status": "passed", "qualified": False, "cleanup_errors": []}

    monkeypatch.setattr(qualification, "run_qualification", run)
    args = NS(allow_inference=True, mode=mode, cases=cases, output=output)
    assert binding._run(args) == int(close_failure)
    assert events == ["construct", "service-close", "backend-close"]
    result = json.loads(output.read_text())
    assert result["qualified"] is False and result["operator_mode"] == mode
    assert result["execution_status"] == ("failed" if close_failure else "passed")
    with pytest.raises(FileExistsError):
        binding._run(args)
    assert len(events) == 3
    output.unlink()
    target = output.with_name("absent-report.json")
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        binding._run(args)
    assert not target.exists() and len(events) == 3


def test_disabled_operator_does_not_read_inputs_or_construct_engine():
    with pytest.raises(ValueError, match="opt-in"):
        binding._run(NS(allow_inference=False))
