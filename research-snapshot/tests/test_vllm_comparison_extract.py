"""Tiny CPU fakes only; never instantiate a backbone or read project datasets."""

import asyncio
from contextlib import asynccontextmanager, closing, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from test_evaluate import fixture as fixture
from test_vllm_images import ImageTokenizer, Renderer, acquired
from test_vllm_service import Engine, Guard, config
from fuzev_research import vllm_comparison_extract as module
from fuzev_research.comparison_inputs import INPUTS, MANIFEST, iter_inputs, prepare_inputs
from fuzev_research.comparison_seal import seal_features
from fuzev_research.encoding import candidate_suffix, query_suffix, token_hash
from fuzev_research.evaluate import SUBSET_FILE, TEST_SCOPE, require_comparison_complete, require_complete
from fuzev_research.features import FeatureStore, file_sha256


@pytest.fixture(autouse=True)
def cleanup_tmp(tmp_path):
    assert not torch.cuda.is_available()
    yield
    shutil.rmtree(tmp_path)


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as db:
        with db:
            yield db


class Tokenizer(ImageTokenizer):
    def encode(self, text, **options):
        if len(text) == 2 and text[0] == " " and "A" <= text[1] <= "P":
            assert options == {"add_special_tokens": False, "truncation": False}
            return [300000 + ord(text[1])]
        return super().encode(text, **options)


class API:
    def pooling(self, labels=None):
        return NS(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))


class ReadoutEngine(Engine):
    def __init__(self):
        super().__init__()
        self.tokenizer, self.renderer = Tokenizer(), Renderer()
        self.patch = lambda output, params: None
        self.patch_output = self.controls
        self.closed = False

    def controls(self, output, kind):
        assert kind == "encode"
        params = next(call[2] for call in self.calls if call[3] == output.request_id)
        labels = (params.extra_kwargs or {}).get("baseline_token_ids")
        if labels is not None:
            output.outputs.data = torch.cat((output.outputs.data[:2560].float(),
                                            torch.tensor(labels, dtype=torch.float32)))
        self.patch(output, params)

    async def _run(self, *args):
        async for output in super()._run(*args):
            yield output
            if self.mutate_after_yield:
                output.outputs.data.fill_(999)

    def generate(self, *args, **kwargs):
        raise AssertionError("No free-form generation or second model")


@pytest.fixture
def case(fixture, monkeypatch):
    root = fixture.root
    image_root = root / "data/images"
    image_root.mkdir(parents=True)
    source = acquired(image_root)
    # Complete the existing synthetic freeze fixture with canonical raw inputs.
    # This is fixture construction, before any extraction/reference hash snapshot.
    for entry in fixture.reader.require_ready("calibrate")["preparations"]:
        path = Path(entry["path"])
        rows = [json.loads(raw) for raw in path.read_text().splitlines()]
        for row in rows:
            row.update(context=[{"role": "user", "content": "Synthetic context only"}],
                       question="Synthetic question only")
            if row["modality"] == "image":
                row.update(image_path=source.path, image_sha256=source.sha256)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = file_sha256(path)
        name = "image" if path.stem == "images" else path.stem
        manifest_path = root / f"data/manifests/{name}-splits.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"] = digest
        manifest_path.write_text(json.dumps(manifest))
        with database(fixture.directory / "index.sqlite3") as db:
            db.execute("UPDATE preparations SET sha256=? WHERE path=?", (digest, str(path)))
    manifest_path = fixture.head / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preparation"] = require_complete(fixture.reader)
    manifest_path.write_text(json.dumps(manifest))
    reference = deepcopy(fixture.plan["reference"])
    reference["sha256"] = {name: file_sha256(fixture.head / name) for name in reference["sha256"]}
    inputs = root / "inputs"
    prepare_inputs(reference, inputs)
    inputs_sha256 = file_sha256(inputs / MANIFEST)
    rows = list(iter_inputs(inputs, inputs_sha256, reference))

    monkeypatch.setattr(module, "ACQUIRED_MODELS", {
        **module.ACQUIRED_MODELS, "native-4.05": ("synthetic", "fixture")})
    monkeypatch.setattr("fuzev_research.vllm_service.project_path", lambda name: root / name)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model = root / "synthetic-model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text("synthetic bytes; never loaded as a model")
    pins = {
        "schema_version": 1,
        "artifact": {"repo": module.ACQUIRED_MODELS["vllm-nvidia"][0],
                     "revision": module.ACQUIRED_MODELS["vllm-nvidia"][1],
                     "local_snapshot": str(model),
                     "files_sha256": {path.name: file_sha256(path) for path in model.iterdir()}},
        "runtime_cell": {"model": str(model), "tokenizer": str(model),
                         "cache_dtype_config": "auto", "quantization": "modelopt_fp4",
                         "synthetic_fixture_only": True},
        "runtime_sources": {"recipes": "a" * 40, "vllm-fuzev": "b" * 40},
        "runtime_environment": module.runtime_environment(),
    }
    options = {"model": str(model), "tokenizer": str(model), "enforce_eager": True,
               "safetensors_load_strategy": "lazy", "engram_config": {"cpu_offload": False},
               "additional_config": {"fuzev": {"enabled": True, "ple_storage": "mmap"}}}
    engines, guards, admissions, events = [], [], [], []

    @asynccontextmanager
    async def factory(actual_options):
        assert actual_options == options
        engine = engines[-1]
        events.append("engine-start")
        try:
            yield engine
        finally:
            assert engine.active == 0
            engine.closed = True
            events.append("engine-closed")

    def guard_factory(**kwargs):
        guard = Guard(**kwargs)
        guards.append(guard)
        return guard

    async def run(*, engine=None, **kwargs):
        engines.append(engine or ReadoutEngine())
        return await module.extract_vllm(
            kwargs.pop("cell", "vllm-nvidia"), inputs, inputs_sha256, reference,
            kwargs.pop("output", root / "comparison"),
            options, pins, engine_factory=factory,
            inspector=kwargs.pop("inspector", lambda _: {key: deepcopy(pins[key]) for key in (
                "runtime_cell", "runtime_sources")}),
            source_checker=lambda _: events.append("sources-checked"), guard_factory=guard_factory,
            admission=lambda *args: admissions.append(args), api=API(), **kwargs)

    return NS(root=root, reference=reference, inputs=inputs, inputs_sha256=inputs_sha256,
              rows=rows, run=run, pins=pins, options=options, engines=engines,
              guards=guards, admissions=admissions, events=events, reference_reader=fixture.reader,
              output=root / "comparison", source=source)


@pytest.fixture
def view_case(case, monkeypatch):
    from safetensors.torch import save_file

    snapshot = Path(case.pins["artifact"]["local_snapshot"])
    native = {"architectures": ["SyntheticOnly"], "text_config": {"hidden_size": 2560},
              "quantization_config": {"quant_method": "exl3", "bits": 4.05}}
    (snapshot / "config.json").write_text(json.dumps(native))
    (snapshot / "quantization_config.json").write_text(json.dumps(native["quantization_config"]))
    save_file({"linear.weight": torch.zeros(1), "ngram.weight": torch.ones(1)},
              snapshot / "model.safetensors")
    save_file({"vision.weight": torch.ones(1)}, snapshot / "vision.safetensors")
    index = {"metadata": {"total_size": 12}, "weight_map": {
        "linear.weight": "model.safetensors", "ngram.weight": "model.safetensors",
        "vision.weight": "vision.safetensors"}}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    case.pins["artifact"].update(repo="synthetic", revision="fixture", files_sha256={
        path.name: file_sha256(path) for path in snapshot.iterdir()})
    directory = case.root / "exl3-serving-view"
    directory.mkdir()
    for path in snapshot.iterdir():
        if path.name not in {"config.json", "quantization_config.json"}:
            (directory / path.name).symlink_to(path)
    quant = {"quant_method": "exl3", "bits": 4, "derived_from_headers": "pack_scan.json",
             "native_quantization_config": native["quantization_config"]}
    config = {**native, "quantization_config": quant}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "config.json.native").write_bytes((snapshot / "config.json").read_bytes())
    (directory / "quantization_config.json").write_text(json.dumps(quant))
    scan = {"pack": str(snapshot), "tensors": 3, "ngram_problems": [], "moe_layers": 0}
    canonical_scan = case.root / "canonical-pack-scan.json"
    canonical_scan.write_text(json.dumps(scan))
    (directory / "pack_scan.json").write_text(json.dumps({**scan, "pack": str(directory)}))
    payload = {}
    for path in snapshot.glob("*.safetensors"):
        with path.open("rb") as handle:
            prefix = handle.read(8)
            length = int.from_bytes(prefix, "little")
            header_sha256 = hashlib.sha256(prefix + handle.read(length)).hexdigest()
        payload[path.name] = {"source": str(path.resolve()), "bytes": path.stat().st_size,
                              "header_sha256": header_sha256, "same_file": True}
    receipt = {"schema_version": 1, "source": str(snapshot), "tensor_conversion": False,
               "source_config_sha256": file_sha256(snapshot / "config.json"),
               "payload_bytes": sum(entry["bytes"] for entry in payload.values()), "payload": payload,
               "configuration_only": True, "gpu_qualified": False}
    (directory / "serving-view.json").write_text(json.dumps(receipt))
    case.pins["serving_view"] = {"directory": str(directory), "files_sha256": {},
                                "canonical_pack_scan": {"path": str(canonical_scan),
                                                        "sha256": file_sha256(canonical_scan)}}

    def pin_metadata():
        case.pins["serving_view"]["files_sha256"] = {
            path.name: file_sha256(path) for path in directory.iterdir()
            if path.is_file() and not path.name.endswith(".safetensors")}

    pin_metadata()
    case.options.update(model=str(directory), tokenizer=str(directory))
    case.options["additional_config"]["fuzev"].pop("ple_storage")
    monkeypatch.setenv("VLLM_EXL3_NGRAM_TABLE", "disk")
    case.pins["runtime_environment"] = module.runtime_environment()
    case.pins["runtime_cell"].update(model=str(directory), tokenizer=str(directory), quantization="exl3")
    case.pins["runtime_sources"].update({"vllm-exl3-fuzev": "c" * 40, "exllamav3-fuzev": "d" * 40})
    return NS(case=case, directory=directory, snapshot=snapshot, canonical_scan=canonical_scan,
              pin_metadata=pin_metadata)


def client(engine=None):
    engine = engine or ReadoutEngine()
    return module.ComparisonReadout(engine, engine.tokenizer, None, config(),
                                    {"synthetic": True}, Guard(), api=API(), admission=lambda *_: None)


def test_frozen_complete_extraction_and_independent_seal(case):
    async def run():
        engine = ReadoutEngine()
        engine.mutate_after_yield = True
        result = await case.run(engine=engine)
        assert result["status"] == "extracted_pending_independent_seal"
        assert result["counts"] == {"examined": 8, "records": 8, "skipped": 0}
        assert result["committed"] == 8 and not result["runtime_qualified"]
        assert engine.closed and engine.active == 0 and engine.peak >= 1
        assert all(guard.closed for guard in case.guards)
        assert not (case.output / SUBSET_FILE).exists()
        identity = json.loads((case.output / "identity.json").read_text())
        assert identity["corpus_scope"] == TEST_SCOPE
        assert identity["engine"] == "vllm-fuzev"
        assert identity["artifact"] == case.pins["artifact"]
        assert identity["reference_head"] == case.reference
        assert identity["runtime_sources"] != case.reference_reader.identity["runtime_sources"]
        with module.closing(FeatureStore(case.output, identity, read_only=True)) as features:
            assert features.preparations() == []
            for record in features.iter_records(include_metadata=True):
                meta = json.loads(record["metadata"])
                assert meta["has_baseline"] and meta["execution"]["reused_tokens"] is None
                tensors = features.shard_tensors(record["shard"])
                assert (tensors["baseline"][record["row_"]] == torch.tensor([300065., 300066.])).all()
                assert not (tensors["query"][record["row_"]] == 999).any()
                assert not (tensors["candidates"][record["row_"]] == 999).any()
        assert result["storage"] == module.storage_pins(case.output)
        sealed = seal_features(case.output, case.reference, result["counts"]["examined"])
        with module.closing(FeatureStore(case.output, identity, read_only=True)) as features:
            checked = require_comparison_complete(case.reference_reader, features, sealed["subset_manifest_sha256"])
            assert checked["scope"] == TEST_SCOPE
        for name, digest in case.reference["sha256"].items():
            assert file_sha256(Path(case.reference["head"]) / name) == digest
        with pytest.raises(ValueError, match="sealed"):
            await case.run(resume=True)
    asyncio.run(run())


def test_canonical_query_controls_and_concurrent_candidates(case):
    async def run():
        svc = client()
        row = deepcopy(case.rows[0])
        row["candidates"] = [{"id": str(i), "text": "synthetic " + str(i)} for i in range(6)]
        row["target"] = [1., 0., 0., 0., 0., 0.]
        try:
            query, candidates, baseline, execution = await svc.extract(row)
            tokenizer = svc.tokenizer
            encode = lambda text: tuple(tokenizer.encode(text, add_special_tokens=False, truncation=False))
            expected = encode("[context]Synthetic context only") + encode(query_suffix(row["question"], row["candidates"]))
            calls = svc.engine.calls
            assert calls[0][1] == expected
            assert calls[0][2].extra_kwargs == {"baseline_token_ids": list(range(300065, 300071))}
            assert [call[1] for call in calls[1:]] == [expected + encode(candidate_suffix(candidate))
                                                     for candidate in row["candidates"]]
            assert all(call[2].extra_kwargs is None for call in calls[1:])
            assert svc.engine.peak == len(row["candidates"])
            assert query.shape == (2560,) and candidates.shape == (6, 2560) and baseline.shape == (6,)
            assert execution["query_token_sha256"] == token_hash(expected)
            assert execution["forwarded_tokens"] is None
            assert all(not opts["add_generation_prompt"] for _, opts in tokenizer.renders)
            assert not svc.prefixes
        finally:
            await svc.close()
    asyncio.run(run())


def test_image_expansion_once_and_independent_owned_replay(case):
    async def run():
        svc = client()
        svc.engine.mutate_after_yield = True
        row = next(row for row in case.rows if row["modality"] == "image")
        try:
            _, _, _, execution = await svc.extract(row)
            assert len(svc.engine.renderer.calls) == 1
            prompts = svc.engine.inputs
            query = prompts[0]["prompt_token_ids"]
            assert all(prompt["type"] == "multimodal" for prompt in prompts)
            assert query.count(150002) == 4
            assert all(prompt["prompt_token_ids"][:len(query)] == query for prompt in prompts[1:])
            for prompt in prompts:
                assert torch.all(prompt["mm_kwargs"]["image"][0]["pixel_values"].data == 10)
            assert execution["images"][0]["rgb_sha256"] == case.source.sha256
            for image in svc.engine.renderer.pil_images:
                with pytest.raises(ValueError, match="closed"):
                    image.getpixel((0, 0))
        finally:
            await svc.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["checksum", "late-target", "late-duplicate", "late-role", "head"])
def test_entire_input_freeze_is_checked_before_engine(case, mutation):
    path = case.inputs / INPUTS
    if mutation == "head":
        (Path(case.reference["head"]) / "calibration.json").write_text("changed synthetic freeze")
    elif mutation == "checksum":
        path.write_text(path.read_text() + "\n")
    else:
        rows = deepcopy(case.rows)
        if mutation == "late-target":
            rows[-1]["target"] = [0., 1.]
        elif mutation == "late-duplicate":
            rows[-1] = rows[0]
        else:
            rows[-1]["role"] = "fit"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest_path = case.inputs / MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["inputs"].update(sha256=file_sha256(path), bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest))
        # Keep the outer pin valid to exercise the late semantic/freeze checks.
        original = module.iter_inputs
        monkeypin = file_sha256(manifest_path)
        from unittest.mock import patch
        with patch.object(module, "iter_inputs", lambda directory, _, spec: original(directory, monkeypin, spec)):
            with pytest.raises(ValueError):
                asyncio.run(case.run())
        assert "engine-start" not in case.events and not case.output.exists()
        return
    with pytest.raises(ValueError):
        asyncio.run(case.run())
    assert "engine-start" not in case.events and not case.output.exists()


@pytest.mark.parametrize("mutation", ["revision", "bytes", "unlisted", "remote", "environment", "quantization"])
def test_artifact_and_environment_pins_fail_before_engine(case, mutation):
    artifact = case.pins["artifact"]
    if mutation == "revision":
        artifact["revision"] = "0" * 40
    elif mutation == "bytes":
        (Path(artifact["local_snapshot"]) / "model.safetensors").write_text("changed")
    elif mutation == "unlisted":
        (Path(artifact["local_snapshot"]) / "overlay.json").write_text("unlisted")
    elif mutation == "remote":
        case.options["model"] = "remote/model"
    elif mutation == "environment":
        case.pins["runtime_environment"]["EXL3_SYNTHETIC_CHANGED"] = "1"
    else:
        case.pins["runtime_cell"]["quantization"] = None
    with pytest.raises(ValueError):
        asyncio.run(case.run())
    assert "engine-start" not in case.events and not case.output.exists()


@pytest.mark.parametrize("setting", ["missing-storage", "resident", "missing-offload", "pinned-host",
                                     "ambiguous-offload", "missing-lazy", "eager-loader", "missing-eager", "graph"])
def test_raw_storage_rejects_before_reference_checksums_or_output(case, monkeypatch, setting):
    opts = case.options
    if setting == "missing-storage":
        opts["additional_config"]["fuzev"].pop("ple_storage")
    elif setting == "resident":
        opts["additional_config"]["fuzev"]["ple_storage"] = "resident"
    elif setting == "missing-offload":
        opts.pop("engram_config")
    elif setting in {"pinned-host", "ambiguous-offload"}:
        opts["engram_config"]["cpu_offload"] = True if setting == "pinned-host" else "false"
    elif setting == "missing-lazy":
        opts.pop("safetensors_load_strategy")
    elif setting == "eager-loader":
        opts["safetensors_load_strategy"] = "eager"
    elif setting == "missing-eager":
        opts.pop("enforce_eager")
    else:
        opts["enforce_eager"] = False

    def forbidden(*_):
        raise AssertionError("Storage rejection must precede reference/artifact checksums")

    monkeypatch.setattr(module, "frozen_reference", forbidden)
    monkeypatch.setattr(module, "artifact_files", forbidden)
    monkeypatch.setattr(module, "checked_file", forbidden)
    with pytest.raises(ValueError, match="PLE"):
        asyncio.run(case.run())
    with pytest.raises(ValueError, match="PLE"):
        module.verify_artifact("vllm-nvidia", case.pins, opts, {"model": "synthetic", "revision": "fixture"})
    assert not case.events and not case.guards and not case.admissions and not case.output.exists()


@pytest.mark.parametrize("setting", ["missing-env", "resident-env", "raw-mmap", "mmap-and-resident",
                                     "missing-offload", "eager-loader"])
def test_packed_storage_rejects_before_reference_checksums_or_output(view_case, monkeypatch, setting):
    case = view_case.case
    if setting == "missing-env":
        monkeypatch.delenv("VLLM_EXL3_NGRAM_TABLE")
    elif setting in {"resident-env", "mmap-and-resident"}:
        monkeypatch.setenv("VLLM_EXL3_NGRAM_TABLE", "resident")
    if setting in {"raw-mmap", "mmap-and-resident"}:
        case.options["additional_config"]["fuzev"]["ple_storage"] = "mmap"
    elif setting == "missing-offload":
        case.options.pop("engram_config")
    elif setting == "eager-loader":
        case.options["safetensors_load_strategy"] = "eager"
    case.pins["runtime_environment"] = module.runtime_environment()

    def forbidden(*_):
        raise AssertionError("Storage rejection must precede reference/artifact checksums")

    monkeypatch.setattr(module, "frozen_reference", forbidden)
    monkeypatch.setattr(module, "artifact_files", forbidden)
    monkeypatch.setattr(module, "checked_file", forbidden)
    with pytest.raises(ValueError, match="PLE"):
        asyncio.run(case.run(cell="vllm-exl3-4.05"))
    with pytest.raises(ValueError, match="PLE"):
        module.verify_artifact("vllm-exl3-4.05", case.pins, case.options,
                               {"model": "synthetic", "revision": "fixture"})
    assert not case.events and not case.guards and not case.admissions and not case.output.exists()


def nvidia_mixed_config():
    return {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "FP8"},
        "model.language_model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"}}}


def pin_model_config(case, config):
    path = Path(case.pins["artifact"]["local_snapshot"]) / "config.json"
    path.write_text(json.dumps(config))
    case.pins["artifact"]["files_sha256"][path.name] = file_sha256(path)


@pytest.mark.parametrize("location,nested", [("root", False), ("text", False), ("root", True)])
def test_canonical_nvidia_mixed_factory_name_is_preserved(case, location, nested):
    quant = nvidia_mixed_config()
    if nested:
        quant = {"quant_method": quant.pop("quant_method"), "quantization": quant}
    config = {"quantization_config": quant}
    if location == "text":
        config = {"text_config": config}
    pin_model_config(case, config)
    case.pins["runtime_cell"]["quantization"] = "modelopt_mixed"
    result = asyncio.run(case.run())
    identity = json.loads((case.output / "identity.json").read_text())
    assert result["counts"]["records"] == 8
    assert identity["readout_runtime"]["quantization"] == "modelopt_mixed"
    assert identity["artifact"] == case.pins["artifact"] and identity["reference_head"] == case.reference


@pytest.mark.parametrize("cell", ["vllm-radixark", "vllm-exl3-4.05"])
def test_mixed_factory_is_not_admitted_for_other_cells(case, monkeypatch, cell):
    key = "native-4.05" if cell == "vllm-exl3-4.05" else cell
    case.pins["artifact"].update(zip(("repo", "revision"), module.ACQUIRED_MODELS[key]))
    case.pins["runtime_cell"]["quantization"] = "modelopt_mixed"
    pin_model_config(case, {"quantization_config": nvidia_mixed_config()})

    def forbidden(*_):
        raise AssertionError("Wrong-cell rejection must precede payload checksums")

    monkeypatch.setattr(module, "checked_file", forbidden)
    with pytest.raises(ValueError, match="quantization|NVIDIA"):
        asyncio.run(case.run(cell=cell))
    assert "engine-start" not in case.events and not case.output.exists()


@pytest.mark.parametrize("mutation", ["missing", "method", "algo", "ple", "packed-ple", "layers", "text", "root"])
def test_nvidia_mixed_requires_canonical_config_before_tensor_checksums(case, monkeypatch, mutation):
    quant = nvidia_mixed_config()
    ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    config = {"quantization_config": quant}
    if mutation == "missing":
        config = {}
    elif mutation == "method":
        quant["quant_method"] = "modelopt_fp4"
    elif mutation == "algo":
        quant["quant_algo"] = "NVFP4"
    elif mutation == "ple":
        quant["quantized_layers"].pop(ple)
    elif mutation == "packed-ple":
        quant["quantized_layers"][ple]["quant_algo"] = "EXL3"
    elif mutation == "layers":
        quant["quantized_layers"] = None
    elif mutation == "text":
        config["text_config"] = []
    else:
        config = []
    pin_model_config(case, config)
    case.pins["runtime_cell"]["quantization"] = "modelopt_mixed"
    checked = module.checked_file
    names = []

    def small_only(path, digest):
        names.append(path.name)
        assert path.name == "config.json", "Reject incompatible config before tensor payload hashes"
        return checked(path, digest)

    monkeypatch.setattr(module, "checked_file", small_only)
    with pytest.raises(ValueError, match="MIXED_PRECISION config and FP8 PLE"):
        asyncio.run(case.run())
    assert names == ["config.json"]
    assert "engine-start" not in case.events and not case.output.exists()


def test_nvidia_mixed_keeps_small_config_stamp_through_payload_verification(case, monkeypatch):
    pin_model_config(case, {"quantization_config": nvidia_mixed_config()})
    case.pins["runtime_cell"]["quantization"] = "modelopt_mixed"
    config_path = Path(case.pins["artifact"]["local_snapshot"]) / "config.json"
    original = module.read_json

    def retouched(path):
        value = original(path)
        if Path(path) == config_path:
            config_path.write_bytes(config_path.read_bytes())
        return value

    monkeypatch.setattr(module, "read_json", retouched)
    with pytest.raises(ValueError, match="changed during verification"):
        asyncio.run(case.run())
    assert "engine-start" not in case.events and not case.output.exists()


@pytest.mark.parametrize("canonical_tokenizer,derived_index", [(False, False), (True, True)])
def test_exl3_canonical_payload_and_explicit_loader_view(view_case, canonical_tokenizer, derived_index):
    view, case = view_case, view_case.case
    if canonical_tokenizer:
        case.options["tokenizer"] = case.pins["runtime_cell"]["tokenizer"] = str(view.snapshot)
    if derived_index:
        path = view.directory / "model.safetensors.index.json"
        index = json.loads(path.read_text())
        # Exercise the reviewed regenerated-index path without changing tensors.
        path.unlink()
        index["metadata"]["synthetic_reviewed_index"] = True
        path.write_text(json.dumps(index))
        (view.directory / "model.safetensors.index.json.native").symlink_to(view.snapshot / path.name)
        view.pin_metadata()
    result = asyncio.run(case.run(cell="vllm-exl3-4.05"))
    assert result["counts"]["records"] == 8 and not result["runtime_qualified"]
    identity = json.loads((case.output / "identity.json").read_text())
    assert identity["artifact"]["local_snapshot"] == str(view.snapshot)
    assert identity["readout_runtime"]["model"] == str(view.directory)
    assert identity["serving_view"] == case.pins["serving_view"]
    assert identity["reference_head"] == case.reference
    assert seal_features(case.output, case.reference, 8)["manifest"]["extraction"]["completed"]


@pytest.mark.parametrize("mutation", [
    "tensor-swap", "tensor-copy", "extra-tensor", "missing-tensor", "canonical-bytes", "canonical-hash",
    "unreviewed-config", "nonquant-config", "quant-origin", "quant-sidecar", "native-config",
    "tokenizer", "unlisted-metadata", "receipt-source", "receipt-tensor-source", "receipt-conversion",
    "scan-source", "scan-data", "scan-hash", "unreviewed-index", "index-outside", "index-incomplete",
])
def test_exl3_view_rejects_changed_payload_or_unproven_metadata_before_engine(view_case, mutation):
    view, case = view_case, view_case.case

    def change(name, edit):
        path = view.directory / name
        value = json.loads(path.read_text())
        edit(value)
        if path.is_symlink():
            path.unlink()
        path.write_text(json.dumps(value))

    if mutation in {"tensor-swap", "tensor-copy", "missing-tensor"}:
        path = view.directory / "model.safetensors"
        path.unlink()
        if mutation == "tensor-swap":
            path.symlink_to(view.snapshot / "vision.safetensors")
        elif mutation == "tensor-copy":
            shutil.copyfile(view.snapshot / path.name, path)
    elif mutation == "extra-tensor":
        (view.directory / "extra.safetensors").symlink_to(view.snapshot / "model.safetensors")
    elif mutation == "canonical-bytes":
        (view.snapshot / "model.safetensors").write_bytes(b"different synthetic canonical bytes")
    elif mutation == "canonical-hash":
        del case.pins["artifact"]["files_sha256"]["model.safetensors"]
    elif mutation == "unreviewed-config":
        change("config.json", lambda doc: doc["quantization_config"].update(bits=5))
    elif mutation == "nonquant-config":
        change("config.json", lambda doc: doc["text_config"].update(hidden_size=1024))
    elif mutation == "quant-origin":
        for name in ("config.json", "quantization_config.json"):
            change(name, lambda doc: doc.get("quantization_config", doc)["native_quantization_config"].update(bits=5))
    elif mutation == "quant-sidecar":
        change("quantization_config.json", lambda doc: doc.update(bits=5))
    elif mutation == "native-config":
        change("config.json.native", lambda doc: doc.update(changed=True))
    elif mutation == "tokenizer":
        path = view.directory / "tokenizer.json"
        path.unlink()
        path.write_text("different tokenizer")
    elif mutation == "unlisted-metadata":
        (view.directory / "unreviewed.json").write_text("{}")
    elif mutation == "receipt-source":
        change("serving-view.json", lambda doc: doc.update(source=str(view.directory)))
    elif mutation == "receipt-tensor-source":
        change("serving-view.json", lambda doc: doc["payload"]["model.safetensors"].update(
            source=str(view.snapshot / "vision.safetensors")))
    elif mutation == "receipt-conversion":
        change("serving-view.json", lambda doc: doc.update(tensor_conversion=True))
    elif mutation == "scan-source":
        scan = json.loads(view.canonical_scan.read_text())
        scan["pack"] = str(view.directory)
        view.canonical_scan.write_text(json.dumps(scan))
        case.pins["serving_view"]["canonical_pack_scan"]["sha256"] = file_sha256(view.canonical_scan)
    elif mutation == "scan-data":
        change("pack_scan.json", lambda doc: doc.update(tensors=4))
    elif mutation == "scan-hash":
        view.canonical_scan.write_text(view.canonical_scan.read_text() + " ")
    elif mutation in {"unreviewed-index", "index-outside"}:
        change("model.safetensors.index.json", lambda doc: doc["weight_map"].update(
            {"linear.weight": "../elsewhere.safetensors"}))
    else:
        change("model.safetensors.index.json", lambda doc: doc["weight_map"].pop("linear.weight"))
    if mutation not in {"unreviewed-config", "unreviewed-index"}:
        view.pin_metadata()
    with pytest.raises(ValueError):
        asyncio.run(case.run(cell="vllm-exl3-4.05"))
    assert "engine-start" not in case.events and not case.output.exists()


@pytest.mark.parametrize("cell", ["vllm-nvidia", "vllm-radixark"])
def test_serving_view_exception_never_applies_to_nvfp4(view_case, cell):
    case = view_case.case
    case.pins["artifact"].update(zip(("repo", "revision"), module.ACQUIRED_MODELS[cell]))
    case.pins["runtime_cell"]["quantization"] = "modelopt_fp4"
    case.options["additional_config"]["fuzev"]["ple_storage"] = "mmap"
    with pytest.raises(ValueError, match="supported vLLM comparison pins"):
        asyncio.run(case.run(cell=cell))
    assert "engine-start" not in case.events and not case.output.exists()


def test_exl3_view_requires_explicit_proof_not_relabeling_as_snapshot(view_case):
    case = view_case.case
    del case.pins["serving_view"]
    with pytest.raises(ValueError, match="explicit EXL3 serving view"):
        asyncio.run(case.run(cell="vllm-exl3-4.05"))
    assert "engine-start" not in case.events


def test_rereviewed_view_pins_cannot_adopt_existing_features_on_resume(view_case):
    view, case = view_case, view_case.case
    asyncio.run(case.run(cell="vllm-exl3-4.05"))
    path = view.directory / "config.json"
    path.write_text(path.read_text() + " ")
    view.pin_metadata()
    with pytest.raises(ValueError, match="Resume requires exact pins"):
        asyncio.run(case.run(cell="vllm-exl3-4.05", resume=True))
    assert case.events.count("engine-start") == 1


@pytest.mark.parametrize("target", ["config", "scan", "alias"])
def test_view_and_source_proof_cannot_change_during_extraction_or_resume(view_case, target):
    view, case = view_case, view_case.case
    engine = ReadoutEngine()

    def changed(*_):
        if len(engine.calls) == 24:
            if target == "alias":
                path = view.directory / "model.safetensors"
                path.unlink()
                path.symlink_to(view.snapshot / "vision.safetensors")
            else:
                path = view.directory / "config.json" if target == "config" else view.canonical_scan
                path.write_text(path.read_text() + " ")

    engine.patch = changed
    with pytest.raises(ValueError, match="changed during extraction"):
        asyncio.run(case.run(cell="vllm-exl3-4.05", engine=engine))
    assert engine.closed
    receipt = json.loads((case.output / module.RECEIPT).read_text())
    assert not receipt["resume_allowed"] and receipt["status"] == "failed"
    with pytest.raises(ValueError, match="failures"):
        seal_features(case.output, case.reference, 8)
    with pytest.raises(ValueError):
        asyncio.run(case.run(cell="vllm-exl3-4.05", resume=True))
    assert case.events.count("engine-start") == 1


@pytest.mark.parametrize("target", ["view", "scan"])
def test_view_and_canonical_scan_are_protected_outputs(view_case, target):
    view, case = view_case, view_case.case
    output = view.directory / "nested" if target == "view" else view.canonical_scan
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(case.run(cell="vllm-exl3-4.05", output=output))
    assert "engine-start" not in case.events


@pytest.mark.parametrize("key", ["VLLM_API_KEY", "EXL3_ACCESS_TOKEN", "FUZEV_CLIENT_SECRET", "HF_TOKEN"])
def test_environment_credentials_reject_without_persisting_values(case, monkeypatch, key):
    secret = "synthetic-secret-must-not-appear"
    monkeypatch.setenv(key, secret)
    with pytest.raises(ValueError, match="Credential-named") as error:
        module.runtime_environment()
    assert secret not in str(error.value)
    with pytest.raises(ValueError, match="Credential-named") as error:
        asyncio.run(case.run())
    assert secret not in str(error.value)
    assert not case.output.exists() and "engine-start" not in case.events


@pytest.mark.parametrize("location", ["options", "nested-options", "pinned-environment", "pinned-runtime"])
@pytest.mark.parametrize("value", ["synthetic-secret-must-not-appear", None])
def test_credential_option_and_pin_keys_are_rejected_even_when_empty(case, location, value):
    if location == "options":
        case.options["hf_token"] = value
    elif location == "nested-options":
        case.options["additional_config"] = {"nested": [{"Authorization": value}]}
    elif location == "pinned-environment":
        case.pins["runtime_environment"]["VLLM_API_KEY"] = value
    else:
        case.pins["runtime_cell"]["accessToken"] = value
    with pytest.raises(ValueError, match="Credential-named") as error:
        asyncio.run(case.run())
    assert "synthetic-secret" not in str(error.value)
    assert not case.output.exists() and "engine-start" not in case.events


def test_noncredential_token_settings_remain_pinnable(monkeypatch):
    module.reject_credentials({"tokenizer": "/local", "max_num_batched_tokens": 128,
                               "additional_config": {"token_ids": [1, 2], "max_tokens": 1}})
    monkeypatch.setenv("VLLM_MAX_TOKENS", "128")
    assert module.runtime_environment()["VLLM_MAX_TOKENS"] == "128"


def test_late_environment_credential_blocks_completion_without_recording_value(case, monkeypatch):
    engine = ReadoutEngine()
    secret = "synthetic-secret-must-not-appear"

    def changed(*_):
        if len(engine.calls) == 24:
            monkeypatch.setenv("VLLM_API_KEY", secret)

    engine.patch = changed
    with pytest.raises(ValueError, match="Credential-named") as error:
        asyncio.run(case.run(engine=engine))
    assert secret not in str(error.value) and engine.closed
    assert secret not in (case.output / module.RECEIPT).read_text()
    assert secret not in (case.output / "identity.json").read_text()
    with database(case.output / "index.sqlite3") as db:
        assert all(secret not in row[0] for row in db.execute("SELECT reason FROM failures"))
    receipt = json.loads((case.output / module.RECEIPT).read_text())
    assert receipt["status"] == "failed" and not receipt["resume_allowed"]


def test_progress_is_bounded_128_processed_rows_and_uses_monotonic_rate(tmp_path, monkeypatch):
    clock = NS(monotonic=lambda: 10., time=lambda: 1000.)
    monkeypatch.setattr(module, "time", clock)
    writes = []
    monkeypatch.setattr(module, "write_json", lambda path, doc: writes.append(deepcopy(doc)))
    progress, counts = {"status": "extracting"}, {"records": 0, "skipped": 0, "examined": 0}
    started = clock.monotonic()
    module.write_progress(tmp_path / module.RECEIPT, progress, counts, started, force=True)
    for number in range(1, 260):
        counts["examined"] += 1
        counts["skipped" if number <= 128 else "records"] += 1
        clock.monotonic = lambda: 10. + number / 2
        clock.time = lambda: 1000. - number  # Wall-clock rollback cannot inflate the rate.
        module.write_progress(tmp_path / module.RECEIPT, progress, counts, started)
    assert [doc["counts"]["examined"] for doc in writes] == [0, 128, 256]
    assert writes[0]["records_per_second"] == writes[1]["records_per_second"] == 0.
    assert writes[2]["elapsed_seconds"] == 128. and writes[2]["records_per_second"] == 1.
    module.write_progress(tmp_path / module.RECEIPT, progress, counts, started, force=True)
    assert writes[-1]["counts"] == {"examined": 259, "records": 131, "skipped": 128}
    assert writes[-1]["records_per_second"] == pytest.approx(131 / 129.5)


def test_progress_wiring_writes_rows_skips_and_drained_end_without_live_state(case, monkeypatch):
    writes, original = [], module.write_json

    def capture(path, doc):
        writes.append(deepcopy(doc))
        original(path, doc)

    monkeypatch.setattr(module, "write_json", capture)
    monkeypatch.setattr(module, "PROGRESS_INTERVAL", 2)  # Tiny integration fixture, production is 128.
    asyncio.run(case.run())
    assert [doc["counts"]["examined"] for doc in writes if doc["status"] == "extracting"] == [0, 2, 4, 6, 8]
    assert writes[-1]["status"] == "extracted_pending_independent_seal" and writes[-1]["storage"]
    assert all(not doc["resume_allowed"] for doc in writes[:-1])
    assert all(doc["elapsed_seconds"] >= 0 and doc["records_per_second"] >= 0 for doc in writes)
    writes.clear()
    asyncio.run(case.run(resume=True))
    assert [doc["counts"]["skipped"] for doc in writes if doc["status"] == "extracting"] == [0, 2, 4, 6, 8]
    assert all(doc["records_per_second"] == 0 for doc in writes)
    assert not case.engines[-1].calls and not (case.output / SUBSET_FILE).exists()


def test_failure_is_drained_and_exact_resume_clears_only_retried_failures(case):
    async def run():
        engine = ReadoutEngine()
        engine.fail_call = 6
        with pytest.raises(RuntimeError, match="Synthetic engine failure"):
            await case.run(engine=engine)
        receipt = json.loads((case.output / module.RECEIPT).read_text())
        assert receipt["status"] == "failed" and receipt["storage"] == module.storage_pins(case.output)
        assert receipt["counts"] == {"examined": 2, "records": 1, "skipped": 0}
        assert receipt["elapsed_seconds"] > 0
        assert receipt["records_per_second"] == pytest.approx(1 / receipt["elapsed_seconds"])
        assert engine.closed and engine.active == 0 and engine.aborted
        with database(case.output / "index.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM failures").fetchone()[0] == 2
        assert not (case.output / SUBSET_FILE).exists()
        with pytest.raises(ValueError, match="explicit resume"):
            await case.run()
        resumed = await case.run(resume=True)
        assert resumed["counts"] == {"examined": 8, "records": 7, "skipped": 1}
        with database(case.output / "index.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM failures").fetchone()[0] == 0
        assert len(case.engines[-1].calls) == 21
        assert seal_features(case.output, case.reference, 8)["manifest"]["extraction"]["completed"]
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["binding", "identity", "shard", "index"])
def test_changed_resume_storage_or_binding_never_reaches_engine(case, mutation):
    asyncio.run(case.run())
    starts = case.events.count("engine-start")
    if mutation == "binding":
        case.options["seed"] = 123
    elif mutation == "index":
        with database(case.output / "index.sqlite3") as db:
            db.execute("INSERT INTO failures VALUES('extra', 'synthetic')")
    else:
        path = case.output / ("identity.json" if mutation == "identity" else "shard-000000.safetensors")
        with path.open("ab") as handle:
            handle.write(b" ")
    with pytest.raises(ValueError, match="Resume"):
        asyncio.run(case.run(resume=True))
    assert case.events.count("engine-start") == starts


def test_runtime_mismatch_retains_resumable_empty_corpus_not_false_features(case):
    async def run():
        with pytest.raises(ValueError, match="Actual AsyncLLM"):
            await case.run(inspector=lambda _: {"wrong": True})
        assert case.engines[-1].closed and not case.engines[-1].calls
        receipt = json.loads((case.output / module.RECEIPT).read_text())
        assert receipt["status"] == "failed" and receipt["storage"]
        resumed = await case.run(resume=True)
        assert resumed["counts"]["records"] == 8
    asyncio.run(run())


@pytest.mark.parametrize("error", [MemoryError("synthetic headroom"), torch.OutOfMemoryError("synthetic OOM")])
def test_memory_failure_preserves_guards_and_reports_without_fallback_tuning(case, error):
    async def run():
        engine = ReadoutEngine()

        def fail(*_):
            raise error

        engine.patch = fail
        with pytest.raises(type(error)):
            await case.run(engine=engine)
        receipt = json.loads((case.output / module.RECEIPT).read_text())
        assert receipt["status"] == "memory_infeasible" and receipt["storage"]
        assert engine.closed and engine.aborted and engine.active == 0
        assert not (case.output / SUBSET_FILE).exists()
        assert all(guard.closed for guard in case.guards)
    asyncio.run(run())


def test_repeated_cancellation_waits_for_abort_and_engine_close(case):
    async def run():
        engine = ReadoutEngine()
        engine.wait, engine.abort_wait = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(case.run(engine=engine))
        await engine.entered.wait()
        task.cancel()
        await engine.aborting.wait()
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done() and not engine.closed and engine.active == 1
        engine.abort_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.closed and engine.active == 0
        receipt = json.loads((case.output / module.RECEIPT).read_text())
        assert receipt["status"] == "cancelled" and receipt["storage"]
        assert not (case.output / SUBSET_FILE).exists()
        assert all(guard.closed for guard in case.guards)
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["width", "nan", "prompt", "request", "missing-controls", "branch-controls"])
def test_invalid_engine_payload_never_commits_a_row(case, mutation):
    async def run():
        engine = ReadoutEngine()

        def changed(output, params):
            labels = (params.extra_kwargs or {}).get("baseline_token_ids")
            if mutation == "width":
                output.outputs.data = torch.zeros(2561)
            elif mutation == "nan":
                output.outputs.data[0] = float("nan")
            elif mutation == "prompt":
                output.prompt_token_ids = [99]
            elif mutation == "request":
                output.request_id = "wrong"
            elif mutation == "missing-controls" and labels:
                output.outputs.data = output.outputs.data[:2560]
            elif mutation == "branch-controls" and not labels:
                output.outputs.data = torch.zeros(2562)

        engine.patch = changed
        with pytest.raises(ValueError):
            await case.run(engine=engine)
        with database(case.output / "index.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 0
        assert engine.closed and engine.active == 0
        assert not (case.output / SUBSET_FILE).exists()
    asyncio.run(run())


def test_postflight_freeze_failure_is_not_reported_as_completed(case):
    engine = ReadoutEngine()

    def changed(*_):
        if len(engine.calls) == 24:
            with (Path(case.reference["head"]) / "calibration.json").open("a") as handle:
                handle.write(" ")

    engine.patch = changed
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(case.run(engine=engine))
    assert engine.closed
    receipt = json.loads((case.output / module.RECEIPT).read_text())
    assert receipt["status"] == "failed" and not (case.output / SUBSET_FILE).exists()


def test_complete_but_postflight_failed_corpus_cannot_be_sealed(case):
    inspections = 0

    def inspector(engine):
        nonlocal inspections
        inspections += 1
        actual = {key: deepcopy(case.pins[key]) for key in ("runtime_cell", "runtime_sources")}
        if inspections == 2:
            actual["runtime_cell"]["synthetic_change"] = True
        return actual

    with pytest.raises(ValueError, match="Runtime identity changed"):
        asyncio.run(case.run(inspector=inspector))
    with database(case.output / "index.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 8
        assert db.execute("SELECT reason FROM failures WHERE id=?", (module.RUN_FAILURE,)).fetchone()
    with pytest.raises(ValueError, match="failures"):
        seal_features(case.output, case.reference, 8)
    assert not (case.output / SUBSET_FILE).exists()
    with pytest.raises(ValueError, match="Resume"):
        asyncio.run(case.run(resume=True))


@pytest.mark.parametrize("cell,quantization", [
    ("vllm-radixark", "compressed-tensors"), ("vllm-exl3-4.05", "exl3")])
def test_other_actual_cell_identity_is_separate_from_reference_head(case, monkeypatch, cell, quantization):
    key = "native-4.05" if cell == "vllm-exl3-4.05" else cell
    case.pins["artifact"].update(zip(("repo", "revision"), module.ACQUIRED_MODELS[key]))
    case.pins["runtime_cell"]["quantization"] = quantization
    if quantization == "exl3":
        case.pins["runtime_sources"].update({"vllm-exl3-fuzev": "c" * 40, "exllamav3-fuzev": "d" * 40})
        case.options["additional_config"]["fuzev"].pop("ple_storage")
        monkeypatch.setenv("VLLM_EXL3_NGRAM_TABLE", "disk")
        case.pins["runtime_environment"] = module.runtime_environment()
    result = asyncio.run(case.run(cell=cell))
    assert result["counts"]["records"] == 8
    identity = json.loads((case.output / "identity.json").read_text())
    assert identity["cell"] == cell and identity["reference_head"] == case.reference
    assert identity["model"] == module.ACQUIRED_MODELS[key][0]
    assert identity["readout_runtime"]["quantization"] == quantization
    assert seal_features(case.output, case.reference, 8)["manifest"]["extraction"]["completed"]


def test_abort_failure_closes_admission_and_still_drains(case):
    async def run():
        engine = ReadoutEngine()
        engine.fail_call = 2

        async def abort(request_id):
            engine.aborted.append(request_id)
            raise RuntimeError("synthetic abort failure")

        engine.abort = abort
        svc = client(engine)
        try:
            with pytest.raises(RuntimeError, match="abort failure"):
                await svc.extract(case.rows[0])
            assert svc.cleanup_error and engine.active == 0 and not svc.engine_requests
            with pytest.raises(RuntimeError, match="admission is closed"):
                await svc.extract(case.rows[0])
        finally:
            await svc.close()
    asyncio.run(run())


def test_live_output_lock_prevents_a_second_engine(case):
    async def run():
        engine = ReadoutEngine()
        engine.wait = asyncio.Event()
        task = asyncio.create_task(case.run(engine=engine))
        await engine.entered.wait()
        try:
            with pytest.raises(BlockingIOError):
                await case.run(resume=True)
            assert case.events.count("engine-start") == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(run())


def test_request_deadline_aborts_and_drains(case):
    async def run():
        engine = ReadoutEngine()
        engine.wait = asyncio.Event()
        with pytest.raises(TimeoutError):
            await case.run(engine=engine, timeout=1)
        assert engine.closed and engine.active == 0 and engine.aborted
        receipt = json.loads((case.output / module.RECEIPT).read_text())
        assert receipt["status"] == "failed"
    asyncio.run(run())


def test_context_overflow_and_multitoken_labels_never_submit(case):
    async def run():
        for mode in ("overflow", "labels"):
            svc = client()
            row = deepcopy(case.rows[0])
            if mode == "overflow":
                row["context"][0]["content"] = "x" * module.CONTEXT
            else:
                svc.tokenizer = ImageTokenizer()
            try:
                with pytest.raises(ValueError, match="context|one-token|window"):
                    await svc.extract(row)
                assert not svc.engine.calls
            finally:
                await svc.close()
    asyncio.run(run())


@pytest.mark.parametrize("target", ["reference", "head", "inputs", "model"])
def test_protected_output_paths(case, target):
    directory = {"reference": case.reference["features"], "head": case.reference["head"],
                 "inputs": case.inputs, "model": case.pins["artifact"]["local_snapshot"]}[target]
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(case.run(output=Path(directory) / "nested"))
    assert "engine-start" not in case.events


def test_source_pins_require_both_real_exl3_plugin_modules():
    pins = {"runtime_cell": {"quantization": "exl3"},
            "runtime_sources": {"recipes": "a" * 40, "vllm-fuzev": "b" * 40}}
    with pytest.raises(ValueError, match="both plugin"):
        module.verify_sources(pins)


def test_cli_requires_explicit_inference_before_reading_paths():
    with pytest.raises(SystemExit) as error:
        module.main(["--model", "vllm-nvidia", "--plan", "/nonexistent",
                     "--inputs", "/nonexistent", "--inputs-sha256", "a" * 64,
                     "--engine-config", "/nonexistent", "--pins", "/nonexistent", "--output", "/nonexistent"])
    assert error.value.code == 2
