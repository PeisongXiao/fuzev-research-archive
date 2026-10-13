"""Static contract/lifecycle fakes; no real engine, CUDA, HTTP or weights."""

import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch


class BaseService:
    def __init__(self, engine, tokenizer, head, config, provenance, guard, **kwargs):
        self.engine, self.head, self.config = engine, head, config
        self.provenance = deepcopy(provenance)
        self.identity = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()
        self.images, self.receipt_status = NS(), "awaiting-worker-receipt"
        self.close = AsyncMock()

    def _check(self):
        pass

    def readout_health(self):
        return {"status": self.receipt_status, "required_mode": "query-lm-assisted"}

    async def decide(self, request, request_id):
        return {"id": request_id, "execution": {"requests": ["synthetic-worker-receipt"]}}


def load_server():
    service = ModuleType("fuzev_research.vllm_service")
    service.VllmService = BaseService
    service.VllmServiceConfig = NS(model_validate=lambda value: NS(**value))
    service.build_service = AsyncMock()
    service.describe_runtime, service.validate_engine_config = Mock(), Mock()
    server = ModuleType("fuzev_research.vllm_server")
    server.create_app = Mock()
    path = Path(__file__).parents[1] / "fuzev_research/kev_vllm_research_server.py"
    spec = importlib.util.spec_from_file_location("fuzev_research._synthetic_kev_vllm_research_server", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {service.__name__: service, server.__name__: server}):
        spec.loader.exec_module(module)
    return module


class ResearchServerTests(unittest.TestCase):
    def setUp(self):
        self.server = load_server()

    def fixture(self, cell="vllm-nvidia"):
        s = self.server
        key, repo, revision, quantization, image = s.CELLS[cell]
        config = NS(model_name=cell, head_id="reference", manifest_sha256=s.PLAN_SHA,
                    calibration_sha256=s.PLAN_SHA, feature_identity={"synthetic-reference": True},
                    runtime_cell={"quantization": quantization}, runtime_sources={"vllm-fuzev": s.CORE_COMMIT})
        if quantization == "exl3":
            config.runtime_sources.update({"vllm-exl3-fuzev": "47f4d0ad73624d08b2da7fba9f5be5b28231ee3a",  # unpublished lab commit ID
                                           "exllamav3-fuzev": "ba60611d031f673a12ebbf2f133c583b4e12df7f"})  # unpublished lab commit ID
        options = {"runner": "generate", "dtype": "bfloat16", "kv_cache_dtype": "auto",
                   "gpu_memory_utilization": s.MEMORY_FRACTION, "max_model_len": 262144, "max_num_seqs": 4,
                   "max_num_batched_tokens": 16384, "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                   "data_parallel_size": 1, "async_scheduling": False, "enable_prefix_caching": True,
                   "enable_chunked_prefill": True, "mamba_cache_mode": "align", "enforce_eager": True,
                   "trust_remote_code": False, "limit_mm_per_prompt": {"image": 1, "video": 0},
                   "mm_processor_kwargs": {"truncation": False}, "safetensors_load_strategy": "lazy",
                   "load_format": "safetensors", "engram_config": {"cpu_offload": False},
                   "additional_config": {"fuzev": {"enabled": True, "ple_storage": "resident" if quantization == "exl3" else "mmap"}}}
        paths = {s.PLAN, s.QUALITY, s.AUDIT, s.ACQUISITION, Path("/synthetic/reference.json"),
                 Path("/synthetic/model/config.json"), *(Path(s.__file__).with_name(name) for name in s.RECIPE_FILES)}
        pins = {str(path): "0" * 64 for path in paths}
        pins.update({str(s.PLAN): s.PLAN_SHA, str(s.QUALITY): s.QUALITY_SHA})
        research = {"image_id": image, "source_pins": pins,
                    "binary_pins": dict(s.EXL3_BINARIES) if quantization == "exl3" else dict.fromkeys(s.EXL3_BINARIES, "0" * 64)}
        environment = {"FUZEV_ROOT": str(s.ROOT), "FUZEV_RESEARCH_IMAGE_ID": image or "synthetic-image",
                       "VLLM_PLUGINS": "vllm_exl3" if quantization == "exl3" else "", "VLLM_EXL3_NGRAM_TABLE": "disk"}
        model = {"repo": repo, "revision": revision, "local_snapshot": "/synthetic/model"}
        return config, options, research, environment, model

    def validate(self, config, options, research, environment, model, receipt=None):
        s = self.server
        documents = {str(s.PLAN): {"original_feature_identity": "/synthetic/reference.json"},
                     "/synthetic/reference.json": config.feature_identity}
        if receipt:
            documents[research["image_build_receipt"]] = receipt
        with patch.dict(os.environ, environment), \
                patch.object(s, "model_binding", return_value=(model, {Path("/synthetic/model/config.json")})), \
                patch.object(s, "sha", side_effect=lambda path: research["source_pins"].get(str(path), "e" * 64)), \
                patch.object(s, "read", side_effect=lambda path: documents[str(path)]):
            return s.validate_inputs(options, config, research)

    def test_three_audited_cells_accept_only_declared_reference_and_raw_policy(self):
        self.assertEqual(set(self.server.CELLS), {"vllm-nvidia", "vllm-radixark", "vllm-exl3-4.05"})
        for cell in self.server.CELLS:
            fixture = self.fixture(cell)
            before = deepcopy(fixture)
            binding = self.validate(*fixture)
            self.assertEqual(binding["model_artifact"], fixture[-1])
            self.assertIsNone(binding["calibration_artifact"])
            self.assertEqual(binding["image_binding"]["image_id"], fixture[2]["image_id"])
            self.assertEqual(fixture, before)

    def test_policy_substitutions_rejected_without_runtime_self_binding(self):
        for key, value in (("max_model_len", 8192), ("kv_cache_dtype", "float16"),
                           ("mm_processor_kwargs", {"truncation": True})):
            fixture = self.fixture()
            fixture[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(*fixture)

    def test_missing_source_pin_and_changed_image_are_recorded(self):
        fixture = self.fixture()
        del fixture[2]["source_pins"][str(self.server.AUDIT)]
        binding = self.validate(*fixture)
        self.assertEqual(binding["observed_source_pins"][str(self.server.AUDIT)], "e" * 64)
        fixture = self.fixture()
        fixture[3]["FUZEV_RESEARCH_IMAGE_ID"] = "sha256:" + "f" * 64
        self.validate(*fixture)

    def test_derived_exl3_records_build_receipt(self):
        fixture = self.fixture("vllm-exl3-4.05")
        research, environment = fixture[2:4]
        research.update(image_id="sha256:" + "f" * 64, parent_image_id=self.server.EXL3_IMAGE,
                        image_build_receipt="/synthetic/image-build.json")
        research["source_pins"][research["image_build_receipt"]] = "0" * 64
        environment["FUZEV_RESEARCH_IMAGE_ID"] = research["image_id"]
        receipt = {"image_id": research["image_id"], "parent_image_id": self.server.EXL3_IMAGE}
        binding = self.validate(*fixture, receipt=receipt)
        self.assertEqual(binding["image_binding"]["image_id"], research["image_id"])
        self.assertEqual(binding["image_binding"]["parent_image_id"], self.server.EXL3_IMAGE)
        receipt["image_id"] = self.server.EXL3_IMAGE
        binding = self.validate(*fixture, receipt=receipt)
        self.assertEqual(binding["image_binding"]["build_receipt"]["recorded"], receipt)

    def test_acquired_multimodal_model_and_revision_are_not_guessed(self):
        s = self.server
        _, repo, revision, quantization, _ = s.CELLS["vllm-nvidia"]
        acquired = {"repo": repo, "revision": revision, "local_snapshot": "/synthetic/snapshot"}
        options = {"model": "/synthetic/snapshot", "tokenizer": "/synthetic/snapshot",
                   "revision": revision, "tokenizer_revision": revision, "quantization": quantization}
        metadata = {"architectures": ["Qwen4ExpForConditionalGeneration"],
                    "text_config": {"hidden_size": 2560, "max_position_embeddings": 262144},
                    "vision_config": {"hidden_size": 1152}}
        with patch.object(s, "read", side_effect=lambda path: {"models": {"vllm-nvidia": acquired}}
                          if path == s.ACQUISITION else metadata):
            model, paths = s.model_binding(options, "vllm-nvidia")
            self.assertEqual(model["revision"], revision)
            self.assertIn(Path("/synthetic/snapshot/config.json"), paths)
            options["tokenizer_revision"] = "wrong"
            observed, _ = s.model_binding(options, "vllm-nvidia")
            self.assertEqual(observed["revision"], revision)
            options["tokenizer_revision"] = revision
            metadata["architectures"] = ["TextOnlySubstitution"]
            with self.assertRaises(ValueError):
                s.model_binding(options, "vllm-nvidia")

    def test_loaded_binary_binding_does_not_claim_worker_binary_attestation(self):
        s = self.server
        _, _, research, _, _ = self.fixture()
        package = Path("/synthetic/vllm")
        research["source_pins"].update({str(package.parent / name): "0" * 64 for name in s.READOUT_SOURCE_FILES})
        research["source_pins"][str(package / "_version.py")] = "0" * 64
        torch = ModuleType("torch")
        torch.__version__, torch.version = "2.13.0+cu130", NS(cuda="13.0")
        torch.ops = NS(loaded_libraries={str(package / "_C_stable_libtorch.abi3.so")})
        torch.cuda = NS(is_initialized=Mock(return_value=False))
        vllm, version = ModuleType("vllm"), ModuleType("vllm._version")
        vllm.__file__, version.__file__ = str(package / "__init__.py"), str(package / "_version.py")
        with patch.dict(sys.modules, {"torch": torch, "vllm": vllm, "vllm._version": version}), \
                patch.object(s, "sha", side_effect=lambda path: research["source_pins"].get(str(path), "0" * 64)):
            output = io.StringIO()
            with redirect_stdout(output):
                binding = s.loaded_binding(research, {})
            self.assertFalse(binding["api_cuda_initialized_before_head"])
            self.assertEqual(json.loads(output.getvalue())["torch_version"], "2.13.0+cu130")
            self.assertIn("not a worker binary attestation", binding["binary_binding"]["scope"])
            self.assertTrue(binding["binary_binding"]["libraries"]["_C_stable_libtorch.abi3.so"]["loaded_in_frontend"])
            torch.__version__ = "another-build"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(s.loaded_binding(research, {})["torch_version"], "another-build")

    def make_service(self):
        s = self.server
        config = self.fixture()[0]
        engine = NS(check_health=AsyncMock(), vllm_config=NS())
        head = NS(weights_sha256=s.HEAD_SHA, calibration_id="uncalibrated:T1", requires_lm_scores=True,
                  head_id="reference", provenance={"head_sha256": s.HEAD_SHA, "plan_sha256": s.PLAN_SHA,
                                                    "quality_sha256": s.QUALITY_SHA, "head_execution_device": "cuda:0"})
        provenance = {"runtime_cell": {"actual-observed-cell": True}, "runtime_sources": {"recipes": "synthetic-commit"}}
        service = s.ResearchVllmService(engine, None, head, config, provenance, None,
                                        research_binding={"source_pins": {"/synthetic/config.json": "0" * 64}})
        return service

    def test_provenance_is_augmented_before_identity_and_stable_across_receipts(self):
        service = self.make_service()
        initial = deepcopy(service.provenance)
        expected = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        self.assertEqual(service.identity, expected)
        first = asyncio.run(self.server.research_health(service))
        self.assertEqual(first["status"], "loaded")
        self.assertEqual(first["worker_readout"]["status"], "awaiting-worker-receipt")
        self.assertEqual(first["provenance"]["serving_identity"], first["provenance"]["runtime_cell"])
        self.assertEqual(first["provenance"]["reference_feature_identity"], service.config.feature_identity)
        service.receipt_status = "worker-verified"
        second = asyncio.run(self.server.research_health(service))
        self.assertEqual(second["worker_readout"]["status"], "worker-verified")
        self.assertEqual(service.provenance, initial)
        self.assertNotIn("worker_readout", service.provenance)
        self.assertNotIn("memory", service.provenance)

    def test_decision_keeps_existing_receipts_and_adds_only_research_envelope(self):
        service = self.make_service()
        result = asyncio.run(service.decide(None, "synthetic-request"))
        self.assertEqual(result["execution"], {"requests": ["synthetic-worker-receipt"]})
        self.assertEqual(result["head_sha256"], self.server.HEAD_SHA)
        self.assertEqual(result["head_manifest_sha256"], self.server.PLAN_SHA)
        for key, value in self.server.FLAGS.items():
            self.assertIs(result[key], value)
        service.engine.check_health.side_effect = RuntimeError("synthetic worker failure")
        with self.assertRaises(RuntimeError):
            asyncio.run(self.server.research_health(service))

    def test_opt_in_synthetic_trace_keeps_owned_result_and_context_private(self):
        import torch
        s = self.server
        identifier = 'engine_' + 'a' * 64
        value = torch.arange(2562, dtype=torch.float32)
        receipt = {'request_id': 'engine-internal', 'submitted_tokens': 20, 'num_cached_tokens': 0}
        call = AsyncMock(return_value=(value, receipt))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'reports/trace'
            target.mkdir(parents=True)
            config = target / 'config.json'
            output = target / 'readouts.jsonl'
            config.write_text(json.dumps({'request_ids': ['engine_' + character * 64 for character in 'abcd'], 'output': str(output)}))
            with patch.object(s, 'ROOT', root), patch.dict(os.environ, {'FUZEV_SYNTHETIC_READOUT_TRACE': str(config)}), \
                    patch.object(BaseService, '_engine_call', call, create=True):
                service = self.make_service()
                try:
                    async def exercise():
                        first = await service._engine_call((), None, features=True, baseline_token_ids=(357, 417))
                        self.assertIs(first[0], value)
                        self.assertEqual(output.read_bytes(), b'')
                        token = service._trace_request.set(identifier)
                        service._trace_claimed.add(identifier)
                        service._trace_counts[identifier] = 0
                        try:
                            traced = await service._engine_call((), None, features=True, baseline_token_ids=(357, 417))
                            self.assertIs(traced[0], value)
                        finally:
                            service._trace_request.reset(token)
                        self.assertIsNone(service._trace_request.get())
                        body = NS(primitive='choice', prefix_id='synthetic-prefix',
                                  question='Which marker belongs to branch A?',
                                  candidates=[NS(id='alpha'), NS(id='beta')])
                        second_id = 'engine_' + 'b' * 64
                        await service.decide(body, second_id)
                        with self.assertRaisesRegex(ValueError, 'one-use'):
                            await service.decide(body, second_id)
                        token = service._trace_request.set(identifier)
                        service._trace_counts[identifier] = 3
                        before = call.await_count
                        try:
                            with self.assertRaisesRegex(ValueError, 'two key'):
                                await service._engine_call((), None, features=True)
                            self.assertEqual(call.await_count, before)
                        finally:
                            service._trace_request.reset(token)
                    asyncio.run(exercise())
                    row = json.loads(output.read_bytes())
                    self.assertEqual(row['api_request_id'], identifier)
                    self.assertEqual(row['values'], value.tolist())
                    self.assertEqual(row['baseline_logits'], value[2560:].tolist())
                    self.assertEqual(row['sha256'], hashlib.sha256(value.numpy().tobytes()).hexdigest())
                    self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(service.provenance['source_pins'][str(config)], s.sha(config))
                    self.assertEqual(service.identity, hashlib.sha256(json.dumps(service.provenance, sort_keys=True).encode()).hexdigest())
                finally:
                    os.close(service._trace_fd)

    def test_synthetic_trace_rejects_panel_ids_and_existing_outputs(self):
        s = self.server
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'reports/trace'
            target.mkdir(parents=True)
            config = target / 'config.json'
            output = target / 'readouts.jsonl'
            for ids, exists in ((['kev_' + character * 64 for character in 'abcd'], False),
                                (['engine_' + character * 64 for character in 'abcd'], True)):
                config.write_text(json.dumps({'request_ids': ids, 'output': str(output)}))
                if exists:
                    output.write_text('preserved')
                with patch.object(s, 'ROOT', root), patch.dict(os.environ, {'FUZEV_SYNTHETIC_READOUT_TRACE': str(config)}), \
                        self.assertRaises(ValueError):
                    self.make_service()
            self.assertEqual(output.read_text(), 'preserved')
    def test_builder_preserves_strict_inspector_and_default_guard_admission(self):
        s = self.server
        config, options, research, _, _ = self.fixture()
        service = self.make_service()
        s.build_service.return_value = service
        with patch.object(s, "validate_inputs", return_value={}), patch.object(s, "loaded_binding", return_value={}), \
                patch.object(s, "validate_ple_storage_config"):
            self.assertIs(asyncio.run(s.build_research_service(service.engine, config, options=options, research=research)), service)
        kwargs = s.build_service.call_args.kwargs
        self.assertIs(kwargs["inspector"], s.describe_runtime)
        self.assertIs(kwargs["head_factory"], s.KevVllmResearchHead)
        self.assertIs(kwargs["service_factory"].func, s.ResearchVllmService)
        self.assertNotIn("guard_factory", kwargs)
        self.assertNotIn("admission", kwargs)
        service.engine.check_health.assert_awaited_once()

    def test_single_engine_lifecycle_rechecks_policy_and_shuts_down_on_failure(self):
        s = self.server
        modules = {name: ModuleType(name) for name in ("vllm", "vllm.engine", "vllm.engine.arg_utils",
                    "vllm.v1", "vllm.v1.engine", "vllm.v1.engine.async_llm", "fuzev_research.telemetry")}
        resolved = NS(model_config=NS(quantization="modelopt_mixed"))
        args = Mock(return_value=NS(create_engine_config=lambda: resolved))
        engine = NS(shutdown=Mock())
        construct = Mock(return_value=engine)
        modules["vllm.engine.arg_utils"].AsyncEngineArgs = args
        modules["vllm.v1.engine.async_llm"].AsyncLLM = NS(from_vllm_config=construct)
        modules["fuzev_research.telemetry"].admit = Mock()

        async def enter():
            async with s.engine_context({"synthetic": True}) as actual:
                self.assertIs(actual, engine)
                raise RuntimeError("synthetic shutdown path")

        with patch.dict(sys.modules, modules), patch.object(s, "validate_ple_storage_options") as options_check, \
                patch.object(s, "validate_ple_storage_config") as resolved_check:
            with self.assertRaises(RuntimeError):
                asyncio.run(enter())
            construct.assert_called_once_with(resolved)
            engine.shutdown.assert_called_once()
            options_check.assert_called_once()
            resolved_check.assert_called_once()
            modules["fuzev_research.telemetry"].admit.assert_called_once_with(128 << 20)

    def test_cli_binds_exact_parsed_config_bytes_without_overwriting_runtime_pins(self):
        s = self.server
        fixture = self.fixture()
        config, options, research = fixture[:3]
        envelope = {**vars(config), "research": research}
        uvicorn = ModuleType("uvicorn")
        uvicorn.run = Mock()
        with TemporaryDirectory() as directory:
            engine_path, service_path = Path(directory) / "engine.json", Path(directory) / "service.json"
            engine_body, service_body = json.dumps(options).encode(), json.dumps(envelope).encode()
            engine_path.write_bytes(engine_body)
            service_path.write_bytes(service_body)
            with patch.dict(sys.modules, {"uvicorn": uvicorn}), \
                    patch.object(sys, "argv", ["research", "--engine-config", str(engine_path), "--service-config", str(service_path)]), \
                    patch.object(s, "validate_inputs") as validate, patch.object(s, "create_research_app"):
                s.main()
            observed_options, observed_config, observed_research = validate.call_args.args
            self.assertEqual(observed_options, options)
            self.assertEqual(observed_config.runtime_cell, config.runtime_cell)
            self.assertEqual(observed_config.runtime_sources, config.runtime_sources)
            for path, body in ((engine_path, engine_body), (service_path, service_body)):
                self.assertEqual(observed_research["source_pins"][str(path)], hashlib.sha256(body).hexdigest())

    def test_explicit_capture_callback_reuses_supplied_engine_and_original_intent(self):
        s = self.server
        helper, uvicorn = ModuleType('fuzev_research.kev_vllm_capture'), ModuleType('uvicorn')
        operator = {'service': {'model_name': 'vllm-nvidia', 'prefix_ttl_seconds': 900.},
                    'options': {'observed': True}, 'research': {'operator': True}}
        helper.capture_operator = Mock(return_value=operator)
        service = NS()
        helper.capture_and_build = AsyncMock(return_value=service)
        uvicorn.run = Mock()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cross-cell').mkdir()
            output = root / 'cross-cell/capture.json'
            argv = ['research', '--engine-config', 'engine.json', '--capture-runtime-out', str(output),
                    '--model', 'vllm-nvidia', '--binding-receipt', 'binding.json']
            with patch.dict(sys.modules, {helper.__name__: helper, 'uvicorn': uvicorn}), \
                    patch.object(sys, 'argv', argv), patch.object(s, 'ROOT', root), \
                    patch.object(s, 'create_research_app') as create, patch.object(s, 'validate_inputs') as validate:
                s.main()
                validate.assert_not_called()
                callback = create.call_args.kwargs['builder']
                context, intent = create.call_args.args
                self.assertIs(context.func, s.engine_context)
                self.assertEqual(context.args, (operator['options'],))
                self.assertEqual(vars(intent), operator['service'])
                existing = NS()
                self.assertIs(asyncio.run(callback(existing, intent)), service)
                helper.capture_and_build.assert_awaited_once_with(existing, operator, output)
                helper.capture_operator.assert_called_once_with('vllm-nvidia', 'engine.json',
                    Path('binding.json'), image_build_receipt=None)
                uvicorn.run.assert_called_once()

    def test_capture_modes_cannot_overwrite_or_escape_or_change_normal_serving(self):
        s = self.server
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cross-cell').mkdir()
            existing = root / 'cross-cell/existing.json'
            existing.write_text('preserve')
            for destination in (existing, root / 'outside.json', root / 'cross-cell/missing/out.json'):
                argv = ['research', '--engine-config', 'engine.json', '--capture-runtime-out', str(destination),
                        '--model', 'vllm-nvidia', '--binding-receipt', 'binding.json']
                with patch.object(sys, 'argv', argv), patch.object(s, 'ROOT', root):
                    with self.assertRaises(ValueError):
                        s.main()
            self.assertEqual(existing.read_text(), 'preserve')
            argv = ['research', '--engine-config', 'engine.json', '--service-config', 'service.json',
                    '--model', 'vllm-nvidia']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), \
                    patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    s.main()

    def test_custom_research_builder_is_explicit_and_default_remains_strict(self):
        s = self.server
        context, config, callback = Mock(), NS(model_name='vllm-nvidia'), AsyncMock()
        s.create_research_app(context, config, options={'a': 1}, research={'b': 2}, builder=callback)
        self.assertIs(s.create_app.call_args.kwargs['builder'], callback)
        s.create_research_app(context, config, options={'a': 1}, research={'b': 2})
        default = s.create_app.call_args.kwargs['builder']
        self.assertIs(default.func, s.build_research_service)
        self.assertEqual(default.keywords, {'options': {'a': 1}, 'research': {'b': 2}})


if __name__ == "__main__":
    unittest.main()
