import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from fuzev.encoding import candidate_suffix
from fuzev.service_contracts import ServiceDecisionRequest
from representation_capture import NativeProbeCapture
from visual_capture import measure, run_representation_capture


class Owner:
    def __init__(self):
        self.active = False
        self.cleanup_gate = None
        self.cleanup_started = asyncio.Event()

    async def run_on_owner(self, function):
        async def execute():
            if function.__name__ == "finish" and self.cleanup_gate is not None:
                self.cleanup_started.set()
                await self.cleanup_gate.wait()
            self.active = True
            try:
                return function()
            finally:
                self.active = False
        task = asyncio.create_task(execute())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
            task.result()
            raise


class Site:
    def __init__(self, owner, key, offset):
        self.owner, self.key, self.offset = owner, key, offset
        self.scratch = None

    def value(self, x):
        assert self.owner.active
        if self.scratch is None or self.scratch.shape != x.shape:
            self.scratch = torch.empty_like(x)
        self.scratch.copy_(x + self.offset)
        return self.scratch

    def mix(self, streams, params):
        return torch.full((*streams.shape[:2], 2), -999.), None, self.value(streams)

    def forward(self, x, params, out_dtype=None):
        value = self.value(x)
        return value if out_dtype is None else value.to(out_dtype)


class Model:
    def __init__(self, owner):
        self.owner, self.skip_hooks, self.fail = owner, False, False
        self.blocks = [SimpleNamespace(mlp_hc=Site(owner, f"layers.{i}.mlp_hc", offset))
                       for i, offset in enumerate((1, 10, 100))]
        self.final = Site(owner, "model.hyper_connection_mixer", 1000)
        self.final.use_combine = False
        self.fwd_modules = [(module, None, i) for i, module in enumerate([*self.blocks, self.final])]

    def forward_features(self, input_ids, params=None):
        assert self.owner.active
        assert params["native"]
        x = input_ids.float()[..., None] + torch.arange(4)
        if self.skip_hooks:
            return (x[:, -1:] + 1111).clone()
        for block in self.blocks:
            x = block.mlp_hc.mix(x, params)[2]
        if self.fail:
            raise RuntimeError("native forward failed")
        x = self.final.forward(x, params, out_dtype=torch.float32)
        return x[:, -1:].float().clone()


class SequenceIDs:
    def __init__(self, tokens):
        self.value = torch.tensor([tokens])

    def torch(self):
        return self.value

    def __len__(self):
        return self.value.shape[1]


class Job:
    def __init__(self, generator, tokens, start, root=None):
        self.generator, self.readout_root = generator, root
        self.readout_serial = root is None
        self.start = start
        self.sequences = [SimpleNamespace(sequence_ids=SequenceIDs(tokens), kv_position=start)]

    def event(self, feature):
        seq = self.sequences[0]
        return dict(job=self, features=feature[0, -1].clone(), sequence_ids=seq.sequence_ids.torch().clone(),
                    cached_tokens=self.start, forwarded_tokens=len(seq.sequence_ids) - self.start,
                    readout_root=object() if self.start < 3 else None)

    def finish_readout(self):
        seq = self.sequences[0]
        feature = self.generator.model.forward_features(
            input_ids=seq.sequence_ids.torch()[:, seq.kv_position:], params={"native": True})
        seq.kv_position = len(seq.sequence_ids)
        return self.event(feature)


class Generator:
    def __init__(self, owner):
        self.model = Model(owner)

    def _readout_batch(self, jobs, results):
        width = min(7, *(len(j.sequences[0].sequence_ids) - j.sequences[0].kv_position for j in jobs))
        ids = torch.cat([j.sequences[0].sequence_ids.torch()[:, j.sequences[0].kv_position:
                         j.sequences[0].kv_position + width] for j in jobs])
        features = self.model.forward_features(ids, {"native": True})
        for index, job in enumerate(jobs):
            seq = job.sequences[0]
            seq.kv_position += width
            if seq.kv_position == len(seq.sequence_ids):
                results.append(job.event(features[index:index + 1]))


class RuntimeHead:
    def __init__(self, owner):
        self.owner = owner

    def score(self, primitive, query, candidates, *, baseline, modality="text"):
        assert self.owner.active and primitive == "choice" and modality == "image_text"
        scores = baseline + candidates[:, 0] / 100
        return scores.tolist(), scores.softmax(0).tolist(), scores.softmax(0).tolist()


class Service:
    def __init__(self, *, serial=False, mismatch=False, skip_hooks=False):
        self.async_generator = Owner()
        self.generator = Generator(self.async_generator)
        self.generator.model.skip_hooks = skip_hooks
        self.head = SimpleNamespace(metadata={"hidden_size": 4}, head=RuntimeHead(self.async_generator),
                                    lm=SimpleNamespace(forward=lambda *args: None))
        self.preparer = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=[ord(c) + int(mismatch) for c in text],
                                                          offsets=[(i, i + 1) for i in range(len(text))]))))
        self.serial = serial
        self.pending, self.native_jobs = [], []
        self.dispatch = None

    def _admit_decision(self, request):
        assert request.image is not None
        return "labelled", "image_text"

    async def prepare_state(self, request):
        return SimpleNamespace(tokens=(1, 2, 3))

    def _prepare_decision(self, request, state, context_field, *, route, modality):
        return SimpleNamespace(query=state.tokens, modality=modality, route=route,
            tails=tuple(tuple(ord(c) for c in candidate_suffix(candidate.model_dump()))
                        for candidate in request.candidates))

    async def _run(self, job):
        self.native_jobs.append(job)
        if job.readout_root is None:
            return await self.async_generator.run_on_owner(job.finish_readout)
        future = asyncio.get_running_loop().create_future()
        self.pending.append((job, future))
        if self.dispatch is None:
            self.dispatch = asyncio.create_task(self.flush())
        return await future

    async def flush(self):
        await asyncio.sleep(0)
        pending = list(reversed(self.pending))
        jobs = [job for job, _ in pending]
        try:
            while jobs:
                results = []
                await self.async_generator.run_on_owner(lambda: self.generator._readout_batch(jobs, results))
                for result in results:
                    jobs.remove(result["job"])
                    next(f for j, f in pending if j is result["job"]).set_result(result)
        except BaseException as error:
            for _, future in pending:
                if not future.done():
                    future.set_exception(error)
        finally:
            self.pending.clear()
            self.dispatch = None

    async def decide(self, request, *, prepared):
        query = await self._run(Job(self.generator, prepared.query, len(prepared.query) - 1))
        root = None if self.serial else object()
        jobs = [Job(self.generator, prepared.query + tail,
                    len(prepared.query + tail) - 1 if self.serial else len(prepared.query), root)
                for tail in prepared.tails]
        keys = await asyncio.gather(*(self._run(job) for job in jobs))
        scores, raw, calibrated = await self.async_generator.run_on_owner(lambda: self.head.head.score(
            request.primitive, query["features"], torch.stack([event["features"] for event in keys]),
            baseline=torch.arange(len(keys)).float(), modality=prepared.modality))
        return dict(selected_id=request.candidates[max(range(len(scores)), key=scores.__getitem__)].id,
                    scores=[dict(id=c.id, logit=z, raw_probability=p, calibrated_probability=t)
                            for c, z, p, t in zip(request.candidates, scores, raw, calibrated)],
                    execution={"forwarded_tokens": sum(e["forwarded_tokens"] for e in [query, *keys])})


def inputs(*, duplicate=False):
    request = ServiceDecisionRequest(primitive="choice", question="Caption?",
        messages=[dict(role="user", content="Consider the supplied image.")],
        candidates=[dict(id="left", text="a"), dict(id="right", text="a" if duplicate else "bcd")])
    prepared = SimpleNamespace(query=(1, 2, 3), modality="image_text", route="labelled",
        tails=tuple(tuple(ord(c) for c in candidate_suffix(candidate.model_dump())) for candidate in request.candidates))
    return request, prepared



class NativeCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_matches_scored_inputs_and_owns_its_features(self):
        service = Service()
        request, prepared = inputs()
        reference = await Service().decide(request, prepared=prepared)
        probe = NativeProbeCapture(service, request, prepared)
        response, result = await measure(service, request, probe=probe)
        self.assertEqual(response, reference)
        self.assertEqual(result["candidate_ids"], ["left", "right"])
        self.assertEqual(result["baseline"], [0., 1.])
        self.assertEqual(result["views"]["control"]["query"], [1114., 1115., 1116., 1117.])
        self.assertEqual(set(result["views"]), {"control"})
        self.assertTrue(all(frame["published_matches_score"] for frame in result["coverage"]))
        saved = deepcopy(result)
        await service.decide(request, prepared=prepared)
        self.assertEqual(result, saved)

    async def test_serial_and_batched_capture_have_same_control(self):
        request, prepared = inputs()
        outputs = []
        for serial in (False, True):
            service = Service(serial=serial)
            probe = NativeProbeCapture(service, request, prepared)
            _, result = await measure(service, request, probe=probe)
            outputs.append(result["views"]["control"])
        self.assertEqual(outputs[0], outputs[1])

    async def test_error_is_preserved_and_next_request_can_score(self):
        service = Service()
        service.generator.model.fail = True
        request, prepared = inputs()
        probe = NativeProbeCapture(service, request, prepared)
        with self.assertRaises(RuntimeError):
            await measure(service, request, probe=probe)
        self.assertIn("native forward failed", probe.result["failure"])
        service.generator.model.fail = False
        result = await service.decide(request, prepared=prepared)
        self.assertEqual(len(result["scores"]), 2)

    async def test_visual_driver_preserves_roles_orders_and_targets(self):
        from PIL import Image

        request, _ = inputs()
        rows = []
        for role in ("fit", "development"):
            for order in ("ab", "ba"):
                candidates = [candidate.model_dump() for candidate in request.candidates]
                if order == "ba":
                    candidates.reverse()
                rows.append(dict(id=role + "-" + order, group=role, role=role, order=order,
                    target=[1., 0.] if order == "ab" else [0., 1.], positive_id="left",
                    primitive="choice", modality="image", context=request.messages,
                    question=request.question, candidates=candidates, image_path="sample.png"))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "white").save(root / "sample.png")
            path = root / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = {}
            await run_representation_capture(Service(), [path], output, image_dir=root)
        captured = output["representation_rows"]
        self.assertEqual([r["role"] for r in captured], ["fit", "fit", "development", "development"])
        for actual, source in zip(captured, rows):
            self.assertEqual(actual["target"], source["target"])
            self.assertEqual(actual["candidate_ids"], [c["id"] for c in source["candidates"]])
            self.assertEqual(actual["acceptable_candidate_ids"], ["left"])
            self.assertEqual(set(actual["views"]), {"control"})
        json.dumps(output, allow_nan=False)
