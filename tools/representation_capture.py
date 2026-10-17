"""Request-scoped capture of the native shared head's final-token inputs."""

from contextlib import ExitStack
from unittest.mock import patch

import torch


class NativeProbeCapture:
    def __init__(self, service, request, prepared):
        self.service, self.request, self.prepared = service, request, prepared
        self.stack = ExitStack()
        self.control, self.result, self.failure = None, None, None
        self.jobs, self.events, self.streams = {}, {}, {}
        self.errors = []
        self.sequences = [tuple(prepared.query)] + [tuple(prepared.query + tail) for tail in prepared.tails]

    async def __aenter__(self):
        try:
            await self.service.async_generator.run_on_owner(self.install)
        except BaseException as error:
            await self.__aexit__(type(error), error, error.__traceback__)
            raise
        return self

    async def __aexit__(self, kind, error, traceback):
        self.failure = None if error is None else f"{type(error).__name__}: {error}"
        await self.service.async_generator.run_on_owner(self.finish)
        return False

    def install(self):
        head = self.service.head.head
        original = head.score

        def score(primitive, query, candidates, *, baseline, modality="text"):
            self.control = tuple(self.own(value) for value in (query, candidates, baseline))
            return original(primitive, query, candidates, baseline=baseline, modality=modality)

        self.stack.enter_context(patch.object(head, "score", score))

    def own(self, value):
        if value.is_cuda:
            stream = torch.cuda.current_stream(value.device)
            self.streams[(str(value.device), stream.cuda_stream)] = stream
        return value.detach().float().clone()

    def register(self, job):
        tokens = tuple(job.sequences[0].sequence_ids.torch().reshape(-1).tolist())
        self.jobs[id(job)] = [index for index, sequence in enumerate(self.sequences) if sequence == tokens]

    def complete(self, job, event):
        for index in self.jobs.get(id(job), []):
            self.events[index] = dict(features=self.own(event["features"]),
                cached_tokens=event["cached_tokens"], forwarded_tokens=event["forwarded_tokens"],
                root_available=event.get("readout_root") is not None)

    def finish(self):
        self.stack.close()
        try:
            for stream in self.streams.values():
                stream.synchronize()
            self.result = self.drain()
        finally:
            self.control = None
            self.events.clear()
            self.jobs.clear()
            self.streams.clear()

    def drain(self):
        ids = [candidate.id for candidate in self.request.candidates]
        result = dict(candidate_ids=ids, coverage=[], errors=self.errors,
                      failure=self.failure, views={})
        if self.control is None:
            result["views"]["control"] = dict(unavailable="No whole-slate score inputs")
            return result
        query, candidates, baseline = (value.cpu() for value in self.control)
        result["views"]["control"] = dict(query=query.tolist(), candidates=candidates.tolist())
        result["baseline"] = baseline.tolist()
        for index, expected in enumerate((query, *candidates)):
            event = dict(self.events.get(index, {}))
            published = event.pop("features", None)
            result["coverage"].append(dict(candidate_id=None if index == 0 else ids[index - 1],
                event=event, published_matches_score=bool(torch.equal(published.cpu(), expected))
                if published is not None else None))
        return result
