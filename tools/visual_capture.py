"""Capture the control representation used by image residual selection."""

import asyncio
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch


@contextmanager
def observe(service, *, probe):
    run = service._run

    async def capture(job):
        await service.async_generator.run_on_owner(lambda: probe.register(job))
        event = await run(job)
        try:
            await service.async_generator.run_on_owner(lambda: probe.complete(job, event))
        except BaseException:
            if (root := event.get("readout_root")) is not None:
                await service.async_generator.run_on_owner(root.close)
            raise
        return event

    with patch.object(service, "_run", capture):
        yield


async def measure(service, request, *, probe):
    async with probe:
        with observe(service, probe=probe):
            response = await service.decide(request, prepared=probe.prepared)
    return response, probe.result


async def run_representation_capture(service, paths, output, *, image_dir=Path(".")):
    from prepare_sugarcrepe import native_request
    from representation_capture import NativeProbeCapture

    output["scope"] = "Research representation capture; fixed views, no fitting or timing claim"
    output["representation_inputs"] = [Path(path).name for path in paths]
    output["representation_rows"] = []
    last_group, last_inputs, state = None, None, None
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                request = native_request(source, image_dir)
                route, modality = service._admit_decision(request)
                inputs = (request.messages, request.image, request.additional_context, request.prefix_id)
                if source["group"] != last_group or inputs != last_inputs:
                    state = await service.prepare_state(request)
                    last_group, last_inputs = source["group"], deepcopy(inputs)
                prepared = await asyncio.to_thread(service._prepare_decision, request, state, "candidates",
                                                   route=route, modality=modality)
                probe = NativeProbeCapture(service, request, prepared)
                row = {name: deepcopy(source[name]) for name in ("id", "group", "role", "order", "target")}
                row.update(primitive=request.primitive, modality=modality, source_modality=source["modality"],
                           acceptable_candidate_ids=[source["positive_id"]], route=route)
                output["representation_rows"].append(row)
                try:
                    response, capture = await measure(service, request, probe=probe)
                    row.update(capture, execution=response["execution"],
                               head={"scores": [{"id": item["id"], "logit": item["logit"]}
                                                for item in response["scores"]]})
                except BaseException as error:
                    row.update(probe.result or {}, error=f"{type(error).__name__}: {error}")
                    raise


def main(argv=None):
    import uvicorn
    from fuzev_exllamav3.server import parser, serve

    cli = parser()
    cli.description = __doc__
    cli.add_argument("--representation-input", type=Path, nargs="+", required=True)
    cli.add_argument("--image-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)

    async def capture(server):
        service = server.config.app.state.fuzev_routes.get_service()
        result = {}
        try:
            await run_representation_capture(service, args.representation_input, result,
                                             image_dir=args.image_dir)
        finally:
            result["bounds"] = {"context_tokens": service.context_tokens,
                "cache_tokens": service.generator.cache.max_num_tokens,
                "max_batch_size": service.generator.max_batch_size,
                "max_chunk_size": service.generator.max_chunk_size}
            result["model"] = Path(args.model).name
            if args.head:
                result["head"] = (Path(args.head) / "VERSION").read_text().strip()
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    with patch.object(uvicorn.Server, "serve", capture):
        asyncio.run(serve(args))


if __name__ == "__main__":
    main()
