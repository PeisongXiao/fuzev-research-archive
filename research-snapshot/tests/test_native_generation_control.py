"""Pinned-API CPU fakes only: no stock engine import, inference or CUDA proof."""

import json
import subprocess
import sys
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fuzev_research import native_generation_control as control


class History:
    def __init__(self, ids):
        self.ids = ids.clone()

    def __len__(self):
        return self.ids.shape[-1]

    def torch(self):
        return self.ids

    def torch_slice(self, a, b):
        return self.ids[:, a:b]


class NGram:
    def __init__(self):
        self.mode, self.tables, self.handles = "trellis_disk", None, [object()]
        self._pending, self._pins = [], []


class Model:
    def __init__(self):
        self.config = SimpleNamespace(architecture="Qwen4ExpForConditionalGeneration",
                                      vocab_size=512, eos_token_id_list=[99])
        self.loaded_tp, self.output_device, self.active_devices = False, "cpu", []
        self.caps, self.ngram = {"recurrent_states": True}, NGram()

    def __iter__(self):
        yield self.ngram

    def forward(self, *args, **kwargs):
        raise AssertionError("The control must never call/wrap forward itself")


class FusedStep:
    pass


class ComboSampler:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.fused_only, self.steps = True, [FusedStep()]


class Job:
    def __init__(self, input_ids, **kwargs):
        self.options, self.sampler = kwargs, kwargs["sampler"]
        self.identifier, self.max_new_tokens = kwargs["identifier"], kwargs["max_new_tokens"]
        self.generator, self.serial_number, self.recurrent_state = None, None, None
        self.sequences = [SimpleNamespace(input_ids=History(input_ids), sequence_ids=History(input_ids),
                                          allocated_pages=None)]
        self.new_tokens, self.is_finished, self.deallocations = 0, False, 0
        self.deallocation_error = None

    def prepare_for_queue(self, generator, serial_number):
        self.generator, self.serial_number = generator, serial_number
        self.max_rq_tokens = self.max_new_tokens + 1

    def deallocate_pages(self):
        self.deallocations += 1
        if self.deallocation_error is not None:
            raise self.deallocation_error
        if self.recurrent_state is not None:
            self.generator.cache.free_list.append(self.recurrent_state.slot)
            self.recurrent_state = None
        for seq in self.sequences:
            for page in seq.allocated_pages or []:
                page.ref_count = 0
                self.generator.pagetable.referenced_pages.pop(page.page_index)
                self.generator.pagetable.unreferenced_pages[page.page_index] = page
            seq.allocated_pages = []


class Generator:
    def __init__(self, slots=4, pages=16):
        self.model = Model()
        self.cache = SimpleNamespace(initialized=True, owner_serial=1, num_slots=slots,
                                     free_list=list(range(slots)), max_num_tokens=pages * 256)
        all_pages = [SimpleNamespace(page_index=i, ref_count=0) for i in range(pages)]
        self.pagetable = SimpleNamespace(generator=self, cache=self.cache, max_pages=pages,
                                         all_pages=all_pages, referenced_pages={},
                                         unreferenced_pages={p.page_index: p for p in all_pages})
        self.cache_owner_serial, self.max_total_tokens, self.max_batch_size = 1, pages * 256, slots
        self.pending_jobs, self.active_jobs, self.filter_queue = [], [], []
        self.draft_model, self.draft_cache, self.recurrent_cache = None, None, object()
        self.num_draft_tokens, self.ngram_match_min, self.dynamic_draft = 0, 0, False
        self.job_serial, self.iterations, self.cancelled, self.created = 0, 0, [], []
        self.scripts, self.cached, self.phase = {}, (0, 0), (.1, .2, .3)
        self.enqueue_fail_at, self.callback, self.mutate_events = None, None, None
        self.tokenizer = SimpleNamespace(encode=self.encode)

    @staticmethod
    def encode(text, *, add_bos, encode_special_tokens):
        return torch.tensor([[1] * add_bos + [ord(char) + 1 for char in text]], dtype=torch.long)

    def enqueue(self, job):
        self.created.append(job)
        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        if len(self.created) == self.enqueue_fail_at:
            raise RuntimeError("enqueue failed after attaching job")
        self.pending_jobs.append(job)
        return job.serial_number

    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def get_cache_stats(self):
        return {"page_size": 256, "max_tokens": self.max_total_tokens,
                "used_tokens": len(self.pagetable.referenced_pages) * 256,
                "alloc_pages": len(self.created), "alloc_cached_pages": self.cached[0]}

    def clear_queue(self):
        raise AssertionError("Never clear a borrowed stock queue")

    def cancel(self, job):
        self.cancelled.append(job)
        if job in self.pending_jobs:
            self.pending_jobs.remove(job)
        elif job in self.active_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)

    def iterate(self):
        self.iterations += 1
        events = []
        for job in self.pending_jobs[:]:
            self.pending_jobs.remove(job)
            self.active_jobs.append(job)
            slot = self.cache.free_list.pop(0)
            job.recurrent_state = SimpleNamespace(slot=slot)
            count = (len(job.sequences[0].input_ids) + job.max_rq_tokens + 255) // 256
            pages = list(self.pagetable.unreferenced_pages.values())[:count]
            assert len(pages) == count
            for page in pages:
                page.ref_count = 1
                self.pagetable.unreferenced_pages.pop(page.page_index)
                self.pagetable.referenced_pages[page.page_index] = page
            job.sequences[0].allocated_pages = pages
            events.append({"job": job, "serial": job.serial_number, "stage": "started", "eos": False})
        for job in self.active_jobs[:]:
            script = self.scripts.get(job.identifier, [91, 92, 99])
            token = script[job.new_tokens % len(script)]
            history = job.sequences[0].sequence_ids
            history.ids = torch.cat((history.ids, torch.tensor([[token]])), -1)
            job.new_tokens += 1
            eos = token in job.options["stop_conditions"] or job.new_tokens == job.max_new_tokens
            event = {"job": job, "serial": job.serial_number, "stage": "streaming", "eos": eos}
            if token not in job.options["stop_conditions"]:
                event["token_ids"] = torch.tensor([[token]])
            if eos:
                job.is_finished = True
                reason = "stop_token" if token in job.options["stop_conditions"] else "max_new_tokens"
                event.update(eos_reason=reason, full_completion="synthetic completion",
                             new_tokens=job.new_tokens, prompt_tokens=len(job.sequences[0].input_ids),
                             time_enqueued=self.phase[0], time_prefill=self.phase[1], time_generate=self.phase[2],
                             cached_pages=self.cached[0], cached_tokens=self.cached[1])
                if reason == "stop_token":
                    event["eos_triggering_token_id"] = token
                job.deallocate_pages()
                self.active_jobs.remove(job)
            events.append(event)
        if self.callback:
            self.callback(self, events)
        return self.mutate_events(events) if self.mutate_events else events


class Guard:
    def __init__(self):
        self.started, self.closed, self.checks = False, False, 0
        self.start_error, self.fail_check = None, None

    def start(self):
        self.started = True
        if self.start_error:
            raise self.start_error
        return self

    def check(self):
        self.checks += 1
        if self.fail_check == self.checks:
            raise MemoryError("synthetic sustained swap/headroom failure")

    def close(self):
        self.closed = True


@pytest.fixture
def rig(monkeypatch):
    generator, guard, admissions = Generator(), Guard(), []
    monkeypatch.setattr(control, "_stock_api", lambda: (torch, Generator, Job, ComboSampler, NGram, Model, "cpu_fake"))
    monkeypatch.setattr(control, "MemoryGuard", lambda **kwargs: guard)
    monkeypatch.setattr(control, "admit", lambda size, fraction: admissions.append((size, fraction)))
    return SimpleNamespace(generator=generator, guard=guard, admissions=admissions)


def request(name="one", prompt="A", **kwargs):
    values = dict(name=name, rendered_prompt=prompt, prompt_token_ids=tuple(ord(c) + 1 for c in prompt),
                  add_bos=False, encode_special_tokens=True, max_new_tokens=3,
                  temperature=1., top_p=.95, top_k=20, seed=17)
    values.update(kwargs)
    return control.GenerationControlRequest(**values)


def run(rig, requests=None, **kwargs):
    options = dict(allow_generation=True, exclusive=True, additional_peak_bytes=1 << 20,
                   estimate_basis="tiny CPU fake allocations, not a native peak estimate")
    options.update(kwargs)
    return control.run_native_generation_control(rig.generator, requests or [request()], **options)


def assert_idle(rig):
    control._idle(rig.generator)
    assert rig.guard.closed
    assert rig.generator not in control._owners


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_stock_batch_exact_prompt_accepted_eos_and_counts(rig, batch):
    requests = [request(str(i), prompt="A" * (i + 1)) for i in range(batch)]
    result = run(rig, requests)
    assert result["qualified"] is False and result["evidence_kind"] == "cpu_fake"
    assert result["batch_size"] == batch and result["iterations"] == 3
    assert result["outer_monotonic_seconds"] > 0 and result["peak_memory_qualified"] is False
    for row, spec, job in zip(result["jobs"], requests, rig.generator.created):
        assert row["prompt_token_ids"] == list(spec.prompt_token_ids)
        assert row["accepted_completion_token_ids"] == [91, 92, 99]
        assert row["streamed"] == [91, 92]  # Stock stream omits the stopping EOS.
        assert row["prompt_tokens"] == len(spec.prompt_token_ids) and row["completion_tokens"] == 3
        assert row["physical_forwarded_tokens"] is None and row["stock_phase_timing"]["ttft_seconds"] is None
        assert row["stock_phase_timing"]["clock"] == "stock time.time"
        assert row["first_accepted_observed_seconds"] <= row["last_accepted_observed_seconds"]
        assert job.options["token_healing"] is False and job.options["max_rq_tokens"] is None
        assert job.options["return_logits"] is False and job.options["filters"] is None
        assert job.sampler.options == dict(temperature=1., top_p=.95, top_k=20, min_p=0., rep_p=1.,
                                          freq_p=0., pres_p=0., dry_multiplier=0., temp_last=False,
                                          adaptive_target=1., logit_bias=None)
    json.dumps(result, allow_nan=False)
    assert rig.admissions[0][0] > 1 << 20 and rig.admissions[0][1] is None
    assert_idle(rig)


@pytest.mark.parametrize("script,limit,expected,reason", [([99], 3, [99], "stop_token"),
                                                         ([91, 92, 99], 1, [91], "max_new_tokens")])
def test_single_sample_includes_eos_or_exact_limit(rig, script, limit, expected, reason):
    rig.generator.scripts["one"] = script
    row = run(rig, [request(max_new_tokens=limit)])["jobs"][0]
    assert row["accepted_completion_token_ids"] == expected and row["eos_reason"] == reason
    assert row["completion_tokens"] == 1
    assert_idle(rig)


@pytest.mark.parametrize("temperature,reference,status", [(0., (91, 92, 99), "equal"),
                                                        (0., (88,), "different"),
                                                        (1., (88,), "different"), (1., None, "not_supplied")])
def test_comparison_records_greedy_or_stochastic_difference_without_rng_claim(rig, temperature, reference, status):
    row = run(rig, [request(temperature=temperature, native_service_completion_token_ids=reference)])["jobs"][0]
    assert row["native_service_comparison"]["status"] == status
    assert row["native_service_comparison"]["rng_equivalence_claimed"] is False


def test_stock_cache_reuse_and_wall_clock_phases_are_reported_not_forward_counts(rig):
    rig.generator.cached, rig.generator.phase = (1, 256), (.1, .2, -.01)
    row = run(rig, [request(prompt="A" * 257)])["jobs"][0]
    assert row["stock_cached_pages"] == 1 and row["stock_cached_tokens"] == 256
    assert row["physical_forwarded_tokens"] is None
    assert row["stock_phase_timing"]["status"] == "invalid_wall_clock"
    assert row["stock_phase_timing"]["raw_seconds"]["time_generate"] == -.01


@pytest.mark.parametrize("flag", ["allow_generation", "exclusive"])
def test_disabled_control_never_resolves_stock_bindings(rig, monkeypatch, flag):
    monkeypatch.setattr(control, "_stock_api", lambda: pytest.fail("No engine lookup while disabled"))
    with pytest.raises(ValueError, match="explicit offline"):
        run(rig, **{flag: False})
    assert not rig.guard.started and not rig.generator.created


@pytest.mark.parametrize("occupied", ["pending", "active", "filter", "slot", "page", "superseded"])
def test_occupied_or_superseded_owner_is_untouched(rig, occupied):
    gen = rig.generator
    if occupied in ("pending", "active", "filter"):
        getattr(gen, {"pending": "pending_jobs", "active": "active_jobs", "filter": "filter_queue"}[occupied]).append(object())
    elif occupied == "slot":
        gen.cache.free_list.pop()
    elif occupied == "page":
        gen.pagetable.all_pages[0].ref_count = 1
    else:
        gen.cache.owner_serial += 1
    before = (list(gen.pending_jobs), list(gen.active_jobs), list(gen.cache.free_list))
    with pytest.raises(ValueError):
        run(rig)
    assert before == (gen.pending_jobs, gen.active_jobs, gen.cache.free_list)
    assert not gen.created and not gen.cancelled and not rig.guard.started


@pytest.mark.parametrize("field,value", [("draft_model", object()), ("draft_cache", object()),
                                       ("num_draft_tokens", 1), ("ngram_match_min", 2), ("dynamic_draft", True)])
def test_all_speculative_modes_fail_before_enqueue(rig, field, value):
    setattr(rig.generator, field, value)
    with pytest.raises(ValueError, match="Speculative"):
        run(rig)
    assert not rig.generator.created


@pytest.mark.parametrize("bad", ["resident", "tables", "handles", "pending", "held"])
def test_ple_streaming_and_quiescence_are_required_without_draining(rig, bad):
    table = rig.generator.model.ngram
    if bad == "resident":
        table.mode = "trellis_ram"
    elif bad == "tables":
        table.tables = [torch.ones(1)]
    elif bad == "handles":
        table.handles = []
    elif bad == "pending":
        table._pending = [object()]
    else:
        table._pins = [SimpleNamespace(held=True)]
    with pytest.raises(ValueError, match="PLE"):
        run(rig)
    assert not rig.generator.created


@pytest.mark.parametrize("change", ["slots", "batch", "capacity", "uninitialized", "tp", "unloaded", "architecture"])
def test_unsafe_loaded_capacity_or_model_is_refused(rig, change):
    gen = rig.generator
    if change == "slots":
        gen.cache.num_slots = 9
    elif change == "batch":
        gen.max_batch_size = 5
    elif change == "capacity":
        gen.max_total_tokens += 256
    elif change == "uninitialized":
        gen.cache.initialized = False
    elif change == "tp":
        gen.model.loaded_tp = True
    elif change == "unloaded":
        gen.model.output_device = None
    else:
        gen.model.config.architecture = "Other"
    with pytest.raises(ValueError):
        run(rig)
    assert not gen.created


def test_batch_must_fit_already_loaded_capacity(rig):
    rig.generator.max_batch_size = 1
    with pytest.raises(ValueError, match="batch exceeds"):
        run(rig, [request("a"), request("b")])


@pytest.mark.parametrize("slots", [5, 8])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_physical_slots_above_four_preserve_active_batch_cap(rig, slots, batch):
    rig.generator = Generator(slots=slots)
    rig.generator.max_batch_size = 4
    result = run(rig, [request(str(i)) for i in range(batch)])
    assert result["batch_size"] == batch and result["qualified"] is False
    assert rig.generator.cache.num_slots == slots and rig.generator.max_batch_size == 4
    assert_idle(rig)


def test_engine_capacity_allows_active_batch_above_four(rig):
    rig.generator = Generator(slots=8)
    rig.generator.max_batch_size = 5
    result = run(rig, [request(str(i)) for i in range(5)])
    assert result["batch_size"] == 5
    assert_idle(rig)


def test_stock_plus_one_page_margin_is_not_underestimated(rig):
    rig.generator = Generator(pages=1)
    with pytest.raises(ValueError, match="page budget"):
        run(rig, [request(prompt="A" * 255, max_new_tokens=1)])
    assert not rig.generator.created


def test_aggregate_capacity_does_not_assume_shared_prompt_pages(rig):
    rig.generator = Generator(pages=1)
    with pytest.raises(ValueError, match="page budget"):
        run(rig, [request("a"), request("b")])


def test_maximum_per_request_context_is_cpu_arithmetic_not_native_window_proof(rig):
    rig.generator = Generator(pages=1152)
    result = run(rig, [request(prompt="A" * (control.WINDOW - 1), max_new_tokens=1)])
    assert result["aggregate_context_budget"] == control.WINDOW and result["conservative_pages"] == 1025
    assert result["qualified"] is False and result["evidence_kind"] == "cpu_fake"
    assert_idle(rig)


@pytest.mark.parametrize("batch", [1, 2])
def test_per_request_context_over_limit_is_rejected_even_when_pages_fit(rig, batch):
    rig.generator = Generator(pages=1152)
    requests = [request(prompt="A" * (control.WINDOW - 1), max_new_tokens=2)]
    if batch == 2:
        requests.insert(0, request("small"))
    with pytest.raises(ValueError, match="Per-request.*262144"):
        run(rig, requests)
    assert not rig.generator.created and not rig.guard.started


@pytest.mark.parametrize("batch,total,pages", [(2, 140000, 1094), (4, 70000, 1096), (4, 73727, 1152)])
def test_aggregate_above_one_window_is_allowed_when_each_request_and_pages_fit(rig, batch, total, pages):
    rig.generator = Generator(slots=8, pages=1152)
    rig.generator.max_batch_size = 4
    requests = [request(str(i), prompt="A" * (total - 1), max_new_tokens=1) for i in range(batch)]
    result = run(rig, requests)
    assert result["aggregate_context_budget"] == batch * total > control.WINDOW
    assert result["conservative_pages"] == pages <= rig.generator.pagetable.max_pages
    assert result["qualified"] is False and result["evidence_kind"] == "cpu_fake"
    assert result["memory_fraction"] is None and result["peak_memory_qualified"] is False
    assert rig.admissions[0][0] >= (1 << 20) + batch * total * 192
    assert_idle(rig)


def test_per_request_valid_batch_still_refuses_aggregate_page_overflow(rig):
    rig.generator = Generator(pages=1152)
    requests = [request(str(i), prompt="A" * 73999, max_new_tokens=1) for i in range(4)]
    with pytest.raises(ValueError, match="simultaneous stock page budget"):
        run(rig, requests)
    assert not rig.generator.created and not rig.guard.started


@pytest.mark.parametrize("field,value", [("prompt_token_ids", ()), ("prompt_token_ids", (True,)),
                                       ("prompt_token_ids", (512,)), ("max_new_tokens", 0),
                                       ("temperature", float("nan")), ("temperature", -1.),
                                       ("top_p", 0.), ("top_k", -1), ("seed", -1),
                                       ("add_bos", 1), ("rendered_prompt", "")])
def test_invalid_request_is_rejected_before_enqueue(rig, field, value):
    with pytest.raises(ValueError):
        run(rig, [replace(request(), **{field: value})])
    assert not rig.generator.created


@pytest.mark.parametrize("count", [0])
def test_empty_control_wave_is_rejected(rig, count):
    with pytest.raises(ValueError, match="explicit requests"):
        control.run_native_generation_control(rig.generator, [request(str(i)) for i in range(count)],
                                             allow_generation=True, exclusive=True,
                                             additional_peak_bytes=1, estimate_basis="CPU fake")


@pytest.mark.parametrize("option,value", [("memory_fraction", float("nan")),
                                        ("additional_peak_bytes", 0), ("estimate_basis", ""),
                                        ("timeout_seconds", 0)])
def test_memory_and_deadline_guards_cannot_be_weakened(rig, option, value):
    with pytest.raises(ValueError):
        run(rig, **{option: value})
    assert not rig.generator.created and not rig.guard.started


def test_prompt_encoding_mismatch_closes_guard_without_enqueuing(rig):
    with pytest.raises(ValueError, match="prompt/token IDs"):
        run(rig, [request(prompt_token_ids=(42,))])
    assert not rig.generator.created
    assert_idle(rig)


def test_explicit_bos_ids_are_verified_not_silently_inserted(rig):
    result = run(rig, [request(add_bos=True, prompt_token_ids=(1, 66))])
    assert result["jobs"][0]["prompt_token_ids"] == [1, 66]


def test_partial_enqueue_failure_releases_only_owned_pending_jobs(rig):
    rig.generator.enqueue_fail_at = 2
    with pytest.raises(RuntimeError, match="enqueue failed"):
        run(rig, [request("a"), request("b")])
    assert_idle(rig)


@pytest.mark.parametrize("failure", ["iterate", "error_event", "guard", "cancel", "timeout"])
def test_failure_and_cancellation_release_all_owned_jobs(rig, monkeypatch, failure):
    cancel = Event()
    def callback(gen, events):
        if failure == "iterate":
            raise RuntimeError("synthetic iteration failure")
        if failure == "error_event":
            job = gen.active_jobs.pop(0)
            job.deallocate_pages()
            events.append({"job": job, "serial": job.serial_number, "stage": "error", "eos": True,
                           "error": RuntimeError("synthetic stock job failure")})
        if failure == "cancel":
            cancel.set()
    rig.generator.callback = callback
    if failure == "guard":
        rig.guard.fail_check = 4
    if failure == "timeout":
        counter = iter(range(100))
        monkeypatch.setattr(control.time, "monotonic", lambda: float(next(counter)))
    expected = {"iterate": RuntimeError, "error_event": RuntimeError, "guard": MemoryError,
                "cancel": InterruptedError, "timeout": TimeoutError}[failure]
    with pytest.raises(expected):
        run(rig, [request("a"), request("b")], cancel=cancel, timeout_seconds=4. if failure == "timeout" else 300.)
    assert_idle(rig)


def test_guard_start_or_admission_failure_does_not_leak(rig, monkeypatch):
    rig.guard.start_error = MemoryError("initial headroom failure")
    with pytest.raises(MemoryError):
        run(rig)
    assert_idle(rig)
    monkeypatch.setattr(control, "admit", lambda *args: (_ for _ in ()).throw(MemoryError("admission")))
    with pytest.raises(MemoryError, match="admission"):
        run(rig)
    assert not rig.generator.created


def test_foreign_queue_intrusion_is_left_untouched(rig):
    foreign = Job(torch.tensor([[66]]), sampler=ComboSampler(), identifier="foreign", max_new_tokens=3,
                  stop_conditions=[99])
    rig.generator.callback = lambda gen, events: gen.enqueue(foreign)
    with pytest.raises(ValueError, match="Foreign stock job"):
        run(rig)
    assert rig.generator.pending_jobs == [foreign]
    assert foreign not in rig.generator.cancelled and foreign.deallocations == 0
    assert not rig.generator.active_jobs and rig.guard.closed


def test_owner_takeover_during_iteration_never_releases_replacement_cache(rig):
    def takeover(gen, events):
        gen.cache.owner_serial += 1
    rig.generator.callback = takeover
    with pytest.raises(ValueError, match="no takeover") as error:
        run(rig)
    assert not rig.generator.cancelled and rig.guard.closed
    assert "cleanup" in error.value.__notes__[0]
    assert rig.generator not in control._owners


def test_cleanup_failure_is_reported_and_other_owned_jobs_are_still_released(rig):
    def fail(gen, events):
        gen.active_jobs[0].deallocation_error = RuntimeError("synthetic release failed")
        raise RuntimeError("original iteration failure")
    rig.generator.callback = fail
    with pytest.raises(RuntimeError, match="original iteration failure") as error:
        run(rig, [request("a"), request("b")])
    assert "release failed" in error.value.__notes__[0]
    assert rig.generator.created[1].recurrent_state is None
    assert rig.generator.created[1] not in rig.generator.active_jobs
    assert rig.guard.closed and rig.generator not in control._owners


@pytest.mark.parametrize("bad", ["serial", "requeue", "stream", "count", "eos", "cache", "missing_final"])
def test_contradictory_stock_events_never_produce_passing_receipts(rig, bad):
    rig.generator.scripts["one"] = [99]
    def mutate(events):
        final = events[-1]
        if bad == "serial":
            final["serial"] += 1
        elif bad == "requeue":
            final["requeue"] = True
        elif bad == "stream":
            final["token_ids"] = torch.tensor([[88]])
        elif bad == "count":
            final["new_tokens"] += 1
        elif bad == "eos":
            final["eos_triggering_token_id"] = 98
        elif bad == "cache":
            final["cached_tokens"] = 10
        else:
            events.pop()
        return events
    rig.generator.mutate_events = mutate
    with pytest.raises(ValueError):
        run(rig)
    assert_idle(rig)


def test_nested_control_is_refused_without_altering_original_owner(rig):
    def nested(gen, events):
        with pytest.raises(ValueError, match="already running"):
            run(rig)
    rig.generator.callback = nested
    assert run(rig)["qualified"] is False
    assert_idle(rig)


def test_import_and_disabled_path_do_not_bootstrap_torch_or_native_engine():
    script = ("import sys; from fuzev_research.native_generation_control import run_native_generation_control as run; "
              "assert 'torch' not in sys.modules; assert 'exllamav3' not in sys.modules\n"
              "try: run(None, [], additional_peak_bytes=1, estimate_basis='disabled')\n"
              "except ValueError: pass\nelse: raise AssertionError('disabled control ran')\n"
              "assert 'torch' not in sys.modules; assert 'exllamav3' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=10)
