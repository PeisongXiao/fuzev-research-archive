"""Torch-free checks of explicit and resolved storage policy; no models."""

from copy import deepcopy
from enum import Enum
import os
from types import SimpleNamespace as NS

import pytest

from fuzev_research.vllm_ple_storage import (EXL3_LOOKUP_OP, EXL3_TABLE_ENV,
                                   validate_ple_storage_config, validate_ple_storage_options)


def options(quantization="modelopt_mixed"):
    return {"quantization": quantization, "enforce_eager": True,
            "safetensors_load_strategy": "lazy", "engram_config": {"cpu_offload": False},
            "additional_config": {"fuzev": {"enabled": True,
                **({} if quantization == "exl3" else {"ple_storage": "mmap"})}}}


class GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3


def resolved(quantization="modelopt_mixed"):
    opts = options(quantization)
    return NS(model_config=NS(quantization=quantization, enforce_eager=True),
              load_config=NS(load_format="auto", safetensors_load_strategy="lazy"),
              parallel_config=NS(tensor_parallel_size=1, data_parallel_size=1),
              additional_config=opts["additional_config"], engram_config=NS(cpu_offload=False),
              compilation_config=NS(cudagraph_mode=GraphMode.NONE, splitting_ops=None))


def change(value, path, replacement, *, remove=False):
    for name in path[:-1]:
        value = value[name] if isinstance(value, dict) else getattr(value, name)
    name = path[-1]
    if isinstance(value, dict):
        if remove:
            value.pop(name)
        else:
            value[name] = replacement
    elif remove:
        delattr(value, name)
    else:
        setattr(value, name, replacement)


@pytest.mark.parametrize("quantization", [None, "modelopt", "modelopt_fp4", "modelopt_mixed", "compressed-tensors"])
@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_valid_raw_settings_are_not_rewritten(monkeypatch, quantization, load_format):
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    opts = options(quantization)
    opts["load_format"] = load_format
    before, environment = deepcopy(opts), dict(os.environ)
    validate_ple_storage_options(opts, quantization=quantization)
    validate_ple_storage_config(resolved(quantization))
    assert opts == before and dict(os.environ) == environment


@pytest.mark.parametrize("path,value", [
    (("additional_config",), None), (("additional_config",), []),
    (("additional_config", "fuzev"), None), (("additional_config", "fuzev"), "mmap"),
    (("additional_config", "fuzev", "ple_storage"), None),
    (("additional_config", "fuzev", "ple_storage"), "resident"),
    (("additional_config", "fuzev", "ple_storage"), "disk"),
    (("additional_config", "fuzev", "ple_storage"), []),
    (("engram_config",), None), (("engram_config",), {}),
    (("engram_config",), '{"cpu_offload": false}'),
    (("engram_config", "cpu_offload"), True), (("engram_config", "cpu_offload"), None),
    (("engram_config", "cpu_offload"), 0), (("engram_config", "cpu_offload"), "false"),
    (("enforce_eager",), False), (("enforce_eager",), None),
    (("enforce_eager",), 1), (("enforce_eager",), "true"),
    (("safetensors_load_strategy",), None), (("safetensors_load_strategy",), "auto"),
    (("safetensors_load_strategy",), "eager"), (("safetensors_load_strategy",), "prefetch"),
    (("load_format",), "pt"), (("load_format",), "fastsafetensors"),
    (("tensor_parallel_size",), 2), (("tensor_parallel_size",), True),
    (("data_parallel_size",), 2), (("data_parallel_size",), "1"),
])
def test_invalid_raw_options_fail_closed(path, value):
    opts = options()
    change(opts, path, value)
    before = deepcopy(opts)
    with pytest.raises(ValueError):
        validate_ple_storage_options(opts, quantization="modelopt_mixed")
    assert opts == before


@pytest.mark.parametrize("path", [
    ("additional_config",), ("additional_config", "fuzev"),
    ("additional_config", "fuzev", "ple_storage"), ("engram_config",),
    ("engram_config", "cpu_offload"), ("enforce_eager",), ("safetensors_load_strategy",),
])
def test_missing_explicit_raw_settings_are_not_admitted_by_environment(monkeypatch, path):
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "0")
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options()
    opts["cpu_offload_gb"] = 0
    change(opts, path, None, remove=True)
    with pytest.raises(ValueError):
        validate_ple_storage_options(opts, quantization="modelopt_mixed")


@pytest.mark.parametrize("raw_selector", [None, "resident"])
def test_packed_disk_is_independent_of_raw_selector(monkeypatch, raw_selector):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options("exl3")
    if raw_selector is not None:
        opts["additional_config"]["fuzev"]["ple_storage"] = raw_selector
    before = deepcopy(opts)
    validate_ple_storage_options(opts, quantization="exl3")
    validate_ple_storage_config(resolved("exl3"))
    assert opts == before


@pytest.mark.parametrize("environment", [None, "", "resident", "mmap", "DISK", " disk "])
def test_packed_requires_explicit_disk_environment(monkeypatch, environment):
    monkeypatch.delenv(EXL3_TABLE_ENV, raising=False)
    if environment is not None:
        monkeypatch.setenv(EXL3_TABLE_ENV, environment)
    with pytest.raises(ValueError, match="requires explicit VLLM_EXL3_NGRAM_TABLE=disk"):
        validate_ple_storage_options(options("exl3"), quantization="exl3")


@pytest.mark.parametrize("environment", [None, "resident", "disk"])
@pytest.mark.parametrize("raw_selector", [None, "mmap", "disk", "unknown", []])
def test_packed_never_uses_raw_mmap_even_with_disk_environment(monkeypatch, environment, raw_selector):
    monkeypatch.delenv(EXL3_TABLE_ENV, raising=False)
    if environment is not None:
        monkeypatch.setenv(EXL3_TABLE_ENV, environment)
    opts = options("exl3")
    opts["additional_config"]["fuzev"]["ple_storage"] = raw_selector
    with pytest.raises(ValueError, match="own disk adapter"):
        validate_ple_storage_options(opts, quantization="exl3")


@pytest.mark.parametrize("quantization", ["modelopt_mixed", "exl3"])
@pytest.mark.parametrize("path,value", [
    (("engram_config",), None), (("engram_config", "cpu_offload"), True),
    (("engram_config", "cpu_offload"), None), (("engram_config", "cpu_offload"), 0),
    (("load_config", "safetensors_load_strategy"), "eager"),
    (("load_config", "load_format"), "pt"),
    (("parallel_config", "tensor_parallel_size"), 2),
    (("parallel_config", "data_parallel_size"), 2),
    (("additional_config",), None),
])
def test_resolved_config_cannot_hide_unsafe_storage(monkeypatch, quantization, path, value):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    cfg = resolved(quantization)
    change(cfg, path, value)
    with pytest.raises(ValueError):
        validate_ple_storage_config(cfg)


def test_resolved_raw_eager_and_selector_are_required():
    cfg = resolved()
    cfg.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce_eager"):
        validate_ple_storage_config(cfg)
    cfg.model_config.enforce_eager = True
    cfg.additional_config["fuzev"]["ple_storage"] = "resident"
    with pytest.raises(ValueError, match="ple_storage"):
        validate_ple_storage_config(cfg)


def test_environment_is_rechecked_at_resolved_admission(monkeypatch):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    validate_ple_storage_options(options("exl3"), quantization="exl3")
    monkeypatch.setenv(EXL3_TABLE_ENV, "resident")
    with pytest.raises(ValueError, match="default is resident"):
        validate_ple_storage_config(resolved("exl3"))


@pytest.mark.parametrize("mode,ops,accepted", [
    (GraphMode.NONE, None, True), (GraphMode.PIECEWISE, None, True),
    (GraphMode.PIECEWISE, [], True), (GraphMode.PIECEWISE, [EXL3_LOOKUP_OP], True),
    (GraphMode.PIECEWISE, ["vllm::attention", EXL3_LOOKUP_OP], True),
    (GraphMode.NONE, [EXL3_LOOKUP_OP], True),
    (GraphMode.FULL, [EXL3_LOOKUP_OP], False),
    (GraphMode.FULL_AND_PIECEWISE, [EXL3_LOOKUP_OP], False),
    (GraphMode.PIECEWISE, ["vllm::attention"], False),
    (GraphMode.NONE, ["vllm::attention"], False),
    ("FULL", None, False), ("PIECEWISE", [EXL3_LOOKUP_OP], True),
])
def test_packed_graph_gate_matches_plugin_conditions(monkeypatch, mode, ops, accepted):
    monkeypatch.setenv(EXL3_TABLE_ENV, "disk")
    opts = options("exl3")
    opts["enforce_eager"] = False
    validate_ple_storage_options(opts, quantization="exl3")
    cfg = resolved("exl3")
    cfg.model_config.enforce_eager = False
    cfg.compilation_config.cudagraph_mode, cfg.compilation_config.splitting_ops = mode, ops
    if accepted:
        validate_ple_storage_config(cfg)
    else:
        with pytest.raises(ValueError, match="outside capture"):
            validate_ple_storage_config(cfg)
