"""Local disk-PLE admission only; no engine imports or flag rewriting."""

import os


EXL3_TABLE_ENV = "VLLM_EXL3_NGRAM_TABLE"
EXL3_LOOKUP_OP = "vllm::exl3_ngram_lookup_out"


def validate_ple_storage_options(options, *, quantization):
    """Require explicit nonresident PLE policy, not environment/default offload."""
    if not isinstance(options, dict):
        raise ValueError("Disk PLE requires an engine-options dictionary")
    additional = options.get("additional_config", {})
    if not isinstance(additional, dict) or not isinstance(additional.get("fuzev", {}), dict):
        raise ValueError("Disk PLE requires dictionary additional_config.fuzev options")
    storage = additional.get("fuzev", {}).get("ple_storage", "resident")
    engram = options.get("engram_config")
    if not isinstance(engram, dict) or engram.get("cpu_offload") is not False:
        raise ValueError("Disk PLE requires explicit engram_config.cpu_offload=false; no pinned-host offload")
    if options.get("safetensors_load_strategy") != "lazy":
        raise ValueError("Disk PLE requires explicit safetensors_load_strategy='lazy'")
    if options.get("load_format", "auto") not in ("auto", "safetensors"):
        raise ValueError("Disk PLE requires the standard auto/safetensors loader")
    for key in ("tensor_parallel_size", "data_parallel_size"):
        value = options.get(key, 1)
        if type(value) is not int or value != 1:
            raise ValueError("Disk PLE requires TP/DP=1")
    if quantization == "exl3":
        if storage != "resident":
            raise ValueError("Packed EXL3 PLE must use its own disk adapter, not raw ple_storage='mmap'")
        if os.environ.get(EXL3_TABLE_ENV) != "disk":
            raise ValueError(f"Packed EXL3 PLE requires explicit {EXL3_TABLE_ENV}=disk; default is resident")
    else:
        if storage != "mmap":
            raise ValueError("Raw PLE requires explicit additional_config.fuzev.ple_storage='mmap'")
        if options.get("enforce_eager") is not True:
            raise ValueError("Raw mmap PLE requires explicit enforce_eager=true")


def validate_ple_storage_config(config):
    """Recheck resolved metadata before AsyncLLM or any weights are constructed."""
    model, load, parallel = config.model_config, config.load_config, config.parallel_config
    engram = getattr(config, "engram_config", None)
    validate_ple_storage_options({
        "additional_config": config.additional_config,
        "engram_config": {"cpu_offload": getattr(engram, "cpu_offload", None)},
        "safetensors_load_strategy": load.safetensors_load_strategy,
        "load_format": load.load_format,
        "enforce_eager": model.enforce_eager,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "data_parallel_size": parallel.data_parallel_size,
    }, quantization=model.quantization)
    if model.quantization == "exl3":
        # Match the plugin's host-gather graph gate without importing its model code.
        compilation = config.compilation_config
        mode = compilation.cudagraph_mode
        name = getattr(mode, "name", str(mode))
        ops = compilation.splitting_ops or []
        if "FULL" in name or (ops and EXL3_LOOKUP_OP not in ops):
            raise ValueError("Packed disk PLE requires eager execution or non-FULL graphs with "
                             f"{EXL3_LOOKUP_OP} outside capture")
