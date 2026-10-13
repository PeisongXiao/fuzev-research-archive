# Historical Results

## Superseded vLLM Generation Checks

Checkpoint: [NVIDIA Qwen3.8-Flash-Next NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47).
Platform: NVIDIA GB10, ARM64. Date: 2026-10-12.

These pre-release vLLM measurements predate the graph-replay and cache
slot-mapping repairs. Outputs contain repeated punctuation or token-ID-zero
text; HTTP completion, overlapping lifetimes and cleanup do not show healthy
generation. Corrected measurements are in
[the integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md).

| Output Budget | Output Tokens | HTTP Wall (s) | Accepted / Proposed Draft Tokens |
| --- | ---: | ---: | ---: |
| 256 | 256 | 24.71 | 0/765 |
| 512 | 512 | 43.41 | 0/1533 |

The superseded mixed-request check uses a 419-token prompt and a 64-token output:

- Three decisions arrive at 0.750709, 1.000194 and 1.250376 seconds, after the
  0.491831-second first-token bound.
- Generation HTTP lifetime is 6.949798 seconds. Recorded prefill is 0.398132
  seconds and decode 6.457967 seconds.
- The engine reports 0 accepted draft tokens out of 189 proposed.

Malformed outputs make these timings unusable for speed comparison.

## Initial Long-Prefill Failure

The first 8,192-token prefill (512-token output budget) ends in a fatal CUDA
illegal-memory-access / CUBLAS internal error. The engine container still exits
with code 0 and is not OOM-killed. That attempt never reaches the 32,768-token probe.

The completed generation checks use:

- V1 model runner and MTP with three draft tokens
- FP8 E4M3 KV, piecewise graphs and memory-mapped embeddings
- Context 262,144 and GPU memory fraction 0.85

Later exact-length 8,192/32,768-prefill measurements are in
[the current MTP aggregates](https://github.com/PeisongXiao/fuzev/blob/main/docs/mtp-diagnostics.json).

## Pre-Fix vLLM Short-Workload Performance (Superseded)

These pre-release plugin and engine measurements also predate the graph-replay
and cache slot-mapping fixes.
The ExLlamaV3 rows are in [the performance panel](https://github.com/PeisongXiao/fuzev/blob/main/docs/performance.md).

- **Environment:** NVIDIA GB10, ARM64, Python 3.12, PyTorch 2.13.0+cu130 (CUDA 13.0), 2026-10-12.
- **Settings:** MTP off, KV `auto`, GPU memory fraction 0.75. NVFP4 embeddings
  are memory-mapped; EXL3 embeddings are resident.
- **Workload:** a short Python-file chat prompt with thinking disabled, a
  two-option Choice, and sequential generation-to-decision mixed requests.
- **Runs:** one warmup and three measured runs per mode and stream count, with
  a 32-token generation budget.

Wall values are pipeline means; brackets give the range of the three run means.

| Configuration | Streams | Gen Prefill tok/s | Gen Decode tok/s | Gen Wall s [Range] | Decision Wall s [Range] | Mixed Wall s [Range] |
| --- | ---: | ---: | ---: | --- | --- | --- |
| vLLM NVIDIA NVFP4 | 1 | 792.07 | 16.50 | 2.48 [2.47, 2.50] | 1.01 [0.98, 1.04] | 3.24 [3.20, 3.26] |
| vLLM NVIDIA NVFP4 | 4 | 534.64 | 13.79 | 3.66 [3.45, 3.78] | 3.06 [2.85, 3.29] | 5.80 [5.75, 5.83] |
| vLLM RadixArk NVFP4 | 1 | 799.24 | 16.59 | 2.46 [2.43, 2.48] | 0.96 [0.95, 0.99] | 3.24 [3.17, 3.36] |
| vLLM RadixArk NVFP4 | 4 | 526.34 | 14.32 | 3.59 [3.46, 3.67] | 2.85 [2.78, 2.90] | 5.93 [5.81, 6.09] |
| vLLM EXL3 4.05 bpw | 1 | 813.67 | 15.88 | 2.54 [2.46, 2.70] | 1.51 [1.49, 1.53] | 3.68 [3.67, 3.70] |
| vLLM EXL3 4.05 bpw | 4 | 503.95 | 14.41 | 3.63 [3.61, 3.65] | 4.34 [4.13, 4.46] | 7.97 [7.78, 8.08] |

Source artifacts: `vllm-nvidia-small-performance-v2.json`,
`vllm-radixark-small-performance-v2.json` and
`vllm-exl3-small-performance-v2.json`.

### Full Statistics

Full-precision aggregates, observed token counts, per-run timing samples,
checkpoint revisions and recorded settings:

```json
{
  "schema": "fuzev-small-performance-aggregates-v3",
  "date": "2026-10-12",
  "engine_builds": "Pre-release plugin and engine builds; optimized assets are enabled for native measurements. Published serving builds have no end-to-end quality/performance benchmark.",
  "recorded_settings": {
    "vllm_nvfp4": {
      "context_tokens": 262144,
      "kv_dtype": "auto",
      "gpu_memory_fraction": 0.75,
      "embedding_storage": "memory-mapped",
      "mtp": false
    },
    "vllm_exl3": {
      "context_tokens": 262144,
      "kv_dtype": "auto",
      "gpu_memory_fraction": 0.75,
      "embedding_storage": "resident",
      "mtp": false
    }
  },
  "head_sha256": "f7bc42e780c6da72aa675ae9ddde5e4b4f4320da6b0676946ca7b3e1238ac33c",
  "quality_scored": false,
  "policy": {
    "workloads": 1,
    "prompt": "short Python file; actual engine token counts recorded",
    "generation_input": "chat messages through the fixed template, thinking disabled",
    "modes": [
      "generation",
      "decision",
      "mixed"
    ],
    "streams": [
      1,
      4
    ],
    "requested_generation_tokens": 32,
    "sampling": {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 20,
      "seed": 17,
      "min_p": 0,
      "presence_penalty": 0,
      "repetition_penalty": 1
    },
    "head_calibration": "raw, temperature 1",
    "cache_policy": "Unique leading prefix per pipeline and run; mixed mode reuses the accepted generation prefix for its decision.",
    "quality_scored": false,
    "configuration_ids": [
      "vllm-nvidia",
      "vllm-radixark",
      "vllm-exl3-4.05"
    ],
    "warmup_runs": 1,
    "measured_runs": 3,
    "decision": {
      "primitive": "choice",
      "candidates": 2,
      "question": "Which description matches the file?",
      "options": [
        "A Python source file",
        "A cooking recipe"
      ]
    }
  },
  "records": [
    {
      "configuration_id": "vllm-nvidia",
      "name": "vLLM NVIDIA NVFP4",
      "backbone_repository": "nvidia/Qwen3.8-Flash-Next-NVFP4",
      "backbone_revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
      "status": "completed",
      "summary": [
        {
          "mode": "generation",
          "streams": 1,
          "mean_pipeline_seconds": 2.482173340356288,
          "mean_wave_seconds": 2.4833027200074866,
          "mean_prefill_tokens_per_second": 792.0685217991252,
          "mean_decode_tokens_per_second": 16.495218509428856,
          "observed_generation_prompt_tokens": [
            442,
            444,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            2.474538578069769,
            2.496479246998206
          ]
        },
        {
          "mode": "generation",
          "streams": 4,
          "mean_pipeline_seconds": 3.6583634891915913,
          "mean_wave_seconds": 3.6757799660554156,
          "mean_prefill_tokens_per_second": 534.6397946921526,
          "mean_decode_tokens_per_second": 13.788584247264774,
          "observed_generation_prompt_tokens": [
            441,
            442,
            443,
            445,
            446,
            447,
            449,
            451
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.450391688296804,
            3.780093285284238
          ]
        },
        {
          "mode": "decision",
          "streams": 1,
          "mean_pipeline_seconds": 1.0106939046721284,
          "mean_wave_seconds": 1.0112443966791034,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            0.9760219840100035,
            1.043821464991197
          ]
        },
        {
          "mode": "decision",
          "streams": 4,
          "mean_pipeline_seconds": 3.055891532509122,
          "mean_wave_seconds": 3.503520499682054,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            2.8514835252426565,
            3.2928361460217275
          ]
        },
        {
          "mode": "mixed",
          "streams": 1,
          "mean_pipeline_seconds": 3.237730313984988,
          "mean_wave_seconds": 3.23876782564912,
          "mean_prefill_tokens_per_second": 897.8519317966907,
          "mean_decode_tokens_per_second": 16.395576719733018,
          "observed_generation_prompt_tokens": [
            443,
            448
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.201610396965407,
            3.260588133940473
          ]
        },
        {
          "mode": "mixed",
          "streams": 4,
          "mean_pipeline_seconds": 5.800953631104979,
          "mean_wave_seconds": 6.348811362365571,
          "mean_prefill_tokens_per_second": 619.2700341183742,
          "mean_decode_tokens_per_second": 13.852409890547385,
          "observed_generation_prompt_tokens": [
            442,
            443,
            444,
            445,
            446,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            5.752311770746019,
            5.834660420776345
          ]
        }
      ],
      "measured_runs": [
        {
          "mode": "generation",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 2.4765825299546123,
          "mean_pipeline_seconds": 2.475502196000889
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 2.4755997590254992,
          "mean_pipeline_seconds": 2.474538578069769
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 2.4977258710423484,
          "mean_pipeline_seconds": 2.496479246998206
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 3.7975241229869425,
          "mean_pipeline_seconds": 3.780093285284238
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 3.46850505808834,
          "mean_pipeline_seconds": 3.450391688296804
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 3.7613107170909643,
          "mean_pipeline_seconds": 3.744605493993731
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 1.0444003840675578,
          "mean_pipeline_seconds": 1.043821464991197
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 1.0128370400052518,
          "mean_pipeline_seconds": 1.0122382650151849
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 0.9764957659645006,
          "mean_pipeline_seconds": 0.9760219840100035
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 3.757382686017081,
          "mean_pipeline_seconds": 3.2928361460217275
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 3.2758274560328573,
          "mean_pipeline_seconds": 2.8514835252426565
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 3.4773513569962233,
          "mean_pipeline_seconds": 3.023354926262982
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 3.261522514047101,
          "mean_pipeline_seconds": 3.260588133940473
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 3.251997703919187,
          "mean_pipeline_seconds": 3.250992411049083
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 3.2027832589810714,
          "mean_pipeline_seconds": 3.201610396965407
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 6.466777150053531,
          "mean_pipeline_seconds": 5.752311770746019
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 6.310985242947936,
          "mean_pipeline_seconds": 5.834660420776345
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 6.2686716940952465,
          "mean_pipeline_seconds": 5.815888701792574
        }
      ],
      "measured_settings": {
        "context_tokens": 262144,
        "kv_dtype": "auto",
        "gpu_memory_fraction": 0.75,
        "embedding_storage": "memory-mapped",
        "mtp": false
      }
    },
    {
      "configuration_id": "vllm-radixark",
      "name": "vLLM RadixArk NVFP4",
      "backbone_repository": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
      "backbone_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
      "status": "completed",
      "summary": [
        {
          "mode": "generation",
          "streams": 1,
          "mean_pipeline_seconds": 2.4551129286410287,
          "mean_wave_seconds": 2.4561394500002884,
          "mean_prefill_tokens_per_second": 799.2379010962428,
          "mean_decode_tokens_per_second": 16.587989934989217,
          "observed_generation_prompt_tokens": [
            442,
            444,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            2.432983665028587,
            2.475321057951078
          ]
        },
        {
          "mode": "generation",
          "streams": 4,
          "mean_pipeline_seconds": 3.588005639894012,
          "mean_wave_seconds": 3.6058280393481255,
          "mean_prefill_tokens_per_second": 526.3399772107098,
          "mean_decode_tokens_per_second": 14.316725008957453,
          "observed_generation_prompt_tokens": [
            441,
            442,
            443,
            445,
            446,
            447,
            449,
            451
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.4556128194672056,
            3.666107266995823
          ]
        },
        {
          "mode": "decision",
          "streams": 1,
          "mean_pipeline_seconds": 0.9649602703284472,
          "mean_wave_seconds": 0.9655313153828805,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            0.9519011339871213,
            0.9864064019639045
          ]
        },
        {
          "mode": "decision",
          "streams": 4,
          "mean_pipeline_seconds": 2.8530307385954075,
          "mean_wave_seconds": 3.286756562961576,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            2.781421509978827,
            2.8987664867891
          ]
        },
        {
          "mode": "mixed",
          "streams": 1,
          "mean_pipeline_seconds": 3.235493132340101,
          "mean_wave_seconds": 3.236654299971027,
          "mean_prefill_tokens_per_second": 998.6945339241901,
          "mean_decode_tokens_per_second": 16.575383451642423,
          "observed_generation_prompt_tokens": [
            443,
            448
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.167205263976939,
            3.3586286189965904
          ]
        },
        {
          "mode": "mixed",
          "streams": 4,
          "mean_pipeline_seconds": 5.93218524682258,
          "mean_wave_seconds": 6.3808708843231825,
          "mean_prefill_tokens_per_second": 591.2142613962885,
          "mean_decode_tokens_per_second": 13.927883105653256,
          "observed_generation_prompt_tokens": [
            442,
            443,
            444,
            445,
            446,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            5.811634693003725,
            6.093133599235443
          ]
        }
      ],
      "measured_runs": [
        {
          "mode": "generation",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 2.476338666980155,
          "mean_pipeline_seconds": 2.475321057951078
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 2.434230940998532,
          "mean_pipeline_seconds": 2.432983665028587
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 2.457848742022179,
          "mean_pipeline_seconds": 2.4570340629434213
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 3.6605761750834063,
          "mean_pipeline_seconds": 3.6422968332190067
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 3.684771285043098,
          "mean_pipeline_seconds": 3.666107266995823
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 3.472136657917872,
          "mean_pipeline_seconds": 3.4556128194672056
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 0.9573649610392749,
          "mean_pipeline_seconds": 0.9565732750343159
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 0.9867995900567621,
          "mean_pipeline_seconds": 0.9864064019639045
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 0.9524293950526044,
          "mean_pipeline_seconds": 0.9519011339871213
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 3.3082674129400402,
          "mean_pipeline_seconds": 2.8789042190182954
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 3.2269794299500063,
          "mean_pipeline_seconds": 2.781421509978827
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 3.325022845994681,
          "mean_pipeline_seconds": 2.8987664867891
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 3.359842213918455,
          "mean_pipeline_seconds": 3.3586286189965904
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 3.1683022499782965,
          "mean_pipeline_seconds": 3.167205263976939
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 3.1818184360163286,
          "mean_pipeline_seconds": 3.1806455140467733
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 6.5682653120020404,
          "mean_pipeline_seconds": 6.093133599235443
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 6.244235267979093,
          "mean_pipeline_seconds": 5.811634693003725
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 6.330112072988413,
          "mean_pipeline_seconds": 5.89178744822857
        }
      ],
      "measured_settings": {
        "context_tokens": 262144,
        "kv_dtype": "auto",
        "gpu_memory_fraction": 0.75,
        "embedding_storage": "memory-mapped",
        "mtp": false
      }
    },
    {
      "configuration_id": "vllm-exl3-4.05",
      "name": "vLLM EXL3 4.05 bpw",
      "backbone_repository": "turboderp/Qwen3.8-Flash-Next-exl3",
      "backbone_revision": "55a732e0c4c3d4614bc42b68493bb930d9b02c0a",
      "status": "completed",
      "summary": [
        {
          "mode": "generation",
          "streams": 1,
          "mean_pipeline_seconds": 2.5411132563215992,
          "mean_wave_seconds": 2.541956963017583,
          "mean_prefill_tokens_per_second": 813.6667252005539,
          "mean_decode_tokens_per_second": 15.875088764594054,
          "observed_generation_prompt_tokens": [
            442,
            444,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            2.457404461922124,
            2.700172294047661
          ]
        },
        {
          "mode": "generation",
          "streams": 4,
          "mean_pipeline_seconds": 3.629359377247359,
          "mean_wave_seconds": 3.6486885553846755,
          "mean_prefill_tokens_per_second": 503.95100321206263,
          "mean_decode_tokens_per_second": 14.408353637275479,
          "observed_generation_prompt_tokens": [
            441,
            442,
            443,
            445,
            446,
            447,
            449,
            451
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.613241209764965,
            3.649165967479348
          ]
        },
        {
          "mode": "decision",
          "streams": 1,
          "mean_pipeline_seconds": 1.5092462596561138,
          "mean_wave_seconds": 1.5098955063537385,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            1.488569003995508,
            1.5301731110084802
          ]
        },
        {
          "mode": "decision",
          "streams": 4,
          "mean_pipeline_seconds": 4.343051193582748,
          "mean_wave_seconds": 5.341695525373022,
          "mean_prefill_tokens_per_second": null,
          "mean_decode_tokens_per_second": null,
          "observed_generation_prompt_tokens": [],
          "observed_generation_completion_tokens": [],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            4.133204138255678,
            4.456753904989455
          ]
        },
        {
          "mode": "mixed",
          "streams": 1,
          "mean_pipeline_seconds": 3.6807629596441984,
          "mean_wave_seconds": 3.6818842363233366,
          "mean_prefill_tokens_per_second": 846.8881208805644,
          "mean_decode_tokens_per_second": 18.73521292140957,
          "observed_generation_prompt_tokens": [
            443,
            448
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            3.6678327929694206,
            3.6968281869776547
          ]
        },
        {
          "mode": "mixed",
          "streams": 4,
          "mean_pipeline_seconds": 7.969301688242315,
          "mean_wave_seconds": 9.012933672017729,
          "mean_prefill_tokens_per_second": 511.7549627701093,
          "mean_decode_tokens_per_second": 15.00364467791093,
          "observed_generation_prompt_tokens": [
            442,
            443,
            444,
            445,
            446,
            447
          ],
          "observed_generation_completion_tokens": [
            32
          ],
          "scope": "per-request engine phases; wave wall is separate and includes prefix release",
          "pipeline_run_mean_range_seconds": [
            7.779046067473246,
            8.083211521967314
          ]
        }
      ],
      "measured_runs": [
        {
          "mode": "generation",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 2.4666914669796824,
          "mean_pipeline_seconds": 2.465763012995012
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 2.458184211049229,
          "mean_pipeline_seconds": 2.457404461922124
        },
        {
          "mode": "generation",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 2.7009952110238373,
          "mean_pipeline_seconds": 2.700172294047661
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 3.6687230430543423,
          "mean_pipeline_seconds": 3.649165967479348
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 3.644710983033292,
          "mean_pipeline_seconds": 3.625670954497764
        },
        {
          "mode": "generation",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 3.6326316400663927,
          "mean_pipeline_seconds": 3.613241209764965
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 1.5309265720425174,
          "mean_pipeline_seconds": 1.5301731110084802
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 1.489201807999052,
          "mean_pipeline_seconds": 1.488569003995508
        },
        {
          "mode": "decision",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 1.5095581390196458,
          "mean_pipeline_seconds": 1.5089966639643535
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 5.2999623629730195,
          "mean_pipeline_seconds": 4.456753904989455
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 5.451101168058813,
          "mean_pipeline_seconds": 4.133204138255678
        },
        {
          "mode": "decision",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 5.274023045087233,
          "mean_pipeline_seconds": 4.43919553750311
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 1,
          "run_wall_seconds": 3.6978867220459506,
          "mean_pipeline_seconds": 3.6968281869776547
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 2,
          "run_wall_seconds": 3.6788253309205174,
          "mean_pipeline_seconds": 3.67762789898552
        },
        {
          "mode": "mixed",
          "streams": 1,
          "run": 3,
          "run_wall_seconds": 3.6689406560035422,
          "mean_pipeline_seconds": 3.6678327929694206
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 1,
          "run_wall_seconds": 8.943967203027569,
          "mean_pipeline_seconds": 8.045647475286387
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 2,
          "run_wall_seconds": 8.984541423036717,
          "mean_pipeline_seconds": 8.083211521967314
        },
        {
          "mode": "mixed",
          "streams": 4,
          "run": 3,
          "run_wall_seconds": 9.1102923899889,
          "mean_pipeline_seconds": 7.779046067473246
        }
      ],
      "measured_settings": {
        "context_tokens": 262144,
        "kv_dtype": "auto",
        "gpu_memory_fraction": 0.75,
        "embedding_storage": "resident",
        "mtp": false
      }
    }
  ],
  "measurement_environment": {
    "hardware": "NVIDIA GB10 (DGX Spark)",
    "architecture": "ARM64",
    "python": "3.12",
    "pytorch": "2.13.0+cu130",
    "cuda": "13.0",
    "engine_builds": "Pre-release research or plugin/fork builds"
  }
}
```
