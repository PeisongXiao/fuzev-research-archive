from types import SimpleNamespace

import numpy as np

from prefill_bench import comparison_measurement, prefill_measurement, seed_prefix


class Tokens:
    def __init__(self, values):
        self.values = np.asarray(values)

    def reshape(self, *shape):
        return self.values.reshape(*shape)

    def torch(self):
        return self.values

    def __getitem__(self, key):
        return Tokens(self.values[key])

    def clone(self):
        return Tokens(self.values.copy())

    view = reshape

    def numel(self):
        return self.values.size


def test_prefill_and_decode_golden_timings_use_accepted_tokens():
    ids = Tokens(np.arange(2048))
    for outputs, mtp in ((1, 0), (1024, 0), (1024, 3)):
        accepted = Tokens(np.concatenate([np.arange(2048), np.full(outputs, 100)]))
        job = SimpleNamespace(sequences=[SimpleNamespace(sequence_ids=accepted)],
            draft_stats=[(4, 3, 2), (7, 3, 1), (1024, 3, 1)] if mtp else [],
            accepted_draft_tokens=4 if mtp else 0, rejected_draft_tokens=5 if mtp else 0)
        terminal = {"job": job, "prompt_tokens": 2048, "cached_tokens": 0, "new_tokens": outputs,
                    "time_enqueued": 0.25, "time_prefill": 1., "time_generate": 25.825,
                    "eos_reason": "max_new_tokens"}
        timing = {"time_after_first_s": 25.575 if outputs > 1 else 0., "ttft_s": 1.5}
        row = comparison_measurement(ids, 27.075, terminal, outputs, mtp, timing)
        assert row["processed_tokens"] == 2047
        assert row["prefill_tok_per_s"] == 2047.
        assert row["native_prefill_ready_s"] == 1.25
        assert row["completion_tokens"] == outputs
        assert row["tokens_after_first"] == outputs - 1
        assert row["decode_tok_per_s"] == (40. if outputs > 1 else None)
        assert row["proposed_draft_tokens"] == (9 if mtp else 0)


def test_w4_w5_append_accounting_and_isolated_peak():
    prefix = 32768
    for append in (512, 8192):
        setup_terminal = {"prompt_tokens": prefix + 1, "cached_tokens": 0, "new_tokens": 1}
        setup = seed_prefix(None, Tokens(np.arange(prefix + append)), prefix,
            run_job=lambda generator, ids: (prefix + 1, setup_terminal))
        assert setup["processed_tokens"] == prefix
        assert setup["forwarded_tokens"] == prefix + 1
        terminal = {"prompt_tokens": prefix + append, "cached_tokens": prefix,
                    "time_prefill": (append - 1) / 100, "new_tokens": 1}
        peak = 100 + append * 3
        row = prefill_measurement(terminal, append, peak, chunk=8192,
                                  length=prefix + append, rep=1, offset=0)
        assert row["processed_tokens"] == append - 1
        assert row["tok_per_s"] == 100
        assert row["peak_gpu_bytes"] == peak
        assert row["wall_s"] == append
