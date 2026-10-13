# Research E2E resumability

Both `scripts/kev_e2e_client.py` and `scripts/kev_vllm_e2e_client.py` accept
`--resume-from <prior-attempt-directory>` alongside a **new** `--output-dir`.
Neither the parent journal nor its failure evidence is modified or removed.

The runner reads the consecutive complete HTTP-200 results from the parent,
checks each request/body/index/gold against the unchanged frozen panel, and
revalidates every response using the normal schema, head and worker-provenance
checks. It reconstructs aggregate metrics and official-order Kev rows, then
starts inference at the first unfinished case. A selection that disagrees with
gold is still a completed case, not a reason to rerun it. An explicit failed
request or torn final line is not counted as a completed result.

Compatibility requires identical frozen inputs, model/cell, head, plan, quality
record, raw-T1 policy and **loaded serving provenance**. Changes to the offline
runner alone are admitted, with both versions' pins verified and retained.
Relocated adapter/preparation script pins are matched by their unchanged frozen
hashes; duplicate identical copies are allowed, conflicting copies are not.
No dataset, model, head, serving-source or capture path is normalized away.
Engine/model/serving-source changes are not admitted as a resumption. In
particular, the old NVIDIA unpublished lab commit `46fd72f` run cannot be mixed with the repaired
unpublished lab commit `5398bfc` run: output equivalence has not been established.

This is runner/client recovery against the same loaded server. An unchanged
server binary restarted into a new capture can still have a different PID,
allocation record or capture checksum and will be rejected. It is not silently
treated as the same live provenance. Supporting segmented server sessions would
require a separately reviewed report/comparator contract; that is not enabled.

New attempts hold an advisory exclusive `runner.lock` for their entire lifetime,
including finalization. A live parent cannot be resumed. Attempts made before resume support have
no lock and require their explicit terminal `failure.json`; a possibly live or
unproven crashed legacy attempt is not admitted automatically. New journals
flush and fsync each complete record. Metadata and directory entries are also
fsynced. The parent's hashes are checked again after recovery.
The fully revalidated durable prefix may be one record ahead of a failure
counter if an interrupt landed immediately after fsync and before accounting;
larger counter disagreements are rejected.

`resume.json` identifies the unchanged parent files, reused count, current
session's initial health and any ignored incomplete tail. The consolidated
journal keeps the original first health and current final health, exactly one
decision per index, and the original response bodies and request timings.
Summary `elapsed_seconds` covers only the current session; `resumed_requests`
and `submitted_requests` make that explicit. Do not interpret it as full-panel
throughput. Failed attempts stay separately available for audit.

There are no automatic HTTP retries, population limits, head tuning, calibration
changes, quality waivers or production changes. An existing output directory is
never overwritten.

Tests use synthetic full 5,075-case panels with HTTP and state fixtures only.
They cover interrupted and repeated recovery, identical fresh/resumed metrics
and Kev rows, prefix preservation, locks, torn tails, legacy failures, changed
runtime/input/head/policy and malformed saved responses. No LLM is loaded.

The reviewed implementation passed 29 recovery tests and the combined 120
tests plus 42 subtests with GPU hidden. A separate read-only metadata check
loaded the real full frozen panel and that run's recorded initial health from disk:
all 5,075 inputs, duplicate adapter paths and the unchanged serving provenance
admit the new offline runner. No live HTTP request or model execution was used.
The final parent-hash check runs inside failure handling before any completion
artifact or success status is written; the output lease stays held throughout.

[Snapshot methods and environment](../README.md)
