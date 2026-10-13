# Offline Native 4.05 Operator Entrypoint

The offline entrypoint constructs a research backend and service, runs supplied
procedural cases, then closes both resources and writes the result.

The original launcher and service configuration are not distributed. This
entrypoint is retained as research source; current serving uses the component
CLIs linked from the snapshot overview.

Only the acquired, pinned `native-4.05` model and frozen B8 geometry are
used in the lab. Its cache size, allocation fraction 0.85 and prefill
optimizations were recorded with the frozen feature identity. The existing strict
`ServiceConfig` and `QualificationConfig` schemas are used unchanged. Cases,
including image/full-window controls, must attest `operator-non-heldout` input.
No dataset is opened, no input is generated, and no model or head is fitted.

One backend and service are created on the calling thread. The existing
`run_qualification(service, cases, cpu_fake=False, allow_inference=True)` runs
synchronously and retains its accepted-AB, exact-state, source, page/slot,
image, frozen-head, and tight-tolerance checks. State/image/full-window options
are taken unchanged from the cases file. No missing check becomes a pass.

The returned report is written unchanged to an exclusively created output;
existing files (including symlinks) are never overwritten. The parent directory
must already exist. Failure before a report is returned can leave an empty
reserved output. Successfully created resources always close in order:
`service.close()`, then `backend.close()`, even if either close raises.
Close failures return nonzero without rewriting an already returned report.

Exit 0 means only `execution_status: passed`, never qualification:
`qualified: false` and incomplete checks remain intact. Execution, input, pin,
write, or cleanup failures return 1; argument errors return 2; an uncaught
keyboard interruption returns 130. CPU mock tests are not native evidence.

Historical result: **47 focused CPU tests passed**. The launcher's `--help`
check passed with engine imports blocked.

[Snapshot methods and environment](../README.md)
