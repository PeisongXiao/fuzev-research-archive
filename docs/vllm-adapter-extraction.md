# vLLM Adapter Method

The [vLLM plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin) separates HTTP serving and
decision orchestration from the shared head and engine fork. This note records
the feature-readout method retained from the research implementation.

## Readout

- Each question has one query and independent candidate continuations from the
  same prefix.
- The engine returns its final hidden state before vocabulary projection.
- Query label logits use the model's existing LM head; candidate-only reads skip
  vocabulary projection.
- The adapter copies features before the engine reuses scratch storage.
- The shared head scores each question afresh.
- Calibration is selected at service start: raw temperature 1 by default, or the
  bundle's per-primitive temperatures.
- Retained prefixes preserve accepted token IDs for later input; they do not
  store decision scores.

## Integration

- The service attaches to the application's running AsyncLLM.
- Closing the attachment drains its requests and closes its head; the supplied
  engine stays available to the application.
- The component CLI manages the engine it constructs.
- `POST /v1/systemone` and the optional lower-level `/v1/decisions` route use
  the same decision service.
- Images are an experimental extension, supplied as base64 or a data URL; see
  the [API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md).

## Evidence

- The extraction's CPU fixtures used synthetic engine outputs to check candidate
  independence, accepted-prefix identity, feature copies, cancellation and cleanup.
- The [SystemOne report](2026-10-12-systemone-unit-compatibility.md) records SDK
  comparisons.
- The [integration page](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md)
  has live-engine results and their measured scope.

For current installation and launch commands, see the component's README.
