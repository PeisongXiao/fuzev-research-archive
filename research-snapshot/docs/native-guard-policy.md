# Historical Host-Guard Policy

The research service originally checked available host RAM and sustained
swap activity before and during work. Later experiments removed host-RAM and
swap stopping through a separate launcher while retaining physical cache,
page, slot and context checks.

The state observer recorded which policy was active, so an estimate of memory
demand could be distinguished from an observed allocation failure. Disabling
host admission did not make a physically impossible full-window comparison
feasible.

Historical
resource settings are described once in
[provenance](../../docs/research-provenance.md#historical-settings).
Current serving uses the configurable engine settings in the component docs.

[Snapshot methods and environment](../README.md)
