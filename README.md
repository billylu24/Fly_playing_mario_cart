# Fly playing Mario Kart

Drosophila connectome policy experiments on a short Super Mario Kart bend task.
Code and chronological research notes are in `flykart/`; the latest completed
replication is documented in `flykart/distillation_replication_results.md`.
The experiments concern a short task on one track, not full-race competence.

## Publication contents and local prerequisites

This repository includes source code, experiment protocols, numerical metrics,
action traces, and reports. ROMs, screenshots/video, emulator save states,
frame datasets, virtual environments, raw connectome tables, and binary model
checkpoints remain local and are ignored. Model checkpoints are omitted pending
separate redistribution review; omission is not a conclusion that they infringe.
The repository alone therefore does not reproduce the archived runs end to end.

Dependencies: `flykart/requirements.txt` and `flykart/requirements-bench.txt`.
Experiments run from `flykart/` using `python -m training.<experiment>`.
Provide your own lawfully obtained game ROM and compatible local state. See
`flykart/mario_kart/README.md` for the integration provenance and local setup.
The third-party integration scripts/metadata are MIT licensed, with the original
notice in `flykart/mario_kart/LICENSE.upstream`; that license does not establish
redistribution rights for game assets or emulator memory snapshots.

Connectome sources and CC-BY 4.0 attribution are documented in
`flykart/data/malecns/README.md`. Large original tables are not included.
Numerical reports may reference paths to excluded local files.
