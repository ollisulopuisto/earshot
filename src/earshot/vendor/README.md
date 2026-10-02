# Vendored code

Copied in, not written here. Each file's origin, commit and hash are recorded
in `__init__.py` as `PINNED`, and the licence text sits beside it.

## LavaSR-ONNX — Apache-2.0

`lavasr_core.py`, `lavasr_config.yaml` from
[Topping1/LavaSR-ONNX](https://github.com/Topping1/LavaSR-ONNX) at commit
`1a979b8`, taken 2026-08-24. Licence: `LICENSE-LavaSR-ONNX`. Derived in turn
from [ysharma3501/LavaSR](https://github.com/ysharma3501/LavaSR).

Model weights are **not** here — they are fetched on first use and verified
against a digest.

## UniPASE — MIT, with Apache-2.0 parts

The inference tree under `unipase/` — eighteen files from `models/` of
[xiaobin-rong/unipase](https://github.com/xiaobin-rong/unipase) at commit
`857b60a`, taken 2026-10-02; each pin records its upstream path. Training
code and discriminators are left out. Licence: `LICENSE-unipase` (MIT), and
`LICENSE-unipase-*` for the code it derives from in turn (Cisco's PASE and
ESPnet under Apache-2.0, WavLM, Vocos/WavTokenizer and others).

The post-net imports espnet for one base class and one lookup.
`earshot.engines.unipase` supplies both at the call site instead of
installing espnet; the files here are untouched.

The four weight files (2.2 GB) are fetched and verified like any other.

## Updating

Deliberately, never automatically:

1. `.github/workflows/vendor-check.yml` fails when upstream has moved. That
   is the only thing that notices; it does not update anything.
2. Copy the new file in, update its entry in `PINNED` (commit, date, hash).
3. **Re-run the bench and commit new results.** Different code means
   different numbers, and the old ones stop being comparable the moment the
   file changes. A vendor update with no new results in the same commit is
   the thing to reject in review.

## Local edits

Don't. `tests/test_vendor.py` fails if the file no longer hashes to its
recorded value, because a local fix would mean the bench is measuring
something that is not the project named in the results table. If upstream is
wrong, patch it at call sites in our own code, or send them a pull request.
