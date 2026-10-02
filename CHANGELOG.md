# Changelog

What each version contains, for whoever runs the bench or restores a file
with it. Versions are CalVer (`vYY.MM.DD.N`). Entries up to v26.10.02.1 were
written afterwards from `git log`; the commit messages carry the numbers.

## [v26.10.02.3] - 2026-10-02

- `unipase` engine (and `unipase:noplc`): denoising, bandwidth extension and
  packet loss concealment in one resynthesising model, inference vendored
  and pinned, 2.2 GB of weights fetched on first use. It turns digital
  silence into −32.8 dBFS of sound, so use `keepzero:unipase` on gated
  platform audio.
- UniPASE added to the `bwe` listening set.

## [v26.10.02.2] - 2026-10-02

- Two new engines, each behind its own extra: `universr` (flow-matching
  bandwidth extension to 48 kHz; reads the input's band itself and passes
  full-band audio through untouched) and `novasr` (a 53 KB upsampler).
  UniverSR is slow on a CPU: about 12 minutes per 8 seconds on an M1 Max.
- `scripts/listening_set.py --set bwe`: six bandwidth-extension comparisons
  on the pp53 voices and the one real 7.5 kHz call, each extender alone and
  behind the router.

## [v26.10.02.1] - 2026-10-02

- SAM-Audio (Meta, prompted source separation) added to the README's
  candidates as parked: it separates voices rather than repairing one, and
  needs CUDA and gated weights. Worth testing once there is a bleed or
  crosstalk degradation to measure it against.
- UniPASE, UniverSR and NovaSR queued as candidates.
- This changelog.

## [v26.09.23.1] - 2026-09-23

- Repairs measured on the real podcast voices: DeepFilterNet takes far more
  speech out of a reverberant room there than it did on EARS.
- `scripts/listening_set.py --set podcast` renders nine comparisons from the
  private pp53 tracks; `--set ears` is the previous set. A test checks that
  every plan names only recipes and engines that exist.

## [v26.09.22.1] - 2026-09-22

- Repairs that need no model: `dehum`, `deplosive`, `declip`, `keepzero`,
  with recipes (`hum`, `buzz`, `plosive`, `platform-upload`) to test them.
- A listening page that plays every take of a comparison in sync, with
  loudness matching and a blind mode.
- The `room` recipe corrected: DeepFilterNet removes about 5 dB of speech
  from reverberant input, not the 60 dB reported before.
- A hum probe, a cheaper `dehum`, and its prominence threshold swept rather
  than guessed.

## [v26.08.26.1] - 2026-08-26

- A `ground-loop` recipe and a classical `notch` engine, for the second use
  case (a good local microphone with mains hum), which had nothing. Built on
  the Studio and merged into the branch on 2026-09-23.

## [v26.08.25.1] - 2026-08-25

- `chain` and `router` engines. Chaining two generative engines is worse
  than either alone; the router invents only above the band the input
  actually has, at a 40 dB threshold calibrated on real material.
- DeepFilterNet as an engine: the best denoiser measured, and harmful
  unless its attenuation is limited.
- `earshot restore` for real files, including whole episodes without
  running out of memory.
- Recipes re-modelled on what real platform files contain (lossy encoding,
  gated digital silence), and two probes corrected that measured something
  other than what they named.
- Per-speaker tables, and a handover document for a second machine.

## [v26.08.24.1] - 2026-08-24

- The bench: damage recipes that keep the sample count, metrics including
  `origin` (filtering versus invention), six probes, PESQ/STOI as an extra,
  JSON results and a scoreboard.
- The engine contract (samples in, samples out, aligned), enforced at every
  call and by a test kit each engine must pass.
- Checksummed weight fetching, with `EARSHOT_NO_DOWNLOAD=1` for CI.
- LavaSR vendored, pinned and drift-checked, and measured against dxRevive.
- `docs/dxrevive.md`: what the commercial baseline does, by measurement.
