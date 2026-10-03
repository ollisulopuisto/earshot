# Changelog

What each version contains, for whoever runs the bench or restores a file
with it. Versions are CalVer (`vYY.MM.DD.N`). Entries up to v26.10.02.1 were
written afterwards from `git log`; the commit messages carry the numbers.

## [v26.10.03.5] - 2026-10-03

- A `speaker` probe in the bench: does the restored voice still sound like
  its owner (speaker-embedding similarity to the original). Needs the
  `speaker` extra; left out without it.
- Listening pages: a Huonoin (worst) button beside Paras, and no buttons on
  the reference.
- UniPASE added to the room comparisons of the podcast set.

## [v26.10.03.4] - 2026-10-03

- Listening pages open in blind mode (Sokko) by default.
- A "Paras" button on every take: pick the best per comparison. Served
  with `earshot listen DIR --serve PORT`, picks are kept in
  `DIR/votes.jsonl` with the real take behind the blind label;
  `earshot votes DIR` lists them.
- UniverSR fixed: every earlier UniverSR take was stretched six-fold and
  is invalid.

## [v26.10.03.3] - 2026-10-03

- `body` engine: puts back the low end a telephone line removed, by
  rebuilding the voice's lowest harmonics from its pitch (after Pulakka et
  al.). Chain it after a restorer: `chain:unipase+body`. Leaves voices that
  still have their low end untouched. On phone-band damage it brings the
  low end from about 20 dB short to 5–8 dB short.
- A `body` probe in the bench: the low end's balance against the speech
  band, compared with the original.
- `scripts/listening_set.py --set body`: phone calls and the real call,
  with and without `body`.

## [v26.10.03.2] - 2026-10-03

- Believable call damage, after the synthetic calls were judged not
  believable: `voip-call` (Opus wideband at 12 kbit/s, 5 % of packets lost
  in bursts and concealed by the real decoder, not cut out), `landline`
  (G.711 telephone line) and `overload` (input gain 12 dB past full scale),
  plus `overload-call`. The Opus recipes need ffmpeg and libopus
  (`brew install opus-tools` brings libopus) and are skipped without them.
- Weight downloads that end early are fetched again instead of failing
  their checksum.

## [v26.10.03.1] - 2026-10-03

- `audiosr` engine (`audiosr:<DDIM steps>`, default 50): AudioSR's speech
  checkpoint, run in its own Python because its pinned dependencies cannot
  live beside earshot. Point `EARSHOT_AUDIOSR_PYTHON` at that interpreter;
  `scripts/colab.py --audiosr` builds it on the VM. Its 6.2 GB of weights
  are pinned like every other model's.
- `listening_set.py --one-at-a-time` loads and frees each engine per take,
  for GPUs too small to hold them all; the Colab default now uses it.
- The `bwe-ears` set includes AudioSR, alone and behind the router.

## [v26.10.02.4] - 2026-10-02

- `scripts/colab.py`: runs an earshot command on a Colab GPU and brings the
  result back (`--dry-run` shows every command first). By default it
  renders the new `bwe-ears` listening set on EARS voices fetched on the
  VM, so nothing private leaves the machine; your own audio goes up only
  when named with `--upload`. The VM runs the exact pushed commit.
- `EARSHOT_DEVICE` picks where the PyTorch engines run; unset, CUDA is used
  when present. The Apple GPU is used only when asked for.

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
