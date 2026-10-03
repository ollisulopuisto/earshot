# Handover

Written 25 August 2026, at the end of the session that built this, and
corrected the same day by the session that picked it up on a second machine.
For whoever picks it up next — read `AGENTS.md` first for the rules, then
this for the state.

Corrections are marked inline rather than folded in, because which claims
turned out to be wrong is itself the most useful thing here. The pattern held
every time: valid, accepted, and silently wrong, found by measuring the
output rather than reading the code. The full list is in the README under
*Corrections*.

## Where it stands

The bench works, six engines are wired up and four have been measured on real
podcast material. Every number quoted anywhere in this repo came from that
material on an Apple M2, not from a vendor's claim or a paper.

| engine | what it is | measured |
|---|---|---|
| `passthrough` | control; also `passthrough:6` for a pure gain | yes |
| `vst3:` | any VST3/AU, via pedalboard. The commercial baseline | yes |
| `lavasr` | bandwidth extension, Vocos, Apache-2.0, vendored | yes |
| `deepfilternet` | denoise only, PyTorch, `:N` bounds attenuation | yes |
| `chain:a+b` | engines in series, contract checked between stages | yes |
| `router:e` | the adaptive one: engine only where the input is empty | yes |
| `dehum[:50\|60][@bw]` | mains hum and buzz, subtracted per harmonic | on EARS |
| `deplosive[:corner]` | dynamic low band against the speaker's own balance | on EARS |
| `highpass:N` | the global filter `deplosive` has to beat | on EARS |
| `declip` | cubic redraw of short plateaus | on EARS |
| `keepzero:e` | puts a gate's digital silence back after `e` | listening only |

Seven result sets in `results/`, summarised by `earshot scoreboard`.

## What is actually known

**LavaSR nearly matches the commercial plug-in and preserves the voice
better.** +3.09 dB recovery on `narrowband-voip` against dxRevive's +3.82,
and it leaves the 1–4 kHz speech band completely untouched (`origin` +1.00
against +0.66) because it crossfades its output in above 4 kHz. It does not
denoise and does more harm than dxRevive on material that needed nothing.
This still holds: the `narrowband-voip` recipe has not changed.

> **The 77× against 7× does not hold.** It was measured by timing a single
> cold call. Speed now has its own pass — repeats back to back, median, with
> the spread beside it — and no figure taken the old way is comparable to one
> taken the new way. On an M1 Max under load LavaSR measures 43.6× at 1.1×
> spread. dxRevive has not been re-measured; it is not installed on that
> machine and cannot be automated there anyway.

**DeepFilterNet is the best denoiser here.** PESQ +1.96 on noise against
dxRevive's +1.01, preserving the speech band better. That part stands — it
was measured on `hiss`, which has not changed.

> **The leash is unverified.** "On reverberant speech it removes 60 dB of
> voice" was measured with a `reverb()` that also applied −28.1 dB of level,
> so the engine was working on a signal twenty times quieter than it should
> have been. Corrected, LavaSR's `room` numbers moved from −6.12 dB to
> +0.13 dB and its apparent +19.67 dB of added floor to +1.56. DeepFilterNet
> was not re-measured — it needs an extra that is not installed — so
> `deepfilternet:12` may be a leash on an artefact. ~~This is the most
> valuable single thing left to re-run in this repo.~~ **Re-run 22 September
> on EARS: it was the artefact.** Unbounded DFN takes 5.0 dB of speech on
> the corrected `room`, not 60; @20 dB takes 4.45 and recovers the most,
> +0.81 dB. Still to repeat on the podcast material.

**The router does what it was built to do, conservatively.** On material that
needs nothing it changes nothing at all — `origin` +1.00 in every band,
non-speech untouched. On `narrowband-voip` it recovers most of what an
always-on engine does (+0.35 against +0.42) while holding `origin` far higher.

> **The threshold is not the open question it looked like.** Measured at 25,
> 40 and 60 dB on real material, the margin barely moves anything: on
> `wideband-voip` all three give +0.00 to +0.02, and on `narrowband-voip` the
> recovery is 0.35 / 0.35 / 0.31 with `origin` high at 0.19 / 0.29 / 0.35.
> `CLIFF_DB` is not the lever. Note also that the margin was silently ignored
> until this was fixed, so the two `router-cliff*` result files cannot be
> told apart by anything they record.

**Chaining unconditionally makes things worse.** dxRevive → LavaSR scored
−6.12 dB on clean against dxRevive's −3.78: the second generative stage
overwrites the first's work with its own guess. Routing the second stage
fixes it exactly.

## 22 September 2026: repairs, EARS, and a listening page

Picked up from a planning spec written without knowledge of this repo. What
was taken from it, and what was not, is in the PR that added this section
(#9). In short: the spec's model-free repairs (hum, plosives, sparse
clipping) were built as engines; its architecture (DAG scheduler, device
workers, remote queue) was not, because nothing measured here needs it yet.

**The DeepFilterNet re-measure is done.** Corrected `room`, unbounded DFN
takes 5.0 dB of speech, not 60; the 20 dB bound recovers most (+0.81 dB).
Numbers in the README under *The denoiser*.

**Second-opinion material.** The session had no podcast audio, so it used
EARS (Meta, CC BY-NC 4.0): anechoic studio speech, 48 kHz, 107 speakers,
fetched from GitHub releases. Three speakers are in `material/local/ears/`
on the machine that ran it and nowhere else. It is English and anechoic,
which makes it good ground truth and a poor stand-in for a Finnish podcast
room. Do not commit it: NC is not compatible with this repo's licence.

**Listening.** `earshot listen DIR` builds the synced, loudness-matched page;
`scripts/listening_set.py` renders ten comparisons into `out/kuuntelu/`.
The set from this session is published as a private artifact for the owner.

**Blocked, not tried.** AP-BWE (Google Drive), FlashSR, Resemble Enhance,
VoiceFixer and ClearerVoice (Hugging Face) all host weights where the cloud
session could not reach. DeepFilterNet's own download URL also returned 403
there; its weights were taken from a sparse git checkout of the upstream
repository instead, which is the same file.

## 3 October 2026: four new candidates, and a first impression

UniverSR, NovaSR, UniPASE and AudioSR are wired as engines (see the README's
candidates table); none is benched yet. Two listening sets exist: `bwe`
(pp53 voices and the real 7.5 kHz call, rendered locally) and `bwe-ears`
(EARS, rendered on Colab via `scripts/colab.py`).

**First impression, not a finding.** The owner's quick listening: UniPASE
was usually the most promising. Not blind, not on headphones, and stated by
the owner as very preliminary. It is recorded so the next session knows
where attention is pointing, not as evidence. What would make it one:

- a blind listen on headphones, the page's *Sokko* mode;
- a speaker-similarity probe, since UniPASE resynthesises everything and
  keeps no sample of the input — `origin` will read near zero by design and
  cannot say whether the voice is still its owner's;
- its known fault measured on real material: it speaks into digital silence
  at −32.8 dBFS (contract kit), so gated audio wants `keepzero:unipase`.

**Blind picks, same day** (`votes.jsonl` beside each set, Sokko on, owner,
15:08–15:14 Helsinki). One best take per comparison, on EARS voices:

| set | comparison | picked |
|---|---|---|
| calls-ears | VoIP call, p001 | UniPASE |
| calls-ears | VoIP call, p008 | UniverSR (4 s chunks) |
| calls-ears | overload | declip → UniPASE |
| calls-ears | overload on a call | UniPASE |
| bwe-ears | telephone band, p001 | UniPASE |
| bwe-ears | telephone band, p002 | UniPASE |
| bwe-ears | wideband call | UniPASE |
| bwe-ears | platform upload | UniPASE |

UniPASE 7 of 8, blind this time, and LavaSR, NovaSR, AudioSR and every
router variant never picked. Still one listener and one voice per damage,
and the speaker-similarity question above is unanswered: the picks say
which sounded best, not which still sounds like its speaker.

**The pp53 set (`out/kuuntelu-bwe`), blind, same day.** Saved picks:
UniPASE behind the router on the Nyman call, plain UniPASE on the wideband
call, `keepzero:unipase` on the real call. The owner's spoken notes named
four takes terrible or worst (clean microphone, Nyman call, platform, real
call), and all four were the broken UniverSR — heard blind, before anyone
had said it was broken. Everything else on the clean microphone "sounds
pretty much the same", which is what a clean input should do.

Mapping a spoken "Otto E" to a take by recomputing the page's shuffle went
wrong twice: the page shuffles in JavaScript doubles, and an exact Python
copy diverges once the product passes 2^53. The saved picks carry the
real take; trust those, or reproduce the float arithmetic.

**The podcast repairs set (`out/kuuntelu-podcast`), blind.** On the Nyman
room the owner's best pick was *the damaged take*: every DeepFilterNet
setting was worse than doing nothing ("all bad"). On hiss, unbounded
DeepFilterNet. Nothing on the bench is built for dereverberation;
UniPASE (trained with reverberation) and VoiceFixer are the next to try.

**The body set (`out/kuuntelu-body`), blind.** Plain UniPASE was picked
over UniPASE → Body on all three comparisons with a pick (Nyman call,
landline, real call). The body probe says the repair restores most of the
missing low end (−20 → −5..−8 dB); the ear did not prefer it. Unknown
whether the synthesised harmonics sound artificial, or whether UniPASE's
own low end was already enough — untested.

**Speaker similarity says the opposite of the ear** (`earshot.speaker`,
WeSpeaker ResNet34, cosine to the clean original, each take minus the
damaged input of the same comparison, all rendered sets): plain UniPASE
−0.041 median, worse than the damage it was given in 13 of 16
comparisons (worst −0.179); UniPASE behind the router +0.001, better in 11
of 13; LavaSR −0.026; UniPASE → Body −0.052. The owner's blind favourite
is the take that moves the voice furthest from its owner — the trade the
project's rule is about. The router keeps the speaker's own signal below
the band edge and with it the identity. One embedding model, few
comparisons per engine: a strong lead, not a verdict. The broken UniverSR
scored −0.81, so the probe sees a voice that is gone.

**First bench of the new candidates on podcast voices** (`results/
2026-10-03-*-pp53-m1max.json`: six 8 s excerpts of pp53 Nyman and Wancke,
nine damages, M1 Max CPU; UniverSR absent — 12 min a call on this CPU).

- **UniPASE recovers the most where there is something to recover:**
  log-spectral gain +1.95 dB on room, +2.70 wideband call, +1.33 VoIP call,
  +1.64 landline, +1.63 overload on a call; on the landline it also puts
  back 11 dB of the 32 dB of low end the line removed (body −32.6 →
  −21.1). Nothing else recovers more than +0.65 anywhere.
- **It damages clean audio:** −3.55 dB, PESQ −1.06, speaker −0.109. Run
  blind on everything, it would harm the good microphones; it needs a gate
  that leaves full-band clean audio alone, as `universr` and the router do.
- **Speaker similarity** moves −0.03 to −0.16 under UniPASE (+0.18 on the
  gated platform audio); LavaSR is worse at −0.08 to −0.21. The router
  variants read ±0.00 — but the speaker model hears only 0–8 kHz, and they
  change nothing below their band edge, so that reading is blind to them.
- **router(unipase) does almost nothing** on most damages, because the edge
  sits high, and costs on narrowband (−1.47 dB, PESQ −0.33).
- **declip → unipase** equals UniPASE on the overloaded call (the codec
  smears the flat tops declip looks for) and is marginally better on plain
  overload (−1.25 vs −1.40).
- **PESQ falls under UniPASE** almost everywhere, as metrics.md predicts for
  a generative engine; it is not the arbiter here.

**Sidon on the same bench** (`results/2026-10-03-sidon-pp53-m1max.json`):
it out-recovers UniPASE on every call-like damage — wideband +3.28 vs
+2.70 dB, narrowband +1.32 vs −0.44, VoIP call +2.93 vs +1.33, overloaded
call +2.76 vs +1.63 — and puts the low end back (narrowband body −17.4 →
+1.0, landline −32.6 → −1.2, where UniPASE leaves −17 and −21). It loses on
room (+0.76 vs +1.95), plain overload (−3.41 vs −1.40) and clean audio,
which it harms more than UniPASE (−5.41 dB, PESQ −3.15). Speaker drift is
mixed: smaller on the calls (−0.010 vs −0.028 VoIP), larger on the landline
(−0.161 vs −0.029). Neither wins everywhere — the case for a per-segment
choice and a gate that leaves clean audio alone.

**Every UniverSR take in that first listen was broken** (fixed in ceab298).
The engine handed upstream's `enhance()` 48 kHz audio labelled with the
band's rate; upstream takes an array to be at that rate, so speech came out
stretched six-fold — 1–3 kHz 49.7 dB down, 40–80 Hz 50 dB up. The UniverSR
takes in `out/kuuntelu-bwe` and the first Colab `bwe-ears` are invalid, and
so is the "+7 to +25 dB of low end" once reported for it. The contract kit
checks length and alignment, not content; a test now requires the speech
band through at r > 0.95. The lesson for every new engine: check that it
returns the input where it claims to keep it, not only that it returns
the right number of samples.

## What is not known, in priority order

**1. ~~Whether any of this holds on real VoIP.~~ Answered — see the README's
*What the material actually is*.** Sixteen files were measured. The answer
was neither of the two options this section offered: the platform's "local
uploads" are lossily encoded, cutting at 14.0–15.6 kHz with a stopband ripple
of 0.3 dB that no room produces, and exactly one file is a genuine
codec-limited call at 7.5 kHz. A `.wav` extension proved nothing.

The dominant defect turned out to be one no recipe modelled: the silence is
exact digital zero, 8.0 to 57.8 per cent of every platform file against
nothing at all in the studio tracks. `platform-upload` models it now.

What remains open here is narrower and still matters:

- **The ground truth is two speakers, not sixteen files.** Only `nyman` and
  `wancke` are genuinely full-band. Everything else can support unreferenced
  probes only, because damaging an already-encoded file measures recovery
  toward a ceiling that is not there.
- **No clean-and-damaged pair of the same speech exists**, and the owner
  believes none can: every remote platform encodes on the way in. The one
  near-pair found — a Zencastr WAV against its own MP3 backup — is two
  encodings, not clean against damaged, and the WAV cut *lower* than the MP3.
- **`origin`'s `air` band is 14–20 kHz**, which sits inside the stopband of
  every platform file. On that material the column correlates two noise
  floors and means nothing. It has not been guarded against.

**2. ~~The router's threshold.~~ Measured, and it is not the lever.** 25, 40
and 60 dB were compared on real material and barely differ; see the
correction above. What the router does need is a reason to engage on gated,
band-limited platform audio at all, which is a different question from where
its threshold sits.

**3. Whether the numbers match what anyone hears.** Still open, and now the
most useful thing an owner can do that an agent cannot. `out/kuuntelu/` holds
47 takes across 10 comparisons and an `index.html` that plays them in sync so
switching is instant — reload-and-restart A/B cannot answer a timbre question.
Four comparisons carry a specific prediction: LavaSR's +34.54 dB of added
floor on `platform-upload`, whole-file against chunked, the real 7.5 kHz call,
and whether −2.92 dB on untouched material is audible at all.

**4. Whether chunked restoration is good enough.** It is not the same as
whole-file processing and cannot be: LavaSR is deterministic but neither
linear nor local. Chunked against whole-file is −22.0 dB, and that is not the
crossfade — no overlap at all measures −26.1 dB. Only ears can settle which
is better.

## Traps

**Do not edit anything in `src/earshot/vendor/`.** A test fails if the hash
moves, because the results name an unmodified project. Patch at the call site.

**A vendor bump must arrive with new bench results in the same change.**
Different code means different numbers.

**Weights are never committed.** They are fetched and checksummed.
`EARSHOT_NO_DOWNLOAD=1` must keep working; CI sets it.

**CI has no extras and no network.** Anything requiring onnxruntime, torch, a
plug-in or ffmpeg must skip with a stated reason. A test that reaches through
`engines.load()` to check a fetcher message will pass locally and fail in CI —
that happened; ask the fetcher directly instead.

**Two failed experiments are recorded so they are not repeated:** a knee-based
edge detector (`router.py` docstring) and unconditional chaining (README).

**A parameter that is accepted and ignored puts a number in the record that
never reached the code.** This happened twice: `router:<engine>@<margin>` was
parsed, stored and written into the result notes while the edge detector read
the module constant, and `RouterEngine.name` did not carry the margin, so a
run at three thresholds filed 111 rows under one name and averaged three
engines together. Check that a knob moves an output before trusting a result
that names it.

**A degradation can model two damages at once and nobody notices.** `reverb()`
applied −28.1 dB along with the reverberation. Every recipe is now checked by
`test_a_recipe_is_damage_and_not_a_fader`, which asserts no recipe smuggles a
level change past the ones that level on purpose.

**A recipe calibrated on one material can do nothing on another.** The
`platform-upload` gate sits 18 dB under the speech because podcast room tone
sits 22 dB under; EARS pauses sit 18 dB under, and on EARS the gate never
closed. The bench ran, produced a full table, and measured a 15 kHz
low-pass. Check that a damage did what its name says (here: the fraction of
exact zeros) before reading a table about it.

**LSD cannot see hum.** A few narrow bins vanish into a band average: dehum
scored +0.01 dB of `gained` while removing 6.6 dB at the harmonics. The
`tonal` probe measures the lines themselves.

**Synthetic material fools the hum detector.** `probes.default_material` is a
pulse train that dwells near 450 Hz long enough to look like a ninth
harmonic. Tests that need "no hum" use a gliding tone instead.

**Fading a chunk in place rewrites the caller's array.** An engine that
returns its input unchanged returns a *view*. This nearly shipped in
`process_in_chunks` and would have damaged the file being restored, with 0.38
of error at every seam.

**Memory scales with the length of one `process()` call, not with the file.**
LavaSR needs 0.30 GB for 5 s and 1.41 GB for 160 s. `restore` chunks by
default because of it; `--chunk 0` restores the old whole-file behaviour.

## Running it on another machine

Almost nothing needs copying. That is deliberate.

```bash
git clone git@github.com:ollisulopuisto/earshot.git
cd earshot
uv sync --extra dev
uv run pytest -q                 # 110 pass, 1 skipped, no models needed
uv run earshot bench             # synthetic material, works immediately
```

Then, as needed:

```bash
uv pip install -e ".[lavasr]"          # onnxruntime; 58 MB fetched on first use
uv pip install -e ".[deepfilternet]"   # PyTorch, ~2 GB; weights fetched on use
uv pip install -e ".[perceptual]"      # PESQ and STOI
uv pip install -e ".[vst3]"            # pedalboard, for the commercial baseline
```

**Model weights do not need transferring.** `earshot.fetch` downloads them on
first use and verifies every one against a digest recorded in the repo, so a
second machine gets provably identical models. DeepFilterNet fetches its own
into `~/Library/Caches/DeepFilterNet`.

What does need attention on a second machine:

- **ffmpeg**, for the real-codec damage recipe. Without it that recipe skips
  with a reason rather than failing.
- **dxRevive**, if the commercial baseline is wanted. It is licensed
  separately and its models live in `/Users/Shared/Accentize/`.
- **Material.** The podcast audio lives in Dropbox and much of it is
  online-only; it has to be materialised before the bench can read it.
- **Memory, if the machine is loaded.** The bench peaks around 2 GB and
  `restore` around 3.4 GB on a full episode. On a machine with little free
  memory, background runs get killed by jetsam while foreground ones survive
  — that happened twice on the second machine, which had 20.7 GB wired by a
  VM and local models and 600 MB of swap left. Run the bench in the
  foreground there.
- **Never sync `.venv`.** It is 778 MB of platform-specific binaries with
  absolute paths in them, and a venv that syncs between machines half-works in
  ways that waste an afternoon. Both project venvs have been marked
  `com.dropbox.ignored`; undo with `xattr -d com.dropbox.ignored .venv`.
- `out/` is gitignored and holds the owner's audio. It reaches a second
  machine through Dropbox, not through git, and it should not be committed.

## The destination

An RX-Connect-style bridge plugin — issue #8 has the full reasoning. Not a
real-time plugin: everything here is offline by construction, which is the
contract the bench rests on. Not yet, either: the engines are still moving.

Final Cut is already served by the sister project `autoraffkat`, which
processes offline and redirects assets. The bridge is for Logic, Hindenburg
and everything else.

## The one rule worth repeating

Four times in the session that built this, something was valid, accepted and
silently wrong — XML that passed the DTD and did nothing, a metric that
reported an engine losing where it gained, a determinism check that was wrong
twice, a program trim that was normalised away by the next stage. Every one
was found by measuring the output rather than trusting the code.

Measure first. A claim without a number is not a finding.
