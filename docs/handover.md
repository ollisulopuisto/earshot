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
> `deepfilternet:12` may be a leash on an artefact. **This is the most
> valuable single thing left to re-run in this repo.**

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
