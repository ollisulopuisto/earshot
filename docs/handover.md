# Handover

Written 25 August 2026, at the end of the session that built this. For
whoever picks it up next — read `AGENTS.md` first for the rules, then this
for the state.

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
better.** +3.09 dB recovery on band-limited material against dxRevive's
+3.82, at 77× realtime against 7×, and it leaves the 1–4 kHz speech band
completely untouched (`origin` +1.00 against +0.66) because it crossfades
its output in above 4 kHz. It does not denoise and does more harm than
dxRevive on material that needed nothing.

**DeepFilterNet is the best denoiser here and needs a leash.** PESQ +1.96 on
noise against dxRevive's +1.01, preserving the speech band better. On
reverberant speech, unbounded, it removes 60 dB of *voice* — it decides a
live room is noise. `deepfilternet:12` is safe everywhere and costs 5 dB of
denoising power. The right bound is a judgement about real rooms.

**The router does what it was built to do, conservatively.** On material that
needs nothing it changes nothing at all — `origin` +1.00 in every band,
non-speech untouched. It recovers about a third of what an always-on engine
recovers on band-limited input. That trade (5 dB less harm for 1.3 dB less
recovery) is the open tuning question.

**Chaining unconditionally makes things worse.** dxRevive → LavaSR scored
−6.12 dB on clean against dxRevive's −3.78: the second generative stage
overwrites the first's work with its own guess. Routing the second stage
fixes it exactly.

## What is not known, in priority order

**1. Whether any of this holds on real VoIP.** Everything above rests on
synthetic damage plus one real Opus recipe. The owner has ~40 remote episodes
and ~20 studio episodes of his own podcast (`vikasietotila`, three male
speakers, Finnish, about an hour each) — and **none of it is on disk**;
Dropbox holds them as zero-byte placeholders. This is the single biggest
constraint on everything else. Issue #6.

There is a question to settle before measuring: those files are named like a
remote-recording platform's local uploads (`panu-recording-4_…wav`). If they
are local recordings, the "VoIP" episodes are not codec-damaged at all — they
are good recordings of bad rooms, which needs denoising rather than bandwidth
extension. One spectrum settles it: content stopping at 8 kHz means a call,
running to 20 kHz means local.

**2. The router's threshold.** `CLIFF_DB = 40` was calibrated on four
excerpts from one episode. The measurement is in the source comment. Real
VoIP would tighten it.

**3. Whether the numbers match what anyone hears.** The bench has no
perceptual ground truth. `out/kuuntelu/` holds a listening set; nothing has
been checked by ear against the tables.

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

## Running it on another machine

Almost nothing needs copying. That is deliberate.

```bash
git clone git@github.com:ollisulopuisto/earshot.git
cd earshot
uv sync --extra dev
uv run pytest -q                 # 86 tests, no models needed
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
