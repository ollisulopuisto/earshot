# Working in this repo

For agents and for people. The two need the same things: what the project is
for, what it refuses to do, and how to tell whether a change was an
improvement.

## What this is for

Speech restoration for **podcast production**. Human voices, two source
classes: good local microphones (little to restore, much to protect) and VoIP
remote guests (band-limited, codec-damaged, clipped, auto-gained). Not music,
not field recordings, not general audio.

The design goal, in the owner's words: *restore human soundingness*. When a
change would trade a speaker sounding like themselves for a better score,
that is not an improvement, and the `origin` probe exists to catch it.

## The rule the whole bench rests on

**Samples in, samples out, aligned.** Every engine returns exactly as many
samples as it was given, at the same positions. Every degradation does too.
The bench compares sample against sample; anything that shifts the signal is
scored as damage that no engine could ever undo, and every number downstream
becomes a measurement of the offset instead of the restoration.

`engines.check_contract` enforces this at the point of use rather than
trusting it. Do not remove it, and do not add a code path that skips it.

## Measure, then claim

No performance or quality claim goes into this repo without a number and how
it was obtained. `docs/dxrevive.md` is the model: what was measured, on what
material, with what result. "Sounds better" is not a finding; "recovers
1.5 dB LSD in the band removed at 4 kHz on a 48 kHz podcast microphone" is.

If you cannot measure it, add the probe first.

## Weights and results

Model weights are declared, not downloaded by hand: `fetch.Asset(name, url,
sha256)` and `fetch.ensure(asset)`. The checksum is not optional — a model
file that changed upstream turns every number ever measured with it into a
number about a different model, and nothing else would notice.
`EARSHOT_NO_DOWNLOAD=1` must keep working; CI sets it.

Measured results go in `results/` as JSON and are summarised by `earshot
scoreboard`. Post the numbers in the PR too, but the file is the record.

## Material

`material/` holds committed excerpts that speakers have agreed to publish.
`material/local/` is gitignored and is where anything unconsented stays. A
voice is a person: having the file is not the same as being allowed to
publish it.

## Vendored code

`src/earshot/vendor/` holds third-party source copied in under its own
licence. Vendoring is a fork unless the drift is handled, so:

- `PINNED` records the origin, commit, date and SHA-256 of every file, and a
  test fails if the file on disk no longer hashes to it. **Never edit a
  vendored file**, not even to fix a bug — the stored results name an
  unmodified project. Patch at the call site instead, or send upstream a PR.
- A scheduled workflow fails when upstream moves. It updates nothing.
- A vendor bump must arrive **with new bench results in the same change**.
  Different code means different numbers, and the old ones stop being
  comparable the moment the file changes. A bump with no new results is the
  thing to reject in review.

Weights are never vendored: they are fetched and verified against a digest,
which pins them just as firmly without putting tens of megabytes in the repo.

## Adding an engine

`src/earshot/engines/` — one class with `name` and `process(audio, rate)`, one
`@register("scheme")` loader, and one test that calls
`earshot.testing.assert_engine_contract`. That call is not optional: it is
how the contract stays true without a reviewer holding it in their head.
Then:

- Real weights and models are **not** committed. They are downloaded, cached,
  or found on the system. Keep the repo small enough to clone on a phone.
- If the engine needs an optional dependency, put it behind an extra in
  `pyproject.toml` and raise `EngineError` with the install command. CI has
  none of them installed and must still pass.
- Never vendor someone else's weights, and never lift weights out of a
  commercial product. Licences differ; check before integrating, and record
  the licence in the README table.

## Adding a probe or a degradation

A probe answers one question with one number, and it must have changed a
decision — or be able to. Say which, in the docstring.

A degradation models something that actually happens to a podcast. It keeps
the sample count. If it needs an external tool, set `needs` so the bench can
report a skip with a reason instead of measuring something else quietly.

Both need a test that would fail if the thing broke. `test_probes.py` shows
the pattern: passthrough must score as doing nothing, and a plain gain change
must not look like cleaning.

## Testing

```bash
uv run pytest -q
```

Must pass with no models, no plug-ins and no ffmpeg — that is what CI has.
Anything requiring more is skipped with a stated reason, never silently.

The synthetic material in `probes.default_material` exists so the bench runs
anywhere. It is not a substitute for real voices and must never be used to
support a claim about quality.

## Style

English, in code and prose. Comments explain *why*, not *what* — the code
already says what. Where a number appears in a comment, say where it came
from.

## Check-in steps

Small commits that each leave the tests green. The commit message says what
changed and why it mattered; if it was a measurement, put the number in the
message. Work in public: an incomplete finding stated honestly is worth more
than a confident one that was never checked.
