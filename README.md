# earshot

**Measure first, then restore.** A bench for speech restoration, aimed at one
job: making podcast voices sound like people again.

Restoration models are chosen by demo, and demos are chosen to flatter. This
repo exists because a plug-in that rescues a phone call spectacularly may also,
on a recording that needed almost nothing, quietly replace a speaker's voice
with its idea of a voice — and no listening test on a demo file will tell you
that. A correlation per band will, in one number.

## The goal, stated narrowly

Podcast production. Human speech, two source classes:

- **Local microphones.** Decent gear, a real room. There is little to restore;
  the wins are a lower noise floor and whatever "character" means, and the risk
  is an engine doing damage in the name of help.
- **Remote guests.** VoIP: band-limited, codec-mangled, clipped, auto-gained,
  dropping packets. Here there is a great deal to restore, and it is the case
  every restoration model advertises.

Not music. Not field recordings. Not general audio. The narrowness is the
point — it lets the bench use degradations that actually happen to podcasts
and metrics that mean something for voices.

## How it works

Real damaged recordings have no clean counterpart, so there is nothing to
measure recovery against. The bench takes material that is already good,
**breaks it in known ways**, restores it, and measures the distance back to
where it started.

```
clean speech ──▶ degrade ──▶ engine ──▶ metrics ──▶ table
     │                                     ▲
     └─────────────── reference ───────────┘
```

Six probes, each of which has changed a decision at least once:

| probe | question |
|---|---|
| `recovery` | Did it get the missing band back? (needs the clean reference) |
| `origin` | Per band: is the output still the input, or something new? |
| `cleanup` | Did the noise floor drop *without* the speech dropping with it? |
| `preservation` | What happens to sound that is not a voice? |
| `stability` | Same input twice, how close — per band? |
| `throughput` | How much faster than realtime, on how many cores? |

`origin` is the one that does not exist elsewhere. A spectral mask leaves the
waveform's phase intact and correlates near 1. A vocoder that resynthesises a
band correlates near 0 even when it sounds superb. Neither is wrong — but an
engine scoring 0.1 in a band that was undamaged has replaced something you did
not ask it to replace.

## Quickstart

```bash
uv pip install -e ".[dev]"

# Works with nothing installed: synthetic material, the passthrough control
earshot bench

# Your own voice, your own damage, and audio you can listen to
earshot bench --input take.wav --start 300 --seconds 30 \
              --damage narrowband-voip \
              --engine passthrough \
              --engine 'vst3:/Library/Audio/Plug-Ins/VST3/Thing.vst3?mix=100' \
              --out out/ --write

earshot damages     # what the recipes are
earshot scoreboard  # one table across everything measured so far
```

Point `--input` at a **directory** and it samples excerpts across it,
avoiding silence, spreading them over the recordings, and reporting the
spread as well as the mean. One excerpt of one voice is an anecdote:

```bash
earshot bench --input material/local --seconds 10 --excerpts 8 --out out/
```

`--out DIR` writes `report.md` and `results.json` side by side; `--json`
prints the machine-readable form to stdout. Results record the earshot
version and the machine, because `throughput` means nothing without the
latter and a probe changing its definition is the obvious way for two
numbers to stop being comparable without anyone noticing.

`passthrough` is always worth including. A metric that scores "do nothing" as
an improvement is measuring itself.

## Adding an engine

One method, three rules:

```python
from earshot.engines import Loaded, register

class MyEngine:
    name = "mine"
    def process(self, audio, rate):   # mono float32 in, mono float32 out
        return restore(audio, rate)

@register("mine")
def _load(argument):
    return Loaded(MyEngine())
```

1. **Same number of samples out as in.** The bench compares sample against
   sample; an engine that trims its own latency reports as damage everything
   it shifted.
2. **Aligned to the input.** If it has latency, it compensates it. Processing
   here is offline — there is no excuse.
3. **One channel in, one channel out.** Restoration is per microphone.

Violations raise `EngineError` and are reported, never swallowed.

And one line in your test, which checks all three plus the cases that have
actually broken engines — silence, lengths that are not a whole number of
frames, a DC offset, and state leaking from one call into the next:

```python
from earshot.testing import assert_engine_contract

def test_my_engine_keeps_the_contract():
    assert_engine_contract(MyEngine())
```

It passes a real stochastic commercial plug-in, so being non-deterministic is
not a failure. Carrying the previous file's tail into this one is.

## Reference numbers

The probes are calibrated against a commercial plug-in on real podcast
material — see [`docs/dxrevive.md`](docs/dxrevive.md) for the full
characterisation, including what its architecture turns out to be and where
its limits are. In short, on a good local microphone it buys about **6 dB of
noise floor** with the speech level untouched, recovers **1.5 dB of
log-spectral distance** in a band removed at 4 kHz, preserves phase below
12 kHz while inventing everything above 14 kHz, and **deletes non-speech
entirely** (−57 to −67 dB). Those are the numbers an open replacement has to
beat, and they are the reason the `preservation` probe exists.

The bench also found something no review reports: **its bandwidth extension is
stochastic.** Below 1 kHz it repeats to +74 dB; in the bands it invents, two
runs of the same input differ by *more than the signal itself*. Deterministic
where it filters, a dice roll where it generates.

## Read the metrics before trusting them

They disagree with each other, on purpose. Measured on real podcast
material — six excerpts, two speakers, dxRevive at `mix=100`:

| damage | `recovery/gained` | PESQ | STOI |
|---|---|---|---|
| clean | −3.45 dB | −1.89 | −0.07 |
| hiss | **+8.65 dB** | **+0.75** | −0.04 |
| narrowband-voip | **+2.84 dB** | **−0.83** | +0.03 |

On noise everything agrees. On band-limiting **PESQ says the opposite of
everything else** — because the engine invents the missing band, and PESQ
scores invention as distortion. Every candidate model's paper leads with
PESQ, so selecting by it picks the least generative model, which for a
remote guest on a telephone codec is exactly the wrong one.

[`docs/metrics.md`](docs/metrics.md) has the argument and the numbers.

## Candidates being evaluated

| engine | approach | licence | status |
|---|---|---|---|
| [LavaSR](https://github.com/ysharma3501/LavaSR) | Vocos, bandwidth extension + denoise, 56 MB | Apache-2.0 | **measured** — see below |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | real-time denoise only, very fast | MIT/Apache-2.0 | queued |
| [Sidon](https://arxiv.org/html/2509.17052v1) | w2v-BERT + HiFi-GAN resynthesis, 250M params | CC BY 4.0 | queued |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | denoiser + enhancer | MIT | queued |

## First head-to-head

Four excerpts, two speakers, real podcast material. `gained` is recovery
against the untouched original; `origin` is how much of the output is still
the input in that band.

| | dxRevive | LavaSR | what it means |
|---|---|---|---|
| VoIP recovery | **+3.82 dB** | +3.09 dB | dxRevive still ahead, but not by much |
| speech band 1–4 kHz | +0.66 | **+1.00** | LavaSR leaves the speaker's own waveform alone; dxRevive rewrites it |
| clean material | **−3.36 dB** | −5.87 dB | LavaSR does more harm when nothing needed fixing |
| denoising (hiss) | **+9.71 dB** | +6.93 dB | LavaSR is not a denoiser, and does not claim to be |
| non-speech, 0.5–2 kHz | +56.85 dB | **−0.01 dB** | dxRevive deletes it; LavaSR leaves it |
| speed | 7× realtime | **77×** | eleven times faster, on CPU, no GPU involved |

Two findings worth stating plainly. **LavaSR preserves the speech band
completely** — its `origin` of +1.00 through 1–4 kHz is not an approximation,
it is the untouched input, because the model's output is crossfaded in above
4 kHz and below that your waveform survives. And **it does not destroy
non-speech**: a sweep loses 0.01 dB where dxRevive removes 57. For a
recording with a room in it, that is a different tool, not a slightly
different score.

Its `denoise` option is currently harmful on this material — floor −36.9 dB
and *speech* −5.1 dB, recovering nothing. Left off by default and not
recommended until someone works out why.

The shape this suggests is a chain: denoise with something built for it, then
extend the band with LavaSR. That is [issue #5](https://github.com/ollisulopuisto/earshot/issues/5).

## Results

Measured results are committed under `results/` and summarised by
`earshot scoreboard`, so a new engine's numbers land next to every earlier
one rather than in a comment nobody can find again. Each file records the
earshot version and the machine, because `throughput` means nothing without
the latter and a probe changing its definition is the obvious way for two
numbers to stop being comparable without anyone noticing.

## State

Early. The bench runs, the probes are tested, and the only engines wired up are
the passthrough control and any VST3/AU plug-in. Open models come next — which
is the whole point of building the measuring stick first.

Apache-2.0. Contributions and agents both welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).
