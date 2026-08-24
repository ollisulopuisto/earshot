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

earshot damages    # what the recipes are
```

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

## Candidates being evaluated

| engine | approach | licence | status |
|---|---|---|---|
| [LavaSR](https://github.com/ysharma3501/LavaSR) | Vocos, bandwidth extension + denoise, 50 MB | Apache-2.0 | next up |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | real-time denoise only, very fast | MIT/Apache-2.0 | queued |
| [Sidon](https://arxiv.org/html/2509.17052v1) | w2v-BERT + HiFi-GAN resynthesis, 250M params | CC BY 4.0 | queued |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | denoiser + enhancer | MIT | queued |

## State

Early. The bench runs, the probes are tested, and the only engines wired up are
the passthrough control and any VST3/AU plug-in. Open models come next — which
is the whole point of building the measuring stick first.

Apache-2.0. Contributions and agents both welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`AGENTS.md`](AGENTS.md).
