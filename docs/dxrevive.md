# What dxRevive actually does

Accentize dxRevive is the plug-in this project is trying to replace, and it is
very good. Choosing a replacement by reputation would be guessing, so it was
characterised: first from the binary, then by measurement on real podcast
material.

Everything here was obtained by reading files that ship in the open and by
feeding the plug-in signals and measuring what came back. No protection was
circumvented and no weights were extracted for reuse — the point is to know
what a replacement has to match, not to take anything.

Version measured: **dxRevive 1.1.1 Standard** (December 2024 build), on an
Apple M2, at 48 kHz.

## Architecture, from the binary

The models are not inside the protected plug-in binary. They are separate,
unencrypted dylibs:

```
/Users/Shared/Accentize/dxRevive/Models/
  Studio_v7_e_256_070923_00.dylib    27 MB   (Sept 2023, "e_256" — 256 wide)
  Studio2_050224.dylib               47 MB   (Feb 2024, universal → ~23 MB/arch)
```

The weights sit in the `__cstring` section as base64, tagged `DXREVIVE`:
22.5 MB of base64 ≈ 17 MB binary ≈ **about 4M parameters at float32**. Small,
by 2024 standards.

The imported symbols settle the architecture:

```
_sgemv_                     matrix × vector — and no sgemm, no BNNS, no Metal,
                            no convolution kernels anywhere
_vDSP_create_fftsetup
_vDSP_fft_zrip              real FFT, in place
_vDSP_fft_zop               complex FFT
_vDSP_vmul / _vsmul / _vclr
```

So: **STFT in, per-frame dense/recurrent layers, STFT out.** The absence of
`sgemm` is not an optimisation they forgot. A per-frame recurrent network
cannot batch frames — frame *N+1* depends on frame *N*'s state — so
matrix-vector is forced by the design.

That has a measurable consequence. On an M2 the plug-in runs at **7.25×
realtime using 0.98 cores**, and it is memory-bandwidth bound: each instance
streams its own copy of the weights once per audio frame. Running six
instances in parallel yields only 2.46× total throughput, and running six
*processes* instead of six threads yields 2.16× — so the ceiling is the
hardware, not a lock. Any replacement that can batch, or that runs on the GPU
or Neural Engine, starts with an enormous structural advantage.

## Behaviour, measured

Material: a podcast microphone, 48 kHz, 20 s of speech, `mix=100` unless
stated. "Ground truth" means the untouched original before we damaged it.

### It is a mask below 12 kHz and a synthesiser above 14 kHz

Correlation between output and input, per band:

| band | correlation |
|---|---|
| 100 Hz – 1 kHz | +0.941 |
| 1 – 4 kHz | +0.646 |
| 6 – 12 kHz | +0.975 |
| **14 – 20 kHz** | **+0.083** |

Below 12 kHz the waveform survives — this is filtering. Above 14 kHz the
output is uncorrelated with the input: that content is invented. Most of the
*modification* happens in 1–4 kHz, which is where intelligibility lives.

This is the measurement that motivated the `origin` probe. Nothing about the
sound tells you which of these two things is happening.

### Noise floor down 6.4 dB, speech untouched

| | before | after |
|---|---|---|
| quietest fifth | −69.5 dBFS | **−75.9 dBFS** |
| loudest fifth | −38.2 dBFS | −38.2 dBFS |
| speech-to-silence range | 31.3 dB | **37.6 dB** |

This is the shape a good denoiser has: the floor moves, the speech does not.
An engine that lowered both by 6 dB would have turned the volume down.

### Bandwidth extension is real

Log-spectral distance to the ground truth, in the band the damage removed —
lower is better:

| input | damaged | restored |
|---|---|---|
| 8 kHz lowpass | 3.42 dB | **2.88 dB** |
| 4 kHz lowpass | 5.18 dB | **3.72 dB** |
| telephone band, 3.4 kHz | 5.64 dB | **4.02 dB** |

About 1.5 dB of recovery on a heavily band-limited input. This is the number
to beat.

### Non-speech is deleted, not cleaned

A logarithmic sweep through the plug-in:

| octave | change |
|---|---|
| 100 – 500 Hz | −6.3 dB |
| 500 – 2000 Hz | **−56.8 dB** |
| 2 – 6 kHz | **−64.6 dB** |
| 6 – 16 kHz | **−66.7 dB** |

It is a speech-only model, and it means it. For a lone voice microphone this
is correct behaviour. For a microphone that also has a room, music, a phone
ringing or a guitar in it, the content does not get restored — it disappears.
Hence the `preservation` probe.

### The mix control is not a crossfade

`mix=0` returns the input bit-exactly (−240 dBFS difference). But `mix=50` is
**not** `0.5 × dry + 0.5 × wet`: the deviation from that blend sits only
18.7 dB below the signal, so the blend happens inside the model, probably in
the spectral domain.

Worth knowing because **the factory default is `mix = 50`**. Anyone running
dxRevive without touching its parameters is getting something that is not
"half the processing" in any straightforward sense.

### The bandwidth extension is stochastic

Run the same input through twice and compare the two outputs. How far below
the signal the difference sits — higher means more repeatable:

| band | clean input | telephone-band input |
|---|---|---|
| 100 Hz – 1 kHz | +71.8 dB | +74.2 dB |
| 1 – 3.4 kHz | +40.0 dB | +23.5 dB |
| 3.4 – 8 kHz | +42.9 dB | **−1.2 dB** |
| 8 – 16 kHz | +9.7 dB | **−3.0 dB** |

Below 1 kHz it is effectively a pure function. In the bands it *invents*, the
difference between two runs is **larger than the signal itself** — the
generated content is independent noise, re-rolled every time.

So dxRevive is deterministic where it filters and a dice roll where it
generates, and the boundary moves with how much of the input survived. On a
good microphone only the air band is re-rolled; on a telephone-band input
everything above 3.4 kHz is.

Consequences worth stating:

- Processing the same file twice gives audibly different high frequencies.
  Not worse, not better — different.
- Any A/B of two settings must be repeated, or the difference measured is
  partly the dice.
- Caching a *rendered file* is fine. Caching on the assumption that the same
  input and settings reproduce the same output is not.

This is why `stability` reports per band rather than as one number, and it is
not in any published review — an aggregate figure hides it completely.

### Latency

The plug-in reports its latency and hosts that honour it see none. Processed
through pedalboard with `reset=True` the output is aligned; with
`reset=False` it comes back **4641 samples short**. That is a violation of
the engine contract and the reason `engines/vst3.py` hard-codes `reset=True`.

## What Sound on Sound and others report

The [Sound on Sound review of dxRevive Pro](https://www.soundonsound.com/reviews/accentize-dxrevive-pro)
is emphatic — *"no single processor I've used enables dialogue cleanup
anything like as well as this one"* — with three limits worth carrying into
the bench design:

- It cannot undo the over-compression VoIP platforms' auto-gain applies. This
  is why `degrade.autogain` exists as a recipe: it is the one degradation that
  destroys information rather than adding something on top, and any engine
  claiming to fix it should have to prove it.
- It can produce high-frequency artefacts on poor VoIP material, reacting to
  the effects of filtering.
- DeRoom Pro 2 is "a touch better" at reverb specifically.

Standard versus Pro: Pro adds the **Retain Character** algorithm (noise and
artefacts only, no de-reverb — more respectful of the original), **Spectral
Focus** (four bands with independent processing amounts), extra presets and
A/B memory, at roughly 3× the price and on iLok rather than a serial number.
Production Expert measured the Pro-only Studio 3 algorithm as "marginally
better than the Standard algorithms, but the difference wasn't dramatic".

Given the 1–4 kHz correlation of 0.646 above, Spectral Focus is the Pro
feature most likely to matter on good material: it is the only way to tell it
to leave the band it is rewriting most alone.

## What this means for a replacement

To be a real alternative on podcast material, an engine needs to:

1. recover **≥1.5 dB LSD** in a band removed at 4 kHz;
2. drop the noise floor **≥6 dB** while moving the speech level by **<1 dB**;
3. keep `origin` correlation **high in bands that were not damaged** — an
   engine that resynthesises undamaged speech is doing something the user did
   not ask for, however good it sounds;
4. beat **7.25× realtime**, which is a low bar for anything that can batch or
   use a GPU;
5. state clearly what it does to non-speech, because the answer is going to be
   "destroys it" and users need to know before they run it on a room mic;
6. and ideally be *deterministic where it generates*, which dxRevive is not.
   A restoration you cannot reproduce is one you cannot A/B, cannot cache on
   settings, and cannot debug.

Points 3 and 5 are where an open model most plausibly *loses* to dxRevive
despite better headline numbers, and neither appears in any published
benchmark. That is the gap this bench is for.

## Reproducing the binary findings

Everything in "Architecture, from the binary" comes from four commands. They
read files that ship in the open; nothing is patched, decrypted or extracted.

```bash
PLUGIN="/Library/Audio/Plug-Ins/VST3/Accentize-dxRevive.vst3"

# 1. Where the models actually live. They are not in the bundle: the plug-in
#    loads them at runtime, so the path only appears in a running process.
python - <<'PY'
import os, subprocess, pedalboard
pedalboard.load_plugin(os.environ["PLUGIN"])
print(subprocess.run(["vmmap", str(os.getpid())], capture_output=True, text=True).stdout)
PY
# → /Users/Shared/Accentize/dxRevive/Models/*.dylib

MODEL=/Users/Shared/Accentize/dxRevive/Models/Studio2_050224.dylib

# 2. Where the size is. __cstring holding 22 MB is not string literals.
size -m "$MODEL"

# 3. What maths it calls — this is what settles the architecture.
nm -u "$MODEL" | grep -iE "vdsp|blas|gemm|gemv|fft|bnns"

# 4. How the weights are stored (base64, tagged DXREVIVE).
otool -s __TEXT __cstring "$MODEL" | head -3
```

The behavioural findings are reproduced by the bench itself:

```bash
earshot bench --input your-voice.wav --start 300 --seconds 20 \
              --damage clean --damage hiss --damage narrowband-voip \
              --engine passthrough \
              --engine "vst3:$PLUGIN?mix=100"
```

`origin`, `cleanup`, `stability` and `preservation` in that output are the
tables above. The throughput numbers need a longer file — the plug-in's load
time dominates a short one.
