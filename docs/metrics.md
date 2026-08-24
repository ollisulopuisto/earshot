# Why the metrics disagree, and which one to believe

The bench reports numbers that contradict each other on purpose. This page
is the evidence for why, measured on real podcast material: six 10-second
excerpts, two speakers, dxRevive 1.1.1 at `mix=100`.

| damage | `recovery/gained` | PESQ | STOI |
|---|---|---|---|
| clean | −3.45 dB | −1.89 | −0.07 |
| hiss | **+8.65 dB** | **+0.75** | −0.04 |
| narrowband-voip | **+2.84 dB** | **−0.83** | +0.03 |

Three damages, three different stories.

## On noise, everything agrees

`hiss` is the easy case. The engine removes something that was added, the
result is closer to the original by every measure, and PESQ rises. Any metric
would have picked the right engine here.

## On band-limiting, PESQ says the opposite of everything else

`narrowband-voip` removes everything above 3.4 kHz. dxRevive puts content
back: our log-spectral distance to the untouched original improves by
2.84 dB, and intelligibility rises slightly. **PESQ drops by 0.83.**

PESQ is not broken. It is doing exactly what it was designed to do — measure
how much a degraded signal deviates from a reference — and generated
high frequencies deviate from the reference enormously. They are plausible
speech, they are not *that* speech. The `origin` probe says so directly: in
the 6–12 kHz band the correlation between dxRevive's output and its input is
**+0.03**. That content is invented, and PESQ scores invention as distortion.

**This is the trap.** Every candidate model's paper leads with PESQ. Select
by published PESQ and you will systematically pick the *least* generative
model — which for a band-limited remote guest is precisely the wrong one,
because there is nothing left in the recording to filter and only invention
will help.

## On clean material, everything agrees again — that it does harm

`clean` is the control, and it is the reading nobody publishes. Given
material that needed nothing, dxRevive moves it 3.45 dB *away* from where it
started and costs 1.89 PESQ. That is not a criticism of the plug-in; it is
what a restoration model does when there is nothing to restore. It is also
the single most useful number in the bench for a podcast with good local
microphones, and the reason the `clean` recipe exists.

## How to read the set

- **`recovery/gained`** — did it get back what the damage took? Trust it when
  the damage is known, which in this bench it always is.
- **`origin`** — filtered or invented? Not a score. It is the number that
  tells you *why* PESQ and `gained` disagree, and the one that says whether a
  speaker still sounds like themselves.
- **PESQ** — how the literature will score it. Useful for holding a paper's
  claim against your own material; actively misleading as a selection
  criterion for bandwidth extension.
- **STOI** — intelligibility. Moves very little on podcast speech, which is
  already intelligible. Expect it to say nothing, and be suspicious when it
  says a lot.
- **`cleanup`** — the floor down without the speech down. The most reliable
  proxy for "sounds cleaner" that does not need a reference at all, which
  means it also works on real bad recordings the bench never damaged.
- **`preservation`** — what it does to non-speech. Not a quality metric; a
  warning label.

## The spread matters as much as the mean

Every table cell from a corpus run carries `±`. `recovery/gained` on
`narrowband-voip` is `+2.84 ±2.05` — the engine helps a great deal on some
excerpts and barely at all on others. An engine that wins on average and
loses badly on one speaker is a different proposition from one that wins
everywhere, and a single number cannot tell them apart.

Reproduce with:

```bash
earshot bench --input /path/to/recordings --seconds 10 --excerpts 6 \
              --damage clean --damage hiss --damage narrowband-voip \
              --engine passthrough \
              --engine "vst3:/Library/Audio/Plug-Ins/VST3/Accentize-dxRevive.vst3?mix=100" \
              --out out/
```
