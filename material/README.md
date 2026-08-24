# Material

The bench is only as good as what it is run on, and synthetic speech
supports no claim about quality. This is where real recordings live.

## What goes in here

Committed, redistributable excerpts — a few seconds each, from speakers who
have agreed to it. Two kinds are wanted, and the first is the more valuable:

- **Good local microphones.** The bench damages these itself and measures
  recovery against them as ground truth, so every referenced probe works.
  This is the material that makes results reproducible by a stranger.
- **Genuinely bad VoIP.** No clean counterpart exists, so it supports only
  the unreferenced probes — but it is the honest check on whether the
  synthetic `narrowband-voip` recipe resembles the real thing.

Mono, 48 kHz, WAV. Name files `<speaker> <what it is>.wav`: the bench takes
the first word as the speaker and groups the spread by it.

## What does not go in here

Anything a speaker has not agreed to publish. `material/local/` is
gitignored for exactly that — point `--input` at it and nothing leaves the
machine:

```bash
earshot bench --input material/local --seconds 10 --excerpts 8 ...
```

## Consent

A voice is a person. If a recording has a guest on it, "I have the file" is
not the same as "I may publish it". Ask, and note in this file who agreed to
what.

| file | speaker | consented | notes |
|---|---|---|---|
| _(none yet)_ | | | |
