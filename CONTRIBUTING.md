# Contributing

Issues, pull requests and agents are all welcome. Read
[`AGENTS.md`](AGENTS.md) first — it is short, and it holds the rules that keep
the numbers meaning anything.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest -q
earshot bench                       # synthetic material, no setup needed
```

## Good first contributions

- **Wire up an engine.** LavaSR, DeepFilterNet, Sidon and Resemble Enhance are
  all listed in the README and none is integrated yet. One file in
  `src/earshot/engines/`, one entry in the README table.
- **Bring material.** The bench is only as good as what it is run on. Real
  degraded podcast audio — with permission to publish — is more valuable than
  any code here.
- **Argue with a metric.** If a number rewards the wrong thing, say so with a
  case that demonstrates it. `test_probes.py::test_a_gain_stage_is_not_a_restoration`
  is what that argument looks like once it is won.

## What gets rejected

- Claims without measurements.
- Committed model weights, or weights taken out of a commercial product.
- Anything that changes the sample count or alignment of a signal.
- Broadening the scope to general audio. The narrowness is what makes the
  bench useful.

## Reporting a result

Open an issue with the `earshot bench` output, the material it was run on
(and whether it can be shared), and the versions involved. A result that
cannot be reproduced is a story, not a finding.
